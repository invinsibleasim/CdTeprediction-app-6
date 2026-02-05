# streamlit_app.py
# CdTe EL → Power-loss Attribution (Shunt / Moisture / PID) + Simple Forecast
# Author: Asim's Copilot (M365)
#
# What it does
# ------------
# 1) Upload 2–3 EL images (preferably at different forward-bias voltages) with a single IV point (Voc, Isc, Pmax).
# 2) Computes EL residual maps and segments Shunt, Moisture-like (edge diffuse), PID-like (striping/series) patterns.
# 3) Turns the mechanism indices into a loss split in W and % of nameplate for "today" (using P_nom - Pmax).
# 4) (Optional) Forecasts PID-driven loss from site T/RH time series with a simple exp(T,RH) rate form.
#
# Notes
# -----
# - Designed for CdTe thin film modules; images must be consistent exposure/gain across the set.
# - Only NumPy/Pandas/Pillow/Matplotlib/Scipy are required (no skimage dependency).
# - If you can also supply Ns (series stripes) and per-image bias, EL slope/ideality is computed (display only).

import io
import base64
from typing import List, Tuple, Dict

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_opening, sobel, binary_erosion, binary_dilation

# -----------------------
# Constants
# -----------------------
q = 1.602176634e-19
kB = 1.380649e-23

# -----------------------
# Helpers
# -----------------------
def pil_to_float(im: Image.Image) -> np.ndarray:
    """Convert PIL image to float64 in ~[0,1]."""
    if im.mode not in ("L", "I;16", "I", "F"):
        im = im.convert("L")
    arr = np.asarray(im, dtype=np.float64)
    mx = arr.max() if arr.max() > 0 else 1.0
    return arr / mx

def crop_border(arr: np.ndarray, pct: float = 0.01) -> np.ndarray:
    """Remove a thin border (percent of size) to avoid frame artifacts."""
    h, w = arr.shape
    m_h = max(1, int(pct * h))
    m_w = max(1, int(pct * w))
    return arr[m_h:h - m_h, m_w:w - m_w]

def residual_map(core: np.ndarray, sigma_bg: float = 15.0, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Background-normalized residual: (I - B)/B."""
    bg = gaussian_filter(core, sigma=sigma_bg)
    bg = np.clip(bg, eps, None)
    resid = (core - bg) / bg
    return resid, bg

def segment_mechanisms(resid: np.ndarray,
                       scribe_assist: np.ndarray = None,
                       k_shunt: float = 1.5,
                       perim_band_frac: float = 0.08) -> Dict[str, np.ndarray]:
    """
    Rule-based segmentation:
      - Shunt: strong negative residual, cleaned morphologically, helped by scribe mask if provided.
      - Moist: diffuse perimeter deficit (negative), excluding shunts.
      - PID-like: row/striping negatives that are not explained by shunt/moist (fallback mask).
    """
    mu, sd = np.mean(resid), np.std(resid)
    thr_sh = mu - k_shunt * sd
    sh = resid < thr_sh
    # Morphological opening to remove pepper noise
    sh = binary_opening(sh, structure=np.ones((3, 3)))
    # Small object removal via erosion/dilation pair
    sh = binary_erosion(sh, structure=np.ones((2, 2)))
    sh = binary_dilation(sh, structure=np.ones((2, 2)))

    if scribe_assist is not None:
        sh = np.logical_or(sh, np.logical_and(sh, scribe_assist))

    # Moisture: perimeter band (top/bottom/left/right), negatives that are not shunts
    h, w = resid.shape
    b = int(perim_band_frac * min(h, w))
    band = np.zeros_like(resid, dtype=bool)
    band[:b, :] = True
    band[-b:, :] = True
    band[:, :b] = True
    band[:, -b:] = True

    moist_raw = (resid < (mu - 0.3 * sd)) & band & (~sh)

    # PID-like: residual negatives not claimed by shunt/moist, with focus on row-wise patterns
    pid_raw = (resid < (mu - 0.1 * sd)) & (~sh) & (~moist_raw)

    return dict(shunt=sh, moisture=moist_raw, pid=pid_raw)

def mask_color_overlay(gray: np.ndarray, masks: Dict[str, np.ndarray], alpha=0.35) -> np.ndarray:
    """RGB overlay for visualization."""
    g = (255 * np.clip(gray, 0, 1)).astype(np.uint8)
    rgb = np.repeat(g[..., None], 3, axis=2)

    # Colors
    col = {
        "shunt": (255, 0, 0),      # red
        "moisture": (0, 200, 255), # cyan
        "pid": (255, 165, 0)       # orange
    }
    out = rgb.astype(np.float32)
    for k, m in masks.items():
        if m is None or m.sum() == 0:
            continue
        color = np.array(col[k], dtype=np.float32)[None, None, :]
        mask3 = np.repeat(m[..., None], 3, axis=2)
        out[mask3] = (1 - alpha) * out[mask3] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)

def trimmed_mean_log(arr: np.ndarray, low=0.05, high=0.05, eps=1e-8) -> float:
    v = np.log(np.clip(arr, eps, None)).ravel()
    v.sort()
    n = len(v)
    lo, hi = int(n * low), int(n * (1 - high))
    if hi <= lo:
        return float(v.mean())
    return float(v[lo:hi].mean())

def severity_indices(resid: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute mechanism severity numbers for split weighting."""
    out = {}
    # Shunt severity = area * mean deficit within shunt
    sh = masks["shunt"]
    if sh.sum() > 0:
        sh_def = float(-resid[sh].mean())  # positive number
        sh_area = float(sh.mean())
        out["S_shunt"] = sh_def * sh_area
    else:
        out["S_shunt"] = 0.0

    # Moisture severity = perimeter area * (negative deficit average)
    mo = masks["moisture"]
    if mo.sum() > 0:
        mo_def = float(-resid[mo].mean())  # positive if negative residual
        mo_area = float(mo.mean())
        out["S_moist"] = mo_def * mo_area
    else:
        out["S_moist"] = 0.0

    # PID severity = row non-uniformity index (90th-10th pct of row STD)
    row_std = resid.std(axis=1)
    pid_index = float(np.percentile(row_std, 90) - np.percentile(row_std, 10))
    # subtract the part already explained by shunt/moist signals (heuristic)
    out["S_pid"] = max(0.0, pid_index)

    return out

def weight_by_bias(values: List[float], biases: List[float]) -> float:
    """Weighted average by bias voltage (higher bias slightly more weight)."""
    b = np.array(biases, dtype=float)
    v = np.array(values, dtype=float)
    w = np.clip(b - np.min(b) + 1.0, 1.0, None)  # avoid zero weights
    return float(np.sum(v * w) / np.sum(w))

def png_download_button(fig, label="Download PNG", filename="figure.png"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a download="{filename}" href="data:file/png;base64,{b64}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="CdTe EL → Power-loss Attribution", layout="wide")
st.title("CdTe EL → Power-loss Attribution (Shunt / Moisture / PID) + Forecast")

with st.expander("Method notes & references (click to expand)"):
    st.markdown(
        "- **Quantitative EL** can be correlated to performance with careful normalization and controls. "
        "(e.g., multi‑bias EL, consistent exposure, global intensity slope vs. V). [1](https://bing.com/search?q=IEC+62915%3a2023+retesting+PV+modules+test+sequences+sample+size)\n"
        "- In **CdTe**, **PID progression** often follows **shunting → junction degradation → series‑resistance‑like signatures**, "
        "and the **rate** is linked to **T** and **RH**; **in‑situ dark‑IV** methods relate to **Pmax** prediction. [2](https://cdn.standards.iteh.ai/samples/103839/d0effd6240a84174a9ffd7079becb235/IEC-TS-62915-2023.pdf)[3](https://www.pvknowhow.com/el-inspection-of-pv-modules/)\n"
        "- **Moisture (damp‑heat)**: junction/lifetime losses and the role of barrier quality. [4](https://www.singaporestandardseshop.sg/Product/GetPdf?fileName=240308223630TR%20IEC%20TS%2062915-2024%20Preview.pdf&pdtid=63e92ca5-68b9-47bf-bcef-500eb83c560a)\n"
        "- **Thin‑film EL defect atlas** shows scribe/TCO shunts and light‑soak behavior. [5](https://www.pvel.com/wp-content/uploads/PVEL-White-Paper_Mechancial-Stress-Sequence_Cracking-Down-on-PV-Module-Design.pdf)\n"
    )

with st.sidebar:
    st.header("1) Nameplate & IV")
    P_nom = st.number_input("P_nom (nameplate, W)", min_value=1.0, value=530.0, step=1.0)
    col_iv1, col_iv2 = st.columns(2)
    with col_iv1:
        Voc = st.number_input("Voc (V)", value=232.5431, step=0.001)
        Vmpp = st.number_input("Vmpp (V)", value=185.995, step=0.001)
        Ns = st.number_input("Series stripes (Ns)", value=60, min_value=1, max_value=300, step=1,
                             help="Optional, used only to display EL‑derived ideality from slope.")
    with col_iv2:
        Isc = st.number_input("Isc (A)", value=3.02288, step=0.001)
        Impp = st.number_input("Impp (A)", value=2.81519, step=0.001)

    Pmax = st.number_input("Measured Pmax (W)", value=523.6102068, step=0.001)
    st.caption(f"ΔP_total = {P_nom - Pmax:,.3f} W ({(P_nom-Pmax)/P_nom*100:.2f} % of nameplate)")

    st.header("2) EL Images & Bias Voltages")
    files = st.file_uploader("Upload 2–3 EL images (same exposure/gain)", type=["png", "jpg", "jpeg", "tif", "tiff"],
                             accept_multiple_files=True)
    default_biases = [232.0, 230.0, 225.0]
    bias_inputs = []
    if files:
        st.write("Enter the **module EL bias voltage** for each image (forward bias).")
        for i, f in enumerate(files):
            v = st.number_input(f"Bias for {f.name} (V)", value=default_biases[i] if i < len(default_biases) else 225.0,
                                step=1.0, key=f"bias_{i}")
            bias_inputs.append(v)

    st.header("3) Preprocess parameters")
    border_pct = st.slider("Border crop (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.5)
    sigma_bg = st.slider("Background blur (σ, px)", min_value=5, max_value=40, value=15, step=1)
    k_sh = st.slider("Shunt threshold (k × σ below mean)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    perim_frac = st.slider("Perimeter band for moisture (%)", min_value=2, max_value=20, value=8, step=1)

    st.header("4) Forecast (optional)")
    st.caption("Upload a CSV with columns: time, T_C, RH_percent (optionally bias_V).")
    up_fore = st.file_uploader("T/RH time series", type=["csv"], key="forecsv")
    # Rate parameters (user-tunable)
    st.write("Rate model: dP_PID/dt = A · exp(α·T + β·RH)")
    A_rate = st.number_input("A (W/hour)", min_value=0.0, value=0.0001, step=0.0001,
                             help="Base rate scale (W per hour). Set small; tune to your chamber/site data.")
    alpha_rate = st.number_input("α (1/°C)", value=0.02, step=0.005)
    beta_rate = st.number_input("β (1/%RH)", value=0.01, step=0.002)

run = st.button("Run analysis")

# -----------------------
# Core analysis
# -----------------------
if run:
    if not files or len(files) < 2:
        st.error("Please upload at least two EL images with their bias voltages.")
        st.stop()

    # Compute FF and baseline loss
    FF = (Pmax) / (Voc * Isc) if Voc > 0 and Isc > 0 else np.nan
    dP_total = P_nom - Pmax
    st.subheader("A) Baseline IV summary")
    st.write(f"- **FF** ≈ {FF:.3f}  |  **ΔP_total** = {dP_total:.3f} W ({dP_total/P_nom*100:.2f} % of nameplate)")

    # Process each image → residuals, masks, severities, slope mu(logEL)
    per_image = []
    slope_pairs = []  # (V, trimmed_mean_log)
    st.subheader("B) EL residuals, masks & overlays")
    cols = st.columns(min(3, len(files)))
    for idx, f in enumerate(files):
        with cols[idx % len(cols)]:
            try:
                arr = pil_to_float(Image.open(f))
            except Exception as e:
                st.error(f"Failed to open {f.name}: {e}")
                st.stop()

            core = crop_border(arr, pct=border_pct / 100.0)
            resid, bg = residual_map(core, sigma_bg=sigma_bg)

            # Scribe assist mask (optional simple line finder)
            # Here we keep it None to avoid overfitting; can be enabled with edge maps.
            masks = segment_mechanisms(resid, scribe_assist=None, k_shunt=k_sh, perim_band_frac=perim_frac / 100.0)
            sev = severity_indices(resid, masks)

            # Store slope pair for EL slope display (trimmed mean of log-intensity)
            slope_pairs.append((bias_inputs[idx], trimmed_mean_log(core)))

            # Overlay figure
            fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6))
            ax[0].imshow(core, cmap="gray")
            ax[0].set_title(f"{f.name} (core)")
            ax[0].axis("off")
            overlay = mask_color_overlay(core, masks, alpha=0.38)
            ax[1].imshow(overlay)
            ax[1].set_title("Shunt=Red, Moist=Cyan, PID=Orange")
            ax[1].axis("off")
            st.pyplot(fig)
            png_download_button(fig, label="Download overlay PNG", filename=f"{f.name}_overlay.png")

            # Numbers
            st.write(f"**Shunt severity S_shunt:** {sev['S_shunt']:.4f}  |  "
                     f"**Moisture S_moist:** {sev['S_moist']:.4f}  |  "
                     f"**PID S_pid:** {sev['S_pid']:.4f}")
            per_image.append(dict(name=f.name, bias=bias_inputs[idx], sev=sev))

    # C) EL slope (display only; optional ideality if Ns given)
    st.subheader("C) EL global slope (display)")
    try:
        V_arr = np.array([p[0] for p in slope_pairs], dtype=float)
        L_arr = np.array([p[1] for p in slope_pairs], dtype=float)
        A = np.vstack([np.ones_like(V_arr), V_arr]).T
        a_fit, b_fit = np.linalg.lstsq(A, L_arr, rcond=None)[0]
        st.write(f"- **log(EL) vs V fit**:  logEL ≈ {a_fit:+.4f} + {b_fit:.5f}·V")
        # per-cell slope → ideality (informal)
        T_K = 298.15  # display only; for precision, read module T
        n_EL = (q * Ns) / (b_fit * kB * T_K) if b_fit > 0 else np.nan
        st.write(f"- Display ideality estimate (using Ns={Ns}, T≈{T_K:.1f} K): **n_EL ≈ {n_EL:.2f}**")
    except Exception as e:
        st.warning(f"EL slope display failed: {e}")

    # D) Mechanism weighting across images (bias-weighted averaging)
    S_sh_list, S_mo_list, S_pid_list, B_list = [], [], [], []
    for p in per_image:
        S_sh_list.append(p["sev"]["S_shunt"])
        S_mo_list.append(p["sev"]["S_moist"])
        S_pid_list.append(p["sev"]["S_pid"])
        B_list.append(p["bias"])

    S_sh = weight_by_bias(S_sh_list, B_list)
    S_mo = weight_by_bias(S_mo_list, B_list)
    S_pid = weight_by_bias(S_pid_list, B_list)

    # Normalize to a split (guard against zero)
    eps = 1e-9
    S_sum = max(S_sh + S_mo + S_pid, eps)
    w_sh = S_sh / S_sum
    w_mo = S_mo / S_sum
    w_pid = S_pid / S_sum

    # Apply to today's ΔP_total
    dP_sh = dP_total * w_sh
    dP_mo = dP_total * w_mo
    dP_pid = dP_total * w_pid

    st.subheader("D) Power-loss split (today)")
    st.write(
        f"- **Shunt**: {dP_sh:.3f} W  ({w_sh*100:.1f} % of ΔP)  →  {dP_sh/P_nom*100:.2f} % of nameplate\n\n"
        f"- **Moisture**: {dP_mo:.3f} W  ({w_mo*100:.1f} % of ΔP)  →  {dP_mo/P_nom*100:.2f} % of nameplate\n\n"
        f"- **PID-like**: {dP_pid:.3f} W  ({w_pid*100:.1f} % of ΔP)  →  {dP_pid/P_nom*100:.2f} % of nameplate\n\n"
        f"- **Total**: {dP_total:.3f} W  (100%)"
    )

    # Stacked bar
    fig_bar, axb = plt.subplots(figsize=(5.5, 3.6))
    axb.bar(["Shunt", "Moisture", "PID-like"], [dP_sh, dP_mo, dP_pid],
            color=["tab:red", "tab:blue", "tab:orange"])
    axb.set_ylabel("Power loss (W)")
    axb.set_title("Mechanism split (today)")
    axb.grid(axis="y", alpha=0.3)
    st.pyplot(fig_bar)

    # CSV results
    df_out = pd.DataFrame({
        "Mechanism": ["Shunt", "Moisture", "PID-like", "Total"],
        "Loss_W": [dP_sh, dP_mo, dP_pid, dP_total],
        "Share_of_DeltaP_%": [w_sh * 100, w_mo * 100, w_pid * 100, 100.0],
        "Share_of_Pnom_%": [dP_sh / P_nom * 100, dP_mo / P_nom * 100, dP_pid / P_nom * 100, dP_total / P_nom * 100]
    })
    st.download_button("Download split (CSV)", data=df_out.to_csv(index=False).encode("utf-8"),
                       file_name="cdte_el_loss_split.csv", mime="text/csv")

    # -----------------------
    # Forecast (optional)
    # -----------------------
    st.subheader("E) PID forecast (optional)")
    if up_fore is not None:
        try:
            dfF = pd.read_csv(up_fore)
            # Expect at least T_C and RH_percent; time optional for plotting against index
            assert "T_C" in dfF.columns and "RH_percent" in dfF.columns
            T = dfF["T_C"].astype(float).to_numpy()
            RH = dfF["RH_percent"].astype(float).to_numpy()

            # Simple rate model: dP_PID/dt = A * exp(α T + β RH)
            # Integrate over rows as 1-hour steps unless "dt_hours" provided
            if "dt_hours" in dfF.columns:
                dt = dfF["dt_hours"].astype(float).to_numpy()
            else:
                dt = np.ones_like(T)  # assume 1 hour per row

            rate = A_rate * np.exp(alpha_rate * T + beta_rate * RH)
            dP_add = np.sum(rate * dt)  # W added by PID over horizon

            st.write(f"- **Projected PID loss over series**: **{dP_add:.2f} W** (adds to today’s PID share)")
            # Plot cumulative
            cum = np.cumsum(rate * dt)
            figF, axF = plt.subplots(figsize=(6.0, 3.6))
            axF.plot(cum, lw=2, color="tab:orange")
            axF.set_ylabel("Cumulative PID loss (W)")
            axF.set_xlabel("Index (rows in uploaded CSV)")
            axF.grid(alpha=0.3)
            st.pyplot(figF)

        except Exception as e:
            st.error(f"Forecast parsing error: {e}")
    else:
        st.info("Upload a T/RH CSV (time, T_C, RH_percent[,dt_hours]) to compute a simple PID trajectory.")

    # -----------------------
    # Decision log
    # -----------------------
    st.subheader("F) Decision log (audit‑friendly)")
    st.markdown(
        f"- Nameplate **P_nom** = **{P_nom:.1f} W**, measured **Pmax** = **{Pmax:.2f} W** → **ΔP_total = {dP_total:.2f} W**.\n"
        f"- Mechanism severities computed from **background‑normalized EL residuals**:\n"
        f"  - **Shunt severity**: area × mean deficit in shunt pixels.\n"
        f"  - **Moisture severity**: negative residuals in a **{perim_frac}% perimeter band** (excluding shunts).\n"
        f"  - **PID‑like severity**: row‑wise residual roughness (P90–P10 of row‑STD).\n"
        f"- Bias‑weighted averaging across images (weights ∝ bias−min+1)→ split weights (Shunt={w_sh:.3f}, Moist={w_mo:.3f}, PID={w_pid:.3f}).\n"
        f"- EL slope display: log(EL) ≈ {a_fit:+.4f} + {b_fit:.5f}·V; Ns={Ns} → n_EL≈{n_EL:.2f} (display only).\n"
        f"- **Forecast** uses dP_PID/dt = A·exp(αT+βRH); tune (A, α, β) to your chamber/site data (as in CdTe PID studies)."
    )
