# streamlit_app.py
# CdTe EL → Power-loss Attribution (Shunt / Moisture / PID) + Simple Forecast
# Author: Asim's Copilot (M365)
#
# What it does
# ------------
# 1) Upload 2–3 EL images (preferably at different forward-bias voltages) with one IV point (Voc, Isc, Pmax, Vmpp, Impp).
# 2) Computes EL residual maps and segments Shunt, Moisture-like (edge diffuse), and PID-like (striping/series) patterns.
# 3) Converts mechanism indices into a loss split in W and % of nameplate for "today": ΔP_total = P_nom - Pmax.
# 4) (Optional) Forecasts PID-driven loss from site T/RH time series using a simple exp(T,RH) rate model.
#
# Design choices
# --------------
# - Dependency-light: NumPy / Pandas / Pillow / Matplotlib / SciPy (ndimage) only.
# - Robust overlays: fixed mask_color_overlay without 3-D boolean-indexing assignment.
# - EL slope (display-only): shows log(EL) vs V and an ideality estimate if Ns is supplied (not used in loss split).
# - All results are transparent and auditable in the “Decision log”.

from __future__ import annotations

import io
from typing import List, Tuple, Dict

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    binary_opening,
    binary_erosion,
    binary_dilation,
    sobel,
)

# -----------------------
# Constants
# -----------------------
q = 1.602176634e-19
kB = 1.380649e-23

# -----------------------
# Streamlit page setup
# -----------------------
st.set_page_config(page_title="CdTe EL → Power-loss Attribution", layout="wide")
st.title("CdTe EL → Power‑loss Attribution (Shunt / Moisture / PID) + Forecast")

with st.expander("Method notes (click to expand)"):
    st.markdown(
        "- Quantitative EL can correlate with performance when acquisition is consistent and slopes are handled carefully "
        "(e.g., multi‑bias EL, consistent exposure/gain).\n"
        "- In CdTe, PID often progresses **shunt → junction degradation → series‑resistance‑like** and is accelerated by "
        "**temperature/humidity**; in‑situ dark‑IV concepts are commonly used for Pmax tracking/prediction.\n"
        "- Moisture (damp‑heat) mainly reduces junction lifetime; robust barriers mitigate loss.\n"
        "- Thin‑film EL catalogs document scribe/TCO shunts and light‑soak changes.\n"
    )

# -----------------------
# Helpers
# -----------------------
def pil_to_float(im: Image.Image) -> np.ndarray:
    """Convert PIL image to float64 normalized to ~[0,1]."""
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
    return arr[m_h : h - m_h, m_w : w - m_w]


def residual_map(core: np.ndarray, sigma_bg: float = 15.0, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Background-normalized residual: (I - B)/B."""
    bg = gaussian_filter(core, sigma=sigma_bg)
    bg = np.clip(bg, eps, None)
    resid = (core - bg) / bg
    return resid, bg


def segment_mechanisms(
    resid: np.ndarray,
    scribe_assist: np.ndarray | None = None,
    k_shunt: float = 1.5,
    perim_band_frac: float = 0.08,
) -> Dict[str, np.ndarray]:
    """
    Rule-based segmentation:
      - Shunt: strong negative residual (mu - k_shunt*sd), cleaned morphologically; optional scribe assistance.
      - Moist: negative residuals in a perimeter band (diffuse edge), excluding shunts.
      - PID-like: residual negatives not covered by shunt/moist, emphasizing row-wise non-uniformity regions.
    """
    mu, sd = float(np.mean(resid)), float(np.std(resid))
    thr_sh = mu - k_shunt * sd

    # Shunts
    sh = resid < thr_sh
    sh = binary_opening(sh, structure=np.ones((3, 3)))
    sh = binary_erosion(sh, structure=np.ones((2, 2)))
    sh = binary_dilation(sh, structure=np.ones((2, 2)))
    if scribe_assist is not None:
        sh = np.logical_or(sh, np.logical_and(sh, scribe_assist))

    # Moisture (perimeter band, negative residuals)
    h, w = resid.shape
    band = np.zeros_like(resid, dtype=bool)
    b = int(perim_band_frac * min(h, w))
    if b > 0:
        band[:b, :] = True
        band[-b:, :] = True
        band[:, :b] = True
        band[:, -b:] = True
    moist_raw = (resid < (mu - 0.3 * sd)) & band & (~sh)

    # PID-like: fallback negative residuals not in shunt/moist
    pid_raw = (resid < (mu - 0.1 * sd)) & (~sh) & (~moist_raw)

    return {"shunt": sh, "moisture": moist_raw, "pid": pid_raw}


def mask_color_overlay(gray: np.ndarray, masks: dict, alpha: float = 0.35) -> np.ndarray:
    """
    Create an RGB overlay with per-channel blending using a 2-D mask to avoid
    boolean-indexing shape mismatches.

    Colors:
      - shunt:    red
      - moisture: cyan
      - pid:      orange
    """
    g = (255 * np.clip(gray, 0, 1)).astype(np.uint8)
    out = np.repeat(g[..., None], 3, axis=2).astype(np.float32)

    colors = {
        "shunt": (255.0, 0.0, 0.0),
        "moisture": (0.0, 200.0, 255.0),
        "pid": (255.0, 165.0, 0.0),
    }

    for k, m in masks.items():
        if m is None:
            continue
        m = m.astype(bool)
        if not m.any():
            continue
        c = colors[k]
        # Per-channel blending
        ch0 = out[..., 0]
        ch0[m] = (1 - alpha) * ch0[m] + alpha * c[0]
        out[..., 0] = ch0

        ch1 = out[..., 1]
        ch1[m] = (1 - alpha) * ch1[m] + alpha * c[1]
        out[..., 1] = ch1

        ch2 = out[..., 2]
        ch2[m] = (1 - alpha) * ch2[m] + alpha * c[2]
        out[..., 2] = ch2

    return np.clip(out, 0, 255).astype(np.uint8)


def save_fig_to_bytes(fig) -> bytes:
    """Return PNG bytes for a Matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def trimmed_mean_log(arr: np.ndarray, low: float = 0.05, high: float = 0.05, eps: float = 1e-8) -> float:
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
    # Shunt severity = area × mean deficit (positive number)
    sh = masks["shunt"]
    if sh.sum() > 0:
        sh_def = float(-resid[sh].mean())
        sh_area = float(sh.mean())
        out["S_shunt"] = sh_def * sh_area
    else:
        out["S_shunt"] = 0.0

    # Moisture severity = perimeter area × (negative deficit average → positive)
    mo = masks["moisture"]
    if mo.sum() > 0:
        mo_def = float(-resid[mo].mean())
        mo_area = float(mo.mean())
        out["S_moist"] = mo_def * mo_area
    else:
        out["S_moist"] = 0.0

    # PID severity = row non-uniformity index (P90–P10 of row-wise STD); map later with bias weights
    row_std = resid.std(axis=1)
    pid_index = float(np.percentile(row_std, 90) - np.percentile(row_std, 10))
    out["S_pid"] = max(0.0, pid_index)

    return out


def weight_by_bias(values: List[float], biases: List[float]) -> float:
    """Weighted average by bias voltage (higher bias gets slightly more weight)."""
    b = np.array(biases, dtype=float)
    v = np.array(values, dtype=float)
    w = np.clip(b - np.min(b) + 1.0, 1.0, None)  # avoid zero weights
    return float(np.sum(v * w) / np.sum(w))


# -----------------------
# Sidebar inputs
# -----------------------
with st.sidebar:
    st.header("1) Nameplate & IV")
    P_nom = st.number_input("P_nom (nameplate, W)", min_value=1.0, value=530.0, step=1.0)
    c1, c2 = st.columns(2)
    with c1:
        Voc = st.number_input("Voc (V)", value=232.5431, step=0.001)
        Vmpp = st.number_input("Vmpp (V)", value=185.995, step=0.001)
        Ns = st.number_input(
            "Series stripes (Ns)",
            value=60,
            min_value=1,
            max_value=300,
            step=1,
            help="Optional; used only for EL‑derived ideality display.",
        )
    with c2:
        Isc = st.number_input("Isc (A)", value=3.02288, step=0.001)
        Impp = st.number_input("Impp (A)", value=2.81519, step=0.001)

    Pmax = st.number_input("Measured Pmax (W)", value=523.6102068, step=0.001)
    st.caption(f"ΔP_total = {P_nom - Pmax:,.3f} W ({(P_nom - Pmax)/P_nom*100:.2f} % of nameplate)")

    st.header("2) EL Images & Bias Voltages")
    files = st.file_uploader(
        "Upload 2–3 EL images (same exposure/gain)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )
    default_biases = [232.0, 230.0, 225.0]
    bias_inputs = []
    if files:
        st.write("Enter the **module EL bias voltage** for each image (forward bias).")
        for i, f in enumerate(files):
            v = st.number_input(
                f"Bias for {f.name} (V)",
                value=default_biases[i] if i < len(default_biases) else 225.0,
                step=1.0,
                key=f"bias_{i}",
            )
            bias_inputs.append(v)

    st.header("3) Preprocessing")
    border_pct = st.slider("Border crop (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.5)
    sigma_bg = st.slider("Background blur (σ, px)", min_value=5, max_value=40, value=15, step=1)
    k_sh = st.slider("Shunt threshold (k × σ below mean)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    perim_frac = st.slider("Perimeter band for moisture (%)", min_value=2, max_value=20, value=8, step=1)

    st.header("4) Forecast (optional)")
    st.caption("Upload a CSV with columns: time, T_C, RH_percent (optionally dt_hours).")
    up_fore = st.file_uploader("T/RH time series CSV", type=["csv"], key="forecsv")
    # Rate parameters (tune for your chamber/site)
    st.write("Rate model: dP_PID/dt = A · exp(α·T + β·RH)")
    A_rate = st.number_input("A (W/hour)", min_value=0.0, value=0.0001, step=0.0001)
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

    # A) Baseline IV summary
    FF = (Pmax) / (Voc * Isc) if Voc > 0 and Isc > 0 else np.nan
    dP_total = P_nom - Pmax
    st.subheader("A) Baseline IV summary")
    st.write(f"- **FF** ≈ {FF:.3f}  |  **ΔP_total** = {dP_total:.3f} W ({dP_total/P_nom*100:.2f} % of nameplate)")

    # B) Process images → residuals, masks, severities, slope pairs
    per_image = []
    slope_pairs: list[tuple[float, float]] = []  # (V, trimmed_mean_log)
    st.subheader("B) EL residuals, masks & overlays")
    grid_cols = st.columns(min(3, len(files)))

    for idx, f in enumerate(files):
        with grid_cols[idx % len(grid_cols)]:
            try:
                arr = pil_to_float(Image.open(f))
            except Exception as e:
                st.error(f"Failed to open {f.name}: {e}")
                st.stop()

            core = crop_border(arr, pct=border_pct / 100.0)
            resid, bg = residual_map(core, sigma_bg=sigma_bg)

            # Segmentation (scribe assist could be added if you have a scribe map)
            masks = segment_mechanisms(resid, scribe_assist=None, k_shunt=k_sh, perim_band_frac=perim_frac / 100.0)
            sev = severity_indices(resid, masks)

            # Slope pair (display only)
            slope_pairs.append((bias_inputs[idx], trimmed_mean_log(core)))

            # Overlay plot
            fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6))
            ax[0].imshow(core, cmap="gray")
            ax[0].set_title(f"{f.name} (core)")
            ax[0].axis("off")
            overlay = mask_color_overlay(core, masks, alpha=0.38)
            ax[1].imshow(overlay)
            ax[1].set_title("Shunt=Red, Moist=Cyan, PID=Orange")
            ax[1].axis("off")
            st.pyplot(fig)

            # Download overlay
            st.download_button(
                label=f"Download overlay PNG ({f.name})",
                data=save_fig_to_bytes(fig),
                file_name=f"{f.name}_overlay.png",
                mime="image/png",
            )

            # Numbers
            st.write(
                f"**Shunt severity** S_shunt: {sev['S_shunt']:.4f} &nbsp; | &nbsp; "
                f"**Moisture** S_moist: {sev['S_moist']:.4f} &nbsp; | &nbsp; "
                f"**PID-like** S_pid: {sev['S_pid']:.4f}"
            )

            per_image.append(dict(name=f.name, bias=bias_inputs[idx], sev=sev))

    # C) EL slope (display only)
    st.subheader("C) EL global slope (display)")
    try:
        V_arr = np.array([p[0] for p in slope_pairs], dtype=float)
        L_arr = np.array([p[1] for p in slope_pairs], dtype=float)
        A = np.vstack([np.ones_like(V_arr), V_arr]).T
        a_fit, b_fit = np.linalg.lstsq(A, L_arr, rcond=None)[0]
        T_K = 298.15  # If you logged module temperature per EL image, use that instead.
        n_EL = (q * Ns) / (b_fit * kB * T_K) if b_fit > 0 else np.nan
        st.write(f"- log(EL) ≈ {a_fit:+.4f} + {b_fit:.5f}·V  →  **n_EL ≈ {n_EL:.2f}** (Ns={Ns}, display only)")
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

    # Normalize to split weights
    eps = 1e-12
    S_sum = max(S_sh + S_mo + S_pid, eps)
    w_sh = S_sh / S_sum
    w_mo = S_mo / S_sum
    w_pid = S_pid / S_sum

    # Apply to today's ΔP_total
    dP_sh = dP_total * w_sh
    dP_mo = dP_total * w_mo
    dP_pid = dP_total * w_pid

    st.subheader("D) Power‑loss split (today)")
    st.write(
        f"- **Shunt**: {dP_sh:.3f} W  ({w_sh*100:.1f} % of ΔP)  →  {dP_sh/P_nom*100:.2f} % of nameplate\n\n"
        f"- **Moisture**: {dP_mo:.3f} W  ({w_mo*100:.1f} % of ΔP)  →  {dP_mo/P_nom*100:.2f} % of nameplate\n\n"
        f"- **PID‑like**: {dP_pid:.3f} W  ({w_pid*100:.1f} % of ΔP)  →  {dP_pid/P_nom*100:.2f} % of nameplate\n\n"
        f"- **Total**: {dP_total:.3f} W  (100%)"
    )

    # Stacked bar
    fig_bar, axb = plt.subplots(figsize=(5.6, 3.6))
    axb.bar(["Shunt", "Moisture", "PID‑like"], [dP_sh, dP_mo, dP_pid], color=["tab:red", "tab:blue", "tab:orange"])
    axb.set_ylabel("Power loss (W)")
    axb.set_title("Mechanism split (today)")
    axb.grid(axis="y", alpha=0.3)
    st.pyplot(fig_bar)

    # CSV results
    df_out = pd.DataFrame(
        {
            "Mechanism": ["Shunt", "Moisture", "PID-like", "Total"],
            "Loss_W": [dP_sh, dP_mo, dP_pid, dP_total],
            "Share_of_DeltaP_%": [w_sh * 100, w_mo * 100, w_pid * 100, 100.0],
            "Share_of_Pnom_%": [dP_sh / P_nom * 100, dP_mo / P_nom * 100, dP_pid / P_nom * 100, dP_total / P_nom * 100],
        }
    )
    st.download_button(
        "Download split (CSV)", data=df_out.to_csv(index=False).encode("utf-8"), file_name="cdte_el_loss_split.csv", mime="text/csv"
    )

    # E) Forecast (optional)
    st.subheader("E) PID forecast (optional)")
    if up_fore is not None:
        try:
            dfF = pd.read_csv(up_fore)
            if not {"T_C", "RH_percent"}.issubset(dfF.columns):
                raise ValueError("CSV must contain at least 'T_C' and 'RH_percent' columns")

            T = dfF["T_C"].astype(float).to_numpy()
            RH = dfF["RH_percent"].astype(float).to_numpy()

            # 1-hour steps by default unless dt_hours provided
            dt = dfF["dt_hours"].astype(float).to_numpy() if "dt_hours" in dfF.columns else np.ones_like(T)

            # dP_PID/dt = A * exp(α T + β RH)
            rate = A_rate * np.exp(alpha_rate * T + beta_rate * RH)
            dP_add = float(np.sum(rate * dt))  # W added by PID over horizon

            st.write(f"- **Projected additional PID loss over the uploaded series**: **{dP_add:.2f} W**")

            # Plot cumulative trajectory
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

    # F) Decision log
    st.subheader("F) Decision log (audit‑friendly)")
    st.markdown(
        f"- Nameplate **P_nom** = **{P_nom:.1f} W**, measured **Pmax** = **{Pmax:.2f} W** → **ΔP_total = {dP_total:.2f} W**.\n"
        f"- Mechanism severities computed from **background‑normalized EL residuals**:\n"
        f"  - **Shunt severity**: area × mean deficit (negative residual) within shunt mask.\n"
        f"  - **Moisture severity**: negative residuals in a **{perim_frac}%** perimeter band (excluding shunts).\n"
        f"  - **PID‑like severity**: row‑wise residual roughness (P90–P10 of row‑STD).\n"
        f"- Bias‑weighted averaging across images → split weights (Shunt={w_sh:.3f}, Moisture={w_mo:.3f}, PID‑like={w_pid:.3f}).\n"
        f"- EL slope display (log(EL) vs V): coefficients a,b; Ns={Ns} → **n_EL** shown only for reference.\n"
        f"- Forecast uses **dP_PID/dt = A·exp(αT+βRH)**; tune (A, α, β) to your chamber/site data."
    )
