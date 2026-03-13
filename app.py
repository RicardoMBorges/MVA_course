
# app.py
# Multivariate Data Analysis Course (Streamlit)
# Tabs: Import -> Preprocess -> Explore -> Model -> Validate -> Interpret
# All visualizations are Plotly: hover + zoom + downloadable as HTML 
 
import io
import json
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Multivariate Data Analysis Course",
    layout="wide",
)

# -----------------------------
# LOGOs (optional)
# -----------------------------
STATIC_DIR = Path(__file__).parent / "static"
for logo_name in ["LAABio.png"]: #"logo_massQL.png", 
    p = STATIC_DIR / logo_name
    try:
        from PIL import Image
        st.sidebar.image(Image.open(p), use_container_width=True)
    except Exception:
        pass

st.sidebar.divider()
if st.sidebar.button("🧹 Clear stored figures"):
    st.session_state["figs"] = {}
    st.sidebar.success("Stored figures cleared.")
    

# -------------------------
# Helpers
# -------------------------
def _safe_filename(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            keep.append(ch)
    out = "".join(keep).strip().replace(" ", "_")
    return out or "figure"


def fig_to_html_bytes(fig: go.Figure) -> bytes:
    # truly self-contained (bigger files, but works offline)
    html = fig.to_html(full_html=True, include_plotlyjs="inline")
    return html.encode("utf-8")


def add_download_html_button(fig: go.Figure, label: str, filename: str):
    st.download_button(
        label=label,
        data=fig_to_html_bytes(fig),
        file_name=f"{_safe_filename(filename)}.html",
        mime="text/html",
        use_container_width=True,
    )


def zip_html(figs: Dict[str, go.Figure]) -> bytes:
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, fig in figs.items():
            zf.writestr(f"{_safe_filename(name)}.html", fig_to_html_bytes(fig))
    buff.seek(0)
    return buff.read()

def impute_df_safe(
    X: pd.DataFrame,
    strategy: str = "median",
    fill_value: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    X2 = X.copy()
    X2 = X2.replace([np.inf, -np.inf], np.nan)

    all_nan = X2.isna().all(axis=0)
    dropped = all_nan[all_nan].index.tolist()
    if dropped:
        X2 = X2.loc[:, ~all_nan]

    if X2.shape[1] == 0:
        return X2, dropped

    if strategy == "constant":
        imp = SimpleImputer(strategy="constant", fill_value=fill_value)
    else:
        imp = SimpleImputer(strategy=strategy)

    X_imp = imp.fit_transform(X2.values)
    X_imp_df = pd.DataFrame(X_imp, index=X2.index, columns=X2.columns)
    return X_imp_df, dropped

def try_read_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".csv"):
        # Try common separators
        last_err = None
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, header=0)
                # must have at least 2 columns in this course format
                if df.shape[1] >= 2:
                    return df
            except Exception as e:
                last_err = e
        raise last_err or ValueError("Could not read CSV.")
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw), header=0)
    else:
        raise ValueError("Unsupported file type. Upload CSV or Excel.")


def parse_course_table(
    df: pd.DataFrame,
    row_label_col: str,
    sample_cols: List[str],
    class_row_label: Optional[str],
    feature_rows: List[str],
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Accepts course/MetaboAnalyst-like layout:
      - columns = samples
      - first column = row labels (ATTRIBUTE_class, Feature_1, Feature_2, ...)
    Returns:
      - X_df: rows=samples, cols=features  (sklearn-friendly)
      - y: optional target series aligned to samples
    """
    df2 = df.copy()

    # Normalize: ensure row label col exists
    if row_label_col not in df2.columns:
        raise ValueError(f"Row label column '{row_label_col}' not found.")

    # Set row labels as index
    df2[row_label_col] = df2[row_label_col].astype(str)
    df2 = df2.set_index(row_label_col)

    # Keep only selected sample columns
    missing_samples = [c for c in sample_cols if c not in df2.columns]
    if missing_samples:
        raise ValueError(f"Missing sample columns: {missing_samples}")

    df2 = df2[sample_cols]

    # y (class row)
    y = None
    if class_row_label:
        if class_row_label not in df2.index:
            raise ValueError(f"Class row '{class_row_label}' not found in row labels.")
        y = df2.loc[class_row_label].astype(str)
        # Remove class row from numeric block if present in feature list
        if class_row_label in feature_rows:
            feature_rows = [r for r in feature_rows if r != class_row_label]

    # Feature block
    missing_features = [r for r in feature_rows if r not in df2.index]
    if missing_features:
        raise ValueError(f"Missing feature rows: {missing_features}")

    feat_block = df2.loc[feature_rows].apply(pd.to_numeric, errors="coerce")

    # Transpose to sklearn-friendly: samples x features
    X_df = feat_block.T
    X_df.index.name = "SampleID"

    # y aligned to X_df index
    if y is not None:
        y = y.loc[X_df.index]

    return X_df, y

def numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def build_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rep = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "missing_n": df.isna().sum().values,
            "missing_%": (df.isna().mean().values * 100.0),
            "unique_n": [df[c].nunique(dropna=True) for c in df.columns],
        }
    )
    rep = rep.sort_values(["missing_%", "unique_n"], ascending=[False, True])
    return rep


# -------------------------
# State container
# -------------------------
@dataclass
class AppData:
    raw: Optional[pd.DataFrame] = None
    X_cols: Optional[List[str]] = None
    y_col: Optional[str] = None
    id_col: Optional[str] = None
    color_col: Optional[str] = None

    # processed matrices
    X_raw: Optional[pd.DataFrame] = None
    y_raw: Optional[pd.Series] = None
    meta: Optional[pd.DataFrame] = None

    X_proc: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    X_pre_scale: Optional[pd.DataFrame] = None
    raw_feature_names: Optional[List[str]] = None

    vip_df: Optional[pd.DataFrame] = None
    plsda_scores_df: Optional[pd.DataFrame] = None


if "app" not in st.session_state:
    st.session_state["app"] = AppData()

APP: AppData = st.session_state["app"]

st.sidebar.divider()
if st.sidebar.button("🧹 Clear APP data (reset preprocessing/models)"):
    st.session_state["app"] = AppData()
    APP = st.session_state["app"]
    st.sidebar.success("APP state reset.")

# Keep figures for "download all"
if "figs" not in st.session_state:
    st.session_state["figs"] = {}
FIGS: Dict[str, go.Figure] = st.session_state["figs"]


def store_fig(key: str, fig: go.Figure):
    FIGS[key] = fig

# Sample normalization utilities
def _as_numeric_df(X_df: pd.DataFrame) -> pd.DataFrame:
    return X_df.apply(pd.to_numeric, errors="coerce")


def _clean_data_like_metaboanalyst(X: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate the post-processing safety used after normalization in MetaboAnalystR:
    replace inf with NaN and keep numeric dataframe shape.
    """
    X2 = X.copy()
    X2 = X2.replace([np.inf, -np.inf], np.nan)
    return X2


def _min_nonzero_abs_div10(X: pd.DataFrame) -> float:
    arr = X.to_numpy(dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    arr = arr[arr != 0]
    if arr.size == 0:
        return 1e-12
    val = float(np.min(np.abs(arr)) / 10.0)
    if not np.isfinite(val) or val <= 0:
        return 1e-12
    return val


def quantile_normalize_rows(X: pd.DataFrame) -> pd.DataFrame:
    """
    Match the MetaboAnalystR logic used in:
        t(preprocessCore::normalize.quantiles(t(data), copy=FALSE))
    where `data` is samples x features.

    This makes the feature-value distribution identical across samples.
    """
    A = X.to_numpy(dtype=float, copy=True)
    order = np.argsort(A, axis=1)
    sorted_vals = np.take_along_axis(A, order, axis=1)

    with np.errstate(invalid="ignore"):
        mean_sorted = np.nanmean(sorted_vals, axis=0)

    mean_sorted = np.where(np.isfinite(mean_sorted), mean_sorted, 0.0)

    out = np.empty_like(A)
    np.put_along_axis(out, order, mean_sorted[None, :], axis=1)

    return pd.DataFrame(out, index=X.index, columns=X.columns)


def metaboanalyst_log10(x: pd.Series, min_val: float) -> pd.Series:
    return np.log10((x + np.sqrt(x**2 + min_val**2)) / 2.0)


def metaboanalyst_log2(x: pd.Series, min_val: float) -> pd.Series:
    return np.log2((x + np.sqrt(x**2 + min_val**2)) / 2.0)


def metaboanalyst_sqrt(x: pd.Series, min_val: float) -> pd.Series:
    return ((x + np.sqrt(x**2 + min_val**2)) / 2.0) ** 0.5


def scale_data(X: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Exact feature-wise formulas from general_norm_utils.R:
      MeanCenter: x - mean(x)
      AutoScale: (x - mean(x)) / sd(x)
      ParetoScale: (x - mean(x)) / sqrt(sd(x))
      RangeScale: (x - mean(x)) / (max(x) - min(x))
    """
    if method == "None":
        return X.copy()

    Xs = X.copy()

    if method == "MeanCenter":
        return Xs.apply(lambda col: col - col.mean(), axis=0)

    if method == "AutoScale":
        def _auto(col):
            sd = col.std(skipna=True, ddof=1)
            if not np.isfinite(sd) or sd == 0:
                return col * np.nan
            return (col - col.mean()) / sd
        return Xs.apply(_auto, axis=0)

    if method == "ParetoScale":
        def _pareto(col):
            sd = col.std(skipna=True, ddof=1)
            denom = np.sqrt(sd)
            if not np.isfinite(denom) or denom == 0:
                return col * np.nan
            return (col - col.mean()) / denom
        return Xs.apply(_pareto, axis=0)

    if method == "RangeScale":
        def _range(col):
            cmax = col.max(skipna=True)
            cmin = col.min(skipna=True)
            if pd.isna(cmax) or pd.isna(cmin) or cmax == cmin:
                return col.copy()
            return (col - col.mean()) / (cmax - cmin)
        return Xs.apply(_range, axis=0)

    raise ValueError(f"Unknown scaling method: {method}")


def sample_normalize(
    X: pd.DataFrame,
    method: str,
    sample_factor: Optional[pd.Series] = None,
    ref_sample: Optional[pd.Series] = None,
    ref_feature: Optional[str] = None,
    group_labels: Optional[pd.Series] = None,
    ref_group: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str], List[str]]:
    """
    Exact row-wise normalization behavior adapted from MetaboAnalystR's general_norm_utils.R.

    Returns:
        X_normalized, feature_to_drop_after_norm, messages
    """
    if method == "None":
        return X.copy(), None, []

    Xn = X.copy()
    msgs: List[str] = []
    feature_to_drop = None

    if method == "SpecNorm":
        if sample_factor is None:
            raise ValueError("SpecNorm requires a numeric sample-specific factor column.")
        f = pd.to_numeric(sample_factor, errors="coerce").astype(float)
        if f.isna().any():
            raise ValueError("Sample-specific factor contains missing/non-numeric values.")
        Xn = Xn.div(f.values, axis=0)
        return Xn, None, msgs

    if method == "SumNorm":
        s = Xn.sum(axis=1, skipna=True)
        Xn = Xn.mul(1000.0).div(s.replace(0, np.nan).values, axis=0)
        return Xn, None, msgs

    if method == "MedianNorm":
        m = Xn.median(axis=1, skipna=True)
        Xn = Xn.div(m.replace(0, np.nan).values, axis=0)
        return Xn, None, msgs

    if method == "SamplePQN":
        if ref_sample is None:
            raise ValueError("SamplePQN requires one reference sample.")
        ref = pd.to_numeric(ref_sample, errors="coerce").astype(float)
        if ref.shape[0] != Xn.shape[1]:
            raise ValueError("Reference sample length must equal the number of features.")
        quot = Xn.div(ref.values, axis=1)
        factors = quot.median(axis=1, skipna=True)
        Xn = Xn.div(factors.replace(0, np.nan).values, axis=0)
        return Xn, None, msgs

    if method == "GroupPQN":
        if group_labels is None:
            raise ValueError("GroupPQN requires class/group labels.")
        if ref_group is None:
            raise ValueError("GroupPQN requires selecting a reference group.")
        gl = group_labels.astype(str)
        grp_idx = gl == str(ref_group)
        if grp_idx.sum() == 0:
            raise ValueError(f"Reference group '{ref_group}' was not found.")
        ref = Xn.loc[grp_idx].mean(axis=0)
        quot = Xn.div(ref.values, axis=1)
        factors = quot.median(axis=1, skipna=True)
        Xn = Xn.div(factors.replace(0, np.nan).values, axis=0)
        return Xn, None, msgs

    if method == "CompNorm":
        if ref_feature is None:
            raise ValueError("CompNorm requires a reference feature.")
        if ref_feature not in Xn.columns:
            raise ValueError(f"Reference feature '{ref_feature}' not found.")
        ref_vals = pd.to_numeric(Xn[ref_feature], errors="coerce").astype(float)
        Xn = Xn.mul(1000.0).div(ref_vals.replace(0, np.nan).values, axis=0)
        feature_to_drop = ref_feature
        return Xn, feature_to_drop, msgs

    if method == "QuantileNorm":
        Xn = quantile_normalize_rows(Xn)
        vari = Xn.var(axis=0, skipna=True, ddof=1)
        const_cols = vari.index[(vari == 0) | (~np.isfinite(vari))].tolist()
        if const_cols:
            Xn = Xn.drop(columns=const_cols, errors="ignore")
            msgs.append(
                f"After quantile normalization, {len(const_cols)} constant feature(s) were found and deleted."
            )
        return Xn, None, msgs

    raise ValueError(f"Unknown sample normalization method: {method}")


def transform_data(X: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Transformation formulas matched to MetaboAnalystR.
    """
    if method == "None":
        return X.copy()

    Xt = X.copy()

    if method == "LogTransf":
        min_val = _min_nonzero_abs_div10(Xt)
        return Xt.apply(lambda col: metaboanalyst_log10(col, min_val), axis=0)

    if method == "Log2Transf":
        min_val = _min_nonzero_abs_div10(Xt)
        return Xt.apply(lambda col: metaboanalyst_log2(col, min_val), axis=0)

    if method == "SqrTransf":
        min_val = _min_nonzero_abs_div10(Xt)
        return Xt.apply(lambda col: metaboanalyst_sqrt(col, min_val), axis=0)

    if method == "CrTransf":
        arr = np.abs(Xt.to_numpy(dtype=float)) ** (1.0 / 3.0)
        mask_neg = Xt.to_numpy(dtype=float) < 0
        arr[mask_neg] = -arr[mask_neg]
        return pd.DataFrame(arr, index=Xt.index, columns=Xt.columns)

    if method == "VsnTransf":
        raise ValueError(
            "Exact VsnTransf from MetaboAnalystR requires limma::normalizeVSN from R. "
            "This Python-only app cannot reproduce it exactly without an R backend."
        )

    raise ValueError(f"Unknown transformation method: {method}")


def batch_align(X: pd.DataFrame, batch: Optional[pd.Series], method: str) -> pd.DataFrame:
    """
    Extra optional step for this teaching app.
    Not part of the exact MetaboAnalystR Normalization() pipeline.
    """
    if batch is None or method == "None":
        return X

    b = batch.astype(str)
    Xc = X.copy()

    if method == "Center within batch (subtract batch mean)":
        return Xc - Xc.groupby(b).transform("mean")

    if method == "Center within batch (subtract batch median)":
        return Xc - Xc.groupby(b).transform("median")

    raise ValueError(f"Unknown alignment method: {method}")

def hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    hex_color = str(hex_color).strip().lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(99,110,250,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _confidence_ellipse_from_scores(
    x: np.ndarray,
    y: np.ndarray,
    level: float = 0.95,
    n_points: int = 200,
):
    """
    Returns x/y coordinates for a confidence ellipse based on the sample
    covariance of two score vectors.

    For 95% confidence in 2D, chi-square critical value = 5.991.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 3:
        return None, None

    mean = np.array([x.mean(), y.mean()])
    cov = np.cov(np.vstack([x, y]))

    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None, None

    # chi-square critical values for 2D
    chi2_map = {
        0.90: 4.605,
        0.95: 5.991,
        0.99: 9.210,
    }
    chi2_val = chi2_map.get(level, 5.991)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if np.any(eigvals < 0):
        return None, None

    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(theta), np.sin(theta)])

    # scale by sqrt(eigenvalue * chi2)
    radii = np.sqrt(eigvals * chi2_val)
    ellipse = eigvecs @ np.diag(radii) @ circle

    ex = ellipse[0, :] + mean[0]
    ey = ellipse[1, :] + mean[1]
    return ex, ey


def add_confidence_ellipse_to_fig(
    fig: go.Figure,
    df_plot: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
    level: float = 0.95,
    color_map: Optional[Dict[str, str]] = None,
):
    """
    Adds filled confidence ellipses using the SAME explicit color map as the scatter plot.
    """
    if group_col is None or group_col not in df_plot.columns:
        return fig

    groups = df_plot[group_col].dropna().astype(str).unique()

    for grp in groups:
        sub = df_plot[df_plot[group_col].astype(str) == grp]

        if sub.shape[0] < 3:
            continue

        ex, ey = _confidence_ellipse_from_scores(
            sub[x_col].values,
            sub[y_col].values,
            level=level
        )

        if ex is None:
            continue

        base_hex = None
        if color_map is not None:
            base_hex = color_map.get(str(grp), None)

        fillcolor = hex_to_rgba(base_hex, alpha=0.18) if base_hex else "rgba(99,110,250,0.18)"

        fig.add_trace(
            go.Scatter(
                x=ex,
                y=ey,
                mode="lines",
                line=dict(width=0),
                fill="toself",
                fillcolor=fillcolor,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    return fig

    groups = df_plot[group_col].dropna().astype(str).unique()

    # Read group colors from the existing scatter traces
    color_map = {}
    for trace in fig.data:
        if getattr(trace, "name", None) in groups and hasattr(trace, "marker"):
            color_map[str(trace.name)] = trace.marker.color

    def color_to_rgba(color, alpha=0.18):
        if color is None:
            return f"rgba(99,110,250,{alpha})"

        color = str(color).strip()

        # hex color: #RRGGBB
        if color.startswith("#") and len(color) == 7:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return f"rgba({r},{g},{b},{alpha})"

        # rgb(r,g,b)
        if color.startswith("rgb(") and color.endswith(")"):
            vals = color[4:-1]
            return f"rgba({vals},{alpha})"

        # rgba(r,g,b,a)
        if color.startswith("rgba(") and color.endswith(")"):
            vals = [v.strip() for v in color[5:-1].split(",")]
            if len(vals) >= 3:
                return f"rgba({vals[0]},{vals[1]},{vals[2]},{alpha})"

        # fallback
        return f"rgba(99,110,250,{alpha})"

    for grp in groups:
        sub = df_plot[df_plot[group_col].astype(str) == grp]

        if sub.shape[0] < 3:
            continue

        ex, ey = _confidence_ellipse_from_scores(
            sub[x_col].values,
            sub[y_col].values,
            level=level
        )

        if ex is None:
            continue

        base_color = color_map.get(str(grp), None)
        fillcolor = color_to_rgba(base_color, alpha=0.18)

        fig.add_trace(
            go.Scatter(
                x=ex,
                y=ey,
                mode="lines",
                line=dict(width=0),
                fill="toself",
                fillcolor=fillcolor,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    return fig

    groups = df_plot[group_col].dropna().astype(str).unique()

    # Map group -> color from scatter traces
    color_map = {}
    for trace in fig.data:
        if trace.name in groups:
            color_map[trace.name] = trace.marker.color

    for grp in groups:

        sub = df_plot[df_plot[group_col].astype(str) == grp]

        if sub.shape[0] < 3:
            continue

        ex, ey = _confidence_ellipse_from_scores(
            sub[x_col].values,
            sub[y_col].values,
            level=level
        )

        if ex is None:
            continue

        base_color = color_map.get(grp, "rgba(0,0,200,1)")

        # convert to transparent
        if "rgba" in str(base_color):
            fillcolor = base_color.replace("1)", "0.15)")
        elif "rgb" in str(base_color):
            fillcolor = base_color.replace("rgb", "rgba").replace(")", ",0.15)")
        else:
            fillcolor = "rgba(0,0,200,0.15)"

        fig.add_trace(
            go.Scatter(
                x=ex,
                y=ey,
                mode="lines",
                line=dict(width=0),
                fill="toself",
                fillcolor=fillcolor,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    return fig

    # One ellipse per group
    groups = df_plot[group_col].dropna().astype(str).unique()

    for grp in groups:
        sub = df_plot[df_plot[group_col].astype(str) == grp]

        ex, ey = _confidence_ellipse_from_scores(
            sub[x_col].values,
            sub[y_col].values,
            level=level
        )

        if ex is None:
            continue

        fig.add_trace(
            go.Scatter(
                x=ex,
                y=ey,
                mode="lines",
                line=dict(width=0),
                fill="toself",
                fillcolor="rgba(100,100,200,0.15)",
                name=f"{grp} — {int(level*100)}%",
                hoverinfo="skip",
            )
        )

    return fig

    for grp in df_plot[group_col].dropna().astype(str).unique():
        sub = df_plot[df_plot[group_col].astype(str) == grp]
        ex, ey = _confidence_ellipse_from_scores(sub[x_col].values, sub[y_col].values, level=level)
        if ex is not None:
            fig.add_trace(
                go.Scatter(
                    x=ex,
                    y=ey,
                    mode="lines",
                    name=f"{grp} — {int(level*100)}% confidence",
                    line=dict(width=2, dash="dash"),
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
    return fig


def didactic_help(title: str, key: str, expanded: bool = False):
    text = PARAM_HELP.get(key, "No help text available.")
    with st.expander(f"Help — {title}", expanded=expanded):
        st.markdown(text)

# -------------------------
# Didactic text helpers
# -------------------------
PARAM_HELP = {
    "imputation": """
**Missing value imputation** fills in missing values so the analysis can continue.

Common options:
- **median**: robust to outliers; often a safe default
- **mean**: simple average; more sensitive to extreme values
- **most_frequent**: replaces with the most common value
- **constant (0)**: inserts zero; only appropriate when zero has real meaning

Didactic note:
Imputation does not create real information. It is only a practical strategy
to avoid losing samples/features.
""",
    "missing_thresh": """
Features with too many missing values are often unreliable.

This parameter removes variables whose missing percentage is above the threshold.
Example:
- 90 means a feature is dropped if more than 90% of its values are missing

Didactic note:
Very sparse variables may add noise and instability.
""",
    "sample_norm": """
**Sample normalization** adjusts each sample relative to itself.

Why do this?
Because different samples may differ in:
- dilution
- biomass
- injection amount
- total signal intensity

Examples:
- **SumNorm**: scales by the total signal of each sample
- **MedianNorm**: scales by the sample median
- **PQN**: adjusts using quotients relative to a reference
- **CompNorm**: uses one reference variable
- **QuantileNorm**: forces all samples to have the same distribution

Didactic note:
Normalization acts mainly at the **sample level**.
""",
    "transform": """
**Transformation** changes the numerical shape of the data.

Why do this?
Because analytical data are often:
- right-skewed
- heteroscedastic
- dominated by very large peaks

Examples:
- **LogTransf / Log2Transf**: compress large values
- **SqrTransf**: square-root transform; milder than log
- **CrTransf**: cube-root transform

Didactic note:
Transformation mainly changes the **distribution shape**.
""",
    "alignment": """
**Alignment / batch correction** reduces systematic shifts between batches.

Use this when measurements were acquired in different:
- days
- plates
- runs
- blocks
- batches

Example:
- subtract batch mean
- subtract batch median

Didactic note:
This is not the same as normalization. It tries to reduce structured technical bias.
""",
    "scaling": """
**Scaling** adjusts the relative importance of variables.

Why do this?
Because some features naturally have much larger variance or intensity than others.

Examples:
- **MeanCenter**: subtract the mean only
- **AutoScale**: mean-center and divide by standard deviation
- **ParetoScale**: softer than autoscaling
- **RangeScale**: scales by max-min range

Didactic note:
Scaling acts mainly at the **feature level**.
""",
    "drop_zero_var": """
A zero-variance feature has the same value in all samples.

Such features do not help:
- PCA
- classification
- correlation structure

So they are usually removed.
""",
    "raw_pca": """
This PCA is only for quick visual inspection of the raw data.

Important:
- no normalization
- no transformation
- no scaling
- minimal imputation only

Didactic note:
This can be misleading when variables have very different scales.
""",
    "pre_pca_projection": """
A true PCA score plot is based on:
- mean-centering
- covariance structure
- eigenvectors / loadings

Here we first show arbitrary projections so students can understand
that 'reducing to 2D' is not automatically PCA.
""",
    "pca_components": """
The number of principal components to calculate.

Each component captures a direction of variation in the data:
- PC1 explains the largest variance
- PC2 explains the next largest variance
- and so on
""",
    "plsda_components": """
The number of latent variables in PLS-DA.

Too few components:
- may underfit the class structure

Too many components:
- may overfit noise

Didactic note:
Always interpret PLS-DA together with validation.
""",
    "cv_folds": """
Cross-validation splits the data into parts.

Example:
- 5 folds means the model is trained on 4 parts and tested on 1 part,
  repeated until all parts are tested.

More folds:
- often use more training data
- but may become unstable with very small datasets
""",
    "cv_repeats": """
Repeating cross-validation with different random splits gives a more stable estimate.

Didactic note:
One single split may be lucky or unlucky.
Repeats reduce dependence on one random partition.
""",
    "logreg_C": """
**C** controls regularization strength in logistic regression.

- small C -> stronger regularization
- large C -> weaker regularization

Didactic note:
Stronger regularization helps prevent overfitting.
""",
    "max_iter": """
Maximum number of optimization iterations.

If the model does not converge, increasing this value may help.
""",
    "vip": """
VIP = Variable Importance in Projection.

In PLS-DA, VIP is often used to rank variables by overall contribution
to the latent structure related to the response.
""",

"validation_overview": """
**Validation** asks a crucial question:

> Does the model work well only on the data used to build it, or does it also work on new data?

This tab helps students understand whether the classification model is:
- reliable
- stable
- generalizable
- balanced across classes

Main ideas shown here:
- **cross-validation**
- **accuracy**
- **balanced accuracy**
- **confusion matrix**
- **ROC curve** (for binary classification)
- **classification report**

Didactic note:
A model is not good just because it fits the training data.
A good model must also perform well on data it has not seen before.
""",

"validation_cv_overview": """
**Cross-validation** is a strategy to estimate model performance on unseen data.

Instead of training and testing on the same samples, the dataset is split into parts:
- some samples are used for training
- the remaining samples are used for testing

This process is repeated several times.

### Why do this?
Because testing on the same data used for training gives an overly optimistic result.

### In this app
- **Folds** = how many parts the data is divided into
- **Repeats** = how many times the split procedure is repeated with different random shuffles

Didactic note:
Cross-validation is especially important when the dataset is small.
""",

"validation_accuracy": """
**Accuracy** is the fraction of correctly classified samples.

Formula:

Accuracy = correct predictions / total predictions

### Example
If 18 out of 20 samples are correctly classified:

Accuracy = 18 / 20 = 0.90

### Limitation
Accuracy can be misleading when classes are imbalanced.

Example:
- 90 samples from class A
- 10 samples from class B

A model that always predicts class A already gets 90% accuracy,
even though it completely fails for class B.
""",

"validation_balanced_accuracy": """
**Balanced accuracy** gives equal importance to each class.

It is especially useful when the classes are not equally represented.

### Idea
Instead of focusing only on the total number of correct predictions,
balanced accuracy averages the recall obtained in each class.

### Why is this useful?
Because it prevents the largest class from dominating the metric.

### Interpretation
- high balanced accuracy = model performs reasonably well across classes
- low balanced accuracy = one or more classes are being poorly predicted
""",

"validation_confusion_matrix": """
A **confusion matrix** shows how predicted classes compare to true classes.

Rows = true classes  
Columns = predicted classes

### Diagonal cells
These are correct predictions.

### Off-diagonal cells
These are classification errors.

### Why is this useful?
Because it shows *where* the model is making mistakes.

For example:
- class A may be confused with class B
- class C may be predicted very well
- one class may be systematically misclassified

Didactic note:
A confusion matrix is often more informative than a single performance number.
""",

"validation_roc": """
The **ROC curve** is used for **binary classification**.

ROC = Receiver Operating Characteristic

It shows the trade-off between:
- **True Positive Rate (Sensitivity)**
- **False Positive Rate**

for different classification thresholds.

### Interpretation
A model with a better ROC curve separates the two classes more effectively.

### AUC
The **Area Under the Curve (AUC)** summarizes ROC performance:

- **AUC = 0.5** → no discrimination (random-like)
- **AUC = 1.0** → perfect separation

### Important
ROC is shown here only for **binary targets**.
For multiclass data, this simple ROC view is not directly applicable.
""",

"validation_positive_class": """
In binary classification, one class is treated as the **positive class**.

This matters for:
- ROC curve
- sensitivity
- specificity
- probability interpretation

### Example
If the classes are:
- Control
- Treated

you may choose **Treated** as the positive class if that is the condition of interest.

Didactic note:
Changing the positive class changes how the ROC calculation is interpreted.
""",

"validation_classification_report": """
The **classification report** summarizes performance for each class.

Common terms:
- **Precision**: among predicted positives, how many were truly positive?
- **Recall**: among true positives, how many were recovered?
- **F1-score**: balance between precision and recall
- **Support**: number of true samples in that class

### Why is this useful?
Because two models with the same accuracy may behave very differently for specific classes.

Didactic note:
This table is very useful when one class is harder to predict than another.
""",

"validation_repeats": """
Repeating cross-validation gives a more stable estimate of performance.

Why?
Because one random split may be unusually favorable or unfavorable.

### With repeats
The model is evaluated over several different fold assignments.

This reduces dependence on a single random partition of the data.

Didactic note:
More repeats usually improve stability, but they also increase computation time.
""",
"univariate_overview": """
**Univariate analysis** looks at one variable at a time.

This is useful when you want to:
- inspect individual features
- compare distributions between groups
- test whether one feature differs significantly across groups

This tab complements multivariate analysis:
- multivariate methods look at patterns across many variables together
- univariate methods inspect each variable individually

Typical tools:
- boxplots
- strip plots / individual points
- group means and standard deviations
- t-test (2 groups)
- ANOVA (3 or more groups)
""",

"anova_help": """
**ANOVA** = Analysis of Variance

ANOVA tests whether the mean of a feature differs across multiple groups.

### Null hypothesis
All group means are equal.

### Alternative hypothesis
At least one group mean is different.

### Important
ANOVA tells you whether there is a difference somewhere,
but not exactly which groups differ from each other.

For that, post-hoc tests would be needed later.
""",

"ttest_help": """
A **t-test** compares the means of two groups.

### Null hypothesis
The two group means are equal.

### Alternative hypothesis
The two group means are different.

In this tab, the t-test is shown only when exactly 2 groups are present.
""",

"vip_univariate_help": """
**VIP ranking** comes from PLS-DA.

VIP = Variable Importance in Projection

Using VIP here is useful because:
- it prioritizes features that contributed more to multivariate class separation
- it helps connect multivariate and univariate interpretation

But remember:
- high VIP does not automatically mean statistically significant in univariate analysis
- a feature may be important multivariately but not strongly different alone
""",
}

# =====================================================
# Sidebar: Data import + column mapping (MetaboAnalyst-like)
# =====================================================
st.sidebar.title("Data Import")

uploaded = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
    help="""
Accepted format (MetaboAnalyst-like, wide):

• Columns = SAMPLES (Sample1, Sample2, ...)
• Rows = VARIABLES (Feature_1, Feature_2, ...)
• One special row (optional): ATTRIBUTE_class (labels per sample)

Example:
,Sample1,Sample2
ATTRIBUTE_class,Control,Treated
Feature_1,12.5,18.4
Feature_2,102,150
""",
)

if uploaded is not None:
    try:
        df_in = try_read_table(uploaded)
        st.session_state["raw_uploaded_df"] = df_in
        st.sidebar.success(f"Loaded: {uploaded.name}  ({df_in.shape[0]} rows × {df_in.shape[1]} cols)")
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")
        st.session_state["raw_uploaded_df"] = None

df_u = st.session_state.get("raw_uploaded_df", None)

# -------------------------
# Mapping UI for this special format
# -------------------------
if df_u is not None and not df_u.empty:
    st.sidebar.subheader("Format mapping (columns=samples, rows=variables)")

    # 1) Choose which column holds the row labels (often the first unnamed column)
    candidate_label_cols = df_u.columns.tolist()
    row_label_col = st.sidebar.selectbox(
        "Row-label column (contains ATTRIBUTE_class / Feature_1 / ...)",
        options=candidate_label_cols,
        index=0,
        help="This is usually the first column (often named 'Unnamed: 0' in CSV).",
    )

    row_labels = df_u[row_label_col].astype(str).tolist()

    # 2) Choose sample columns (default: all except row-label column)
    sample_candidates = [c for c in df_u.columns if c != row_label_col]
    sample_cols = st.sidebar.multiselect(
        "Sample columns (observations)",
        options=sample_candidates,
        default=sample_candidates,
        help="These are the columns that correspond to samples (Sample1, Sample2, ...).",
    )

    # 3) Choose the classification row (optional)
    default_class = "ATTRIBUTE_class" if "ATTRIBUTE_class" in row_labels else None
    class_row_label = st.sidebar.selectbox(
        "Classification row (optional)",
        options=["(none)"] + row_labels,
        index=(row_labels.index(default_class) + 1) if default_class else 0,
        help="Pick the row that contains group/class labels per sample (e.g., ATTRIBUTE_class).",
    )
    if class_row_label == "(none)":
        class_row_label = None

    # 4) Choose feature rows (data block)
    default_feature_rows = [r for r in row_labels if r != class_row_label]
    feature_rows = st.sidebar.multiselect(
        "Feature rows (variables used as X)",
        options=row_labels,
        default=default_feature_rows,
        help="Pick the rows that represent numeric features (Feature_1, Feature_2, ...).",
    )

    # 5) Parse + store into APP.* variables (sklearn-friendly orientation)
    if st.sidebar.button("Apply mapping", type="primary"):
        try:
            X_df, y = parse_course_table(
                df=df_u,
                row_label_col=row_label_col,
                sample_cols=sample_cols,
                class_row_label=class_row_label,
                feature_rows=feature_rows,
            )

            # Store in your app state (rows=samples)
            APP.raw = X_df.reset_index()  # includes SampleID
            APP.id_col = "SampleID"

            if y is not None:
                APP.raw["ATTRIBUTE_class"] = y.values
                APP.y_col = "ATTRIBUTE_class"
                APP.color_col = "ATTRIBUTE_class"
            else:
                APP.y_col = None
                APP.color_col = None

            APP.X_cols = [c for c in APP.raw.columns if c not in {"SampleID", "ATTRIBUTE_class"}]

            st.sidebar.success(f"Mapped OK: {X_df.shape[0]} samples × {X_df.shape[1]} features")

            # Reset downstream state (new mapping => preprocessing must be rerun)
            APP.X_proc = None
            APP.feature_names = None
            APP.X_pre_scale = None
            st.session_state["preprocess_ran"] = False

        except Exception as e:
            st.sidebar.error(f"Mapping failed: {e}")

    with st.sidebar.expander("Quick diagnostics", expanded=False):
        st.write("Detected row labels:", len(row_labels))
        st.write("Selected samples:", len(sample_cols))
        st.write("Selected features:", len(feature_rows))
        if class_row_label:
            st.write("Class row:", class_row_label)
        st.caption("After 'Apply mapping', the app will use sklearn-friendly orientation (rows=samples).")


# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(
    [
        "1) Import",
        "2) Preprocessing",
        "Pre-PCA Projection",
        "3) Exploration",
        "4) Modeling",
        "5) Validation",
        "6) Interpretation",
        "7) Univariate Analysis",
    ]
)

# -------------------------
# 1) Import
# -------------------------
with tabs[0]:
    st.header("1) Data Import")

    if APP.raw is None:
        st.info("Upload a dataset in the sidebar to begin.")
    else:
        df = APP.raw
        st.subheader("Preview")
        st.dataframe(df.head(50), use_container_width=True)
        
        st.subheader("Shape")
        st.write(f"Rows: **{df.shape[0]}**")
        st.write(f"Cols: **{df.shape[1]}**")
        
        st.subheader("Missingness report")
        rep = build_missing_report(df)
        st.dataframe(rep.head(30), use_container_width=True, height=420)

        # Build X/y/meta snapshots
        if APP.X_cols:
            APP.X_raw = df[APP.X_cols].copy()
            APP.raw_feature_names = APP.X_cols.copy()
        else:
            APP.X_raw = None
            APP.raw_feature_names = None

        if APP.y_col:
            APP.y_raw = df[APP.y_col].copy()
        else:
            APP.y_raw = None

        meta_cols = []
        if APP.id_col:
            meta_cols.append(APP.id_col)
        if APP.color_col and APP.color_col not in meta_cols:
            meta_cols.append(APP.color_col)
        if APP.y_col and APP.y_col not in meta_cols:
            meta_cols.append(APP.y_col)

        APP.meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

        st.divider()
        st.subheader("Distributions (quick view)")
        num_cols = numeric_columns(df)
        pick = st.multiselect("Pick numeric columns to visualize", num_cols, #default=num_cols[:3]
        )
        figs_local = {}
        for col in pick:
            fig = px.histogram(df, x=col, nbins=40, title=f"Histogram: {col}")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            key = f"import_hist_{col}"
            store_fig(key, fig)
            add_download_html_button(fig, f"Download HTML: {col}", key)
            figs_local[key] = fig

        if figs_local:
            st.download_button(
                "Download ALL Import plots (ZIP of HTML)",
                data=zip_html(figs_local),
                file_name="import_plots_html.zip",
                mime="application/zip",
                use_container_width=True,
            )

    # --- IMPORT TAB: add a RAW PCA block at the end (after the download buttons) ---
    st.divider()
    st.subheader("Optional (Raw PCA)",
        help="Be carefull! This is only a general view.")
        
    
    with st.expander("Help: Raw PCA", expanded=False):
        st.markdown(PARAM_HELP["raw_pca"])
    
    with st.expander("Raw PCA (no normalization / no scaling)", expanded=False):

        if APP.X_cols and APP.raw is not None:
            df = APP.raw.copy()

            X_raw_df = df[APP.X_cols].apply(pd.to_numeric, errors="coerce")

            # --- NEW: minimal imputation for PCA feasibility ---
            miss_pct = float(X_raw_df.isna().mean().mean() * 100)
            st.caption(f"Raw PCA: overall missingness ~ {miss_pct:.1f}%")

            # Drop features that are *mostly* missing (optional but helpful)
            col_miss = X_raw_df.isna().mean()
            keep_cols = col_miss[col_miss <= 0.95].index.tolist()  # keep cols with <=95% missing
            X_raw_df = X_raw_df[keep_cols]

            # Impute remaining NaNs (median per feature) -> still "raw" scale
            X_raw_imp_df, dropped_all_nan_raw = impute_df_safe(X_raw_df, strategy="median")

            if dropped_all_nan_raw:
                st.caption(f"Raw PCA: dropped {len(dropped_all_nan_raw)} all-NaN feature(s) before imputation.")
            
            X_raw_imp = X_raw_imp_df.values

            if X_raw_imp.shape[0] < 3 or X_raw_imp.shape[1] < 2:
                st.warning("Not enough data for raw PCA (need >=3 samples and >=2 features).")
            else:
                n_comp = st.slider(
                    "Raw PCA components",
                    min_value=2,
                    max_value=min(10, X_raw_imp.shape[1]),
                    value=min(3, X_raw_imp.shape[1]),
                    key="import_raw_pca_ncomp",
                    help=PARAM_HELP["pca_components"],
                )

                pca_raw = PCA(n_components=n_comp, random_state=0)
                scores = pca_raw.fit_transform(X_raw_imp)

                scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_comp)])

                # add sample id + metadata
                if APP.id_col and APP.id_col in df.columns:
                    scores_df[APP.id_col] = df[APP.id_col].astype(str).values
                if APP.color_col and APP.color_col in df.columns:
                    scores_df[APP.color_col] = df[APP.color_col].astype(str).values
                if APP.y_col and APP.y_col in df.columns and APP.y_col not in scores_df.columns:
                    scores_df[APP.y_col] = df[APP.y_col].astype(str).values

                color_by = APP.color_col if (APP.color_col and APP.color_col in scores_df.columns) else None
                hover_cols = [c for c in scores_df.columns if not c.startswith("PC")]

                pcx = st.selectbox("X axis", [f"PC{i+1}" for i in range(n_comp)], index=0, key="import_raw_pca_x")
                pcy = st.selectbox("Y axis", [f"PC{i+1}" for i in range(n_comp)], index=1, key="import_raw_pca_y")

                fig_raw_scores = px.scatter(
                    scores_df,
                    x=pcx,
                    y=pcy,
                    color=color_by,
                    hover_data=hover_cols,
                    title=f"RAW PCA Scores (median-imputed only): {pcx} vs {pcy}",
                )
                fig_raw_scores.update_layout(dragmode="zoom")
                st.plotly_chart(fig_raw_scores, use_container_width=True, config={"displaylogo": False})

                evr = pca_raw.explained_variance_ratio_ * 100.0
                evr_df = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(n_comp)], "Explained_%": evr})
                fig_raw_evr = px.bar(evr_df, x="PC", y="Explained_%", title="RAW PCA explained variance (%)")
                st.plotly_chart(fig_raw_evr, use_container_width=True, config={"displaylogo": False})

        else:
            st.info("Select numeric feature columns (X) in the sidebar to run a raw PCA.")


# -------------------------
# 2) Preprocessing
# -------------------------
with tabs[1]:
    st.header("2) Preprocessing")

    with st.expander("What happens in preprocessing?", expanded=False):
        st.info(
        """
Preprocessing prepares the raw analytical matrix for multivariate analysis.

Typical goals:
- handle missing values
- reduce technical bias between samples
- stabilize variance
- put variables on a comparable scale

A useful mental model:
- **Normalization** = makes samples more comparable
- **Transformation** = changes distribution shape
- **Scaling** = changes variable weighting
"""
        )
    
    with st.expander("How should I choose a normalization method?", expanded=False):
        st.markdown(
        """
- **None**: use when data are already comparable or for teaching contrasts
- **SumNorm**: useful when total signal differs strongly among samples
- **MedianNorm**: similar idea, but often more robust
- **SamplePQN**: useful when one reference sample is appropriate
- **GroupPQN**: useful when one class/group should define the reference
- **CompNorm**: useful when one internal standard or marker is trusted
- **QuantileNorm**: forces distributions to match exactly; powerful but strong
- **SpecNorm**: use when you have an external numeric correction factor
"""
        )
        
        
        

    if APP.raw is None or APP.X_raw is None or not APP.X_cols:
        st.info("Select numeric feature columns (X) in the sidebar.")
    else:
        df_full = APP.raw.copy()
        X_df = _as_numeric_df(APP.X_raw.copy())  # samples x features (raw numeric)

        st.subheader("Preprocessing choices")
        
        st.markdown(
    """
**Quick interpretation**
- Imputation answers: *what do we do with missing values?*
- Normalization answers: *how do we make samples comparable?*
- Transformation answers: *how do we reduce skew and variance problems?*
- Scaling answers: *how much weight should each variable have?*
"""
        )

        # ---------------------------------------
        # Controls (organized)
        # ---------------------------------------
        c1, c2, c3 = st.columns(3)

        with c1:
            impute_strategy = st.selectbox(
                "Missing value imputation",
                ["median", "mean", "most_frequent", "constant (0)"],
                index=0,
                help=PARAM_HELP["imputation"],
                key="impute_strategy_preprocess",
            )
        
            missing_col_thresh = st.slider(
                "Drop features with missing % above",
                0, 100, 90,
                help=PARAM_HELP["missing_thresh"],
                key="missing_col_thresh_preprocess",
            )

        with c2:
            sample_norm = st.selectbox(
                "Sample normalization",
                [
                    "None",
                    "QuantileNorm",
                    "GroupPQN",
                    "SamplePQN",
                    "CompNorm",
                    "SumNorm",
                    "MedianNorm",
                    "SpecNorm",
                ],
                index=0,
                help=PARAM_HELP["sample_norm"],
                key="sample_norm_preprocess",
            )

            transform = st.selectbox(
                "Data transformation",
                [
                    "None",
                    "LogTransf",
                    "Log2Transf",
                    "SqrTransf",
                    "CrTransf",
                    "VsnTransf",
                ],
                index=0,
                help=PARAM_HELP["transform"],
                key="transform_preprocess",
            )

        with c3:
            alignment = st.selectbox(
                "Alignment / batch correction",
                [
                    "None",
                    "Center within batch (subtract batch mean)",
                    "Center within batch (subtract batch median)",
                ],
                index=0,
                help=PARAM_HELP["alignment"],
                key="alignment_preprocess",
            )

            scaling = st.selectbox(
                "Scaling",
                ["None", "MeanCenter", "AutoScale", "ParetoScale", "RangeScale"],
                index=2,
                help=PARAM_HELP["scaling"],
            )

            drop_zero_var = st.checkbox(
                "Drop zero-variance features",
                value=True,
                help=PARAM_HELP["drop_zero_var"],
                key="drop_zero_var_preprocess",
            )

        st.divider()
        st.subheader("Extra parameters (only when needed)")

        st.caption(
        """
        These options appear only for specific methods.

        Examples:
        - **SpecNorm** needs a sample-specific factor
        - **SamplePQN** needs a reference sample
        - **GroupPQN** needs a class/group and a reference group
        - **CompNorm** needs a reference feature
        - **Alignment** needs a batch column
        """
        )

        # Identify metadata candidates in APP.raw (anything not in X_cols)
        meta_candidates = [c for c in df_full.columns if c not in (APP.X_cols or [])]

        # sample-specific factor column (weight/volume/etc.)
        factor_col = None
        if sample_norm == "SpecNorm":
            num_meta = [c for c in meta_candidates if pd.api.types.is_numeric_dtype(df_full[c])]
            factor_col = st.selectbox(
                "Factor column (numeric, same rows as samples)",
                options=["(select)"] + num_meta,
                index=0,
                help="Example: sample weight, dilution factor, volume, biomass, etc.",
            )
            if factor_col == "(select)":
                factor_col = None

        # PQN reference sample selection
        ref_sample_id = None
        if sample_norm == "SamplePQN":
            if APP.id_col and APP.id_col in df_full.columns:
                ref_sample_id = st.selectbox(
                    "Reference sample (for PQN)",
                    options=df_full[APP.id_col].astype(str).tolist(),
                    index=0,
                )
            else:
                st.warning(
                    "PQN needs SampleID available (APP.id_col). Add/keep SampleID in your mapped data."
                )

        # group PQN requires y / class column
        group_labels = None
        ref_group = None
        if sample_norm == "GroupPQN":
            if APP.y_col and APP.y_col in df_full.columns:
                group_labels = df_full[APP.y_col].astype(str)
                ref_group = st.selectbox(
                    "Reference group for GroupPQN",
                    options=sorted(group_labels.dropna().astype(str).unique().tolist()),
                    index=0,
                    help="MetaboAnalystR GroupPQN uses the mean profile of one reference group.",
                )
            else:
                st.warning("GroupPQN requires a group/class column (APP.y_col).")

        # reference feature normalization
        ref_feature = None
        if sample_norm == "CompNorm":
            ref_feature = st.selectbox(
                "Reference feature (divide each sample by this feature)",
                options=["(select)"] + (APP.X_cols or []),
                index=0,
            )
            if ref_feature == "(select)":
                ref_feature = None

        # alignment requires batch column
        batch_series = None
        if alignment != "None":
            batch_col = st.selectbox(
                "Batch column (metadata)",
                options=["(select)"] + meta_candidates,
                index=0,
                help="Example: Batch, Plate, RunDay, InjectionBlock, etc.",
            )

            if batch_col != "(select)":
                batch_series = df_full[batch_col]
            else:
                st.warning("Alignment selected but no batch column chosen. Alignment will be skipped.")
                batch_series = None
                alignment = "None"

        # ---------------------------------------
        # Apply preprocessing in a transparent order
        # ---------------------------------------
        if "preprocess_ran" not in st.session_state:
            st.session_state["preprocess_ran"] = False

        run = st.button("Run preprocessing", type="primary", key="run_preprocess")
        already_done = (APP.X_proc is not None) and (APP.feature_names is not None)

        if run:
            st.session_state["preprocess_ran"] = True

        # Stop only if we have NOTHING yet and the user didn't click Run
        if (not run) and (not already_done):
            st.info("Adjust settings, then click **Run preprocessing**.")
            st.stop()

        recompute = run

        if not recompute:
            st.success(
                f"Using stored preprocessing result: {APP.X_proc.shape[0]} samples × {APP.X_proc.shape[1]} features"
            )
        else:
            # 0) Drop high-missing features (based on raw numeric)
            miss_pct = X_df.isna().mean() * 100.0
            keep_cols = miss_pct[miss_pct <= missing_col_thresh].index.tolist()
            dropped_missing = [c for c in X_df.columns if c not in keep_cols]
            X_df2 = X_df[keep_cols].copy()

            # 1) Impute (feature-wise, using selected strategy)
            if impute_strategy == "constant (0)":
                X_imp_df, dropped_all_nan_pre = impute_df_safe(
                    X_df2,
                    strategy="constant",
                    fill_value=0.0,
                )
            else:
                X_imp_df, dropped_all_nan_pre = impute_df_safe(
                    X_df2,
                    strategy=impute_strategy,
                )
            
            if dropped_all_nan_pre:
                st.warning(
                    f"Dropped {len(dropped_all_nan_pre)} feature(s) that were entirely missing before imputation."
                )

            # 2) Sample normalization (row-wise)
            sample_factor = df_full[factor_col] if factor_col else None

            ref_sample_series = None
            if sample_norm == "SamplePQN" and ref_sample_id is not None:
                idx = df_full[APP.id_col].astype(str) == str(ref_sample_id)
                if idx.sum() != 1:
                    st.error("Could not uniquely identify the reference sample for PQN.")
                    st.stop()
                ref_sample_series = X_imp_df.loc[idx].iloc[0]

            try:
                X_norm_df, feature_to_drop_after_norm, norm_msgs = sample_normalize(
                    X_imp_df,
                    method=sample_norm,
                    sample_factor=sample_factor,
                    ref_sample=ref_sample_series,
                    ref_feature=ref_feature,
                    group_labels=group_labels,
                    ref_group=ref_group,
                )
                if feature_to_drop_after_norm is not None:
                    X_norm_df = X_norm_df.drop(columns=[feature_to_drop_after_norm], errors="ignore")
                for _msg in norm_msgs:
                    st.info(_msg)
            except Exception as e:
                st.error(f"Sample normalization failed: {e}")
                st.stop()

            # 3) Data transformation (elementwise)
            try:
                X_tr_df = transform_data(X_norm_df, method=transform)
            except Exception as e:
                st.error(f"Transformation failed: {e}")
                st.stop()

            # 4) Alignment / batch correction (optional)
            try:
                X_al_df = batch_align(X_tr_df, batch=batch_series, method=alignment)
            except Exception as e:
                st.error(f"Alignment failed: {e}")
                st.stop()

            # sanitize after norm/transform/alignment (log/division may create inf/NaN)
            X_al_df = _clean_data_like_metaboanalyst(X_al_df)

            # final imputation to guarantee PCA/models never see NaN
            X_al_df, dropped_all_nan_post = impute_df_safe(X_al_df, strategy="median")
            if dropped_all_nan_post:
                st.warning(
                    f"Dropped {len(dropped_all_nan_post)} feature(s) that became entirely missing after normalization/transformation/alignment."
                )

            # 5) Drop zero-variance (AFTER final imputation is safest)
            if drop_zero_var:
                vari = X_al_df.var(axis=0, skipna=True)
                keep2 = vari[vari > 0].index.tolist()
                dropped_zero = [c for c in X_al_df.columns if c not in keep2]
                X_al_df = X_al_df[keep2]
            else:
                dropped_zero = []

            # ✅ STORE pre-scale (no scaling yet)
            APP.X_pre_scale = X_al_df.copy()

            # 6) Scaling (feature-wise; exact MetaboAnalystR formulas)
            try:
                X_scaled_df = scale_data(X_al_df, method=scaling)
            except Exception as e:
                st.error(f"Scaling failed: {e}")
                st.stop()

            X_scaled_df = _clean_data_like_metaboanalyst(X_scaled_df)

            # final imputation again because exact formulas can create NaN for zero-SD features
            X_scaled_df, dropped_all_nan_scale = impute_df_safe(X_scaled_df, strategy="median")
            if dropped_all_nan_scale:
                st.warning(
                    f"Dropped {len(dropped_all_nan_scale)} feature(s) that became entirely missing after scaling."
                )

            if X_scaled_df.shape[1] != X_al_df.shape[1]:
                st.warning(
                    f"Scaling changed the number of features: "
                    f"before scaling = {X_al_df.shape[1]}, after scaling = {X_scaled_df.shape[1]}"
                )

            X_proc = X_scaled_df.values

            # Store to app state
            APP.X_proc = np.asarray(X_proc, dtype=float)
            APP.feature_names = X_scaled_df.columns.tolist()

            st.success(f"Processed X: {APP.X_proc.shape[0]} samples × {APP.X_proc.shape[1]} features")
            st.info("Preprocessing finished. Diagnostic plots are hidden by default below.")

            if dropped_missing:
                st.warning(f"Dropped (missingness): {len(dropped_missing)} features")
            if dropped_zero:
                st.warning(f"Dropped (zero variance): {len(dropped_zero)} features")

        # ---------------------------------------
        # Visualization: before vs after
        # ---------------------------------------
        st.divider()
        show_preprocess_plots = st.checkbox(
            "Show preprocessing diagnostic plots",
            value=False,
            help="Keep this OFF for fast preprocessing. Turn it ON only when you want visual checks."
        )

        if show_preprocess_plots:
            st.subheader("Before vs After (visual checks)")
            fast_mode = st.checkbox(
                "Fast mode",
                value=True,
                help="Reduces plot complexity so the app stays responsive."
            )

            figs_local = {}

        # ======================================================
        # A) DISTRIBUTION FOR EVERY SAMPLE (across features)
        # ======================================================
        if show_preprocess_plots:
            with st.expander("Distributions: EVERY SAMPLE (across features) — raw vs processed", expanded=False):
                st.caption(
                    "For each sample, we pool all feature values and compare RAW vs PROCESSED distributions. "
                    "This is the main visual check for sample-wise normalization effects."
                )

                feat_labels = APP.feature_names

                if APP.X_proc.shape[1] != len(feat_labels):
                    st.error(
                        f"Internal mismatch: X_proc has {APP.X_proc.shape[1]} columns "
                        f"but feature_names has {len(feat_labels)} names. Please run preprocessing again."
                    )
                    st.stop()

                raw_mat = _as_numeric_df(APP.X_raw.copy()).reindex(columns=feat_labels)
                proc_mat = pd.DataFrame(APP.X_proc, index=raw_mat.index, columns=feat_labels)

                if APP.id_col and APP.id_col in df_full.columns:
                    sample_names_all = df_full[APP.id_col].astype(str).tolist()
                else:
                    sample_names_all = [f"Sample_{i}" for i in range(raw_mat.shape[0])]

                n_feats = int(len(feat_labels))
                if n_feats < 2:
                    st.warning("Not enough features to plot sample distributions.")
                else:
                    min_feat = 2
                    max_feat_allowed = min(5000, n_feats)
                    default_feat = min(50, max_feat_allowed)

                    if min_feat == max_feat_allowed:
                        max_feat = max_feat_allowed
                        st.caption(f"Max features used: {max_feat} (only option)")
                    else:
                        step_feat = 20 if (max_feat_allowed - min_feat) >= 20 else 1
                        max_feat = st.slider(
                            "Max features used for sample distributions (speed control)",
                            min_value=min_feat,
                            max_value=max_feat_allowed,
                            value=default_feat,
                            step=step_feat,
                            help="This limits how many feature columns are pooled within each sample.",
                            key="sample_overlay_maxfeat",
                        )

                    feat_use = feat_labels[:max_feat]

                    n_samples = int(raw_mat.shape[0])
                    if n_samples < 1:
                        st.warning("No samples available.")
                    else:
                        default_samples = sample_names_all[: min(12, len(sample_names_all))]
                        sample_pick = st.multiselect(
                            "Samples to show",
                            options=sample_names_all,
                            default=default_samples,
                            key="sample_overlay_pick",
                        )

                        if not sample_pick:
                            st.info("Select at least one sample.")
                        else:
                            idx_use = [i for i, s in enumerate(sample_names_all) if s in sample_pick]

                            raw_sub = raw_mat.iloc[idx_use][feat_use]
                            proc_sub = proc_mat.iloc[idx_use][feat_use]

                            raw_long = pd.DataFrame(
                                {
                                    "value": raw_sub.to_numpy().ravel(),
                                    "sample": np.repeat([sample_names_all[i] for i in idx_use], len(feat_use)),
                                    "stage": "raw",
                                }
                            )

                            proc_long = pd.DataFrame(
                                {
                                    "value": proc_sub.to_numpy().ravel(),
                                    "sample": np.repeat([sample_names_all[i] for i in idx_use], len(feat_use)),
                                    "stage": "processed",
                                }
                            )

                            df_long = pd.concat([raw_long, proc_long], ignore_index=True)
                            df_long = df_long[np.isfinite(df_long["value"].values)]

                            plot_kind = st.radio(
                                "Plot type",
                                ["Histogram", "Violin"],
                                horizontal=True,
                                key="sample_overlay_plotkind",
                            )

                            if plot_kind == "Histogram":
                                nbins = st.slider("Bins", 20, 200, 80, key="sample_overlay_bins")

                                fig = px.histogram(
                                    df_long,
                                    x="value",
                                    color="sample",
                                    facet_col="stage",
                                    barmode="overlay",
                                    nbins=nbins,
                                    histnorm="probability density",
                                    title="Sample-wise distributions (across features): RAW vs PROCESSED",
                                )
                                fig.update_layout(dragmode="zoom", height=520)
                                fig.update_traces(opacity=0.45)

                            else:
                                fig = px.violin(
                                    df_long,
                                    x="sample",
                                    y="value",
                                    color="sample",
                                    facet_col="stage",
                                    box=True,
                                    points=False,
                                    title="Sample-wise distributions (across features): RAW vs PROCESSED",
                                )
                                fig.update_layout(dragmode="zoom", height=520)

                            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                            key = "preprocess_sample_overlay_raw_vs_processed"
                            store_fig(key, fig)
                            add_download_html_button(fig, "Download HTML: sample distributions (raw vs processed)", key)
                            figs_local[key] = fig


# -------------------------
# 2.5) Pre-PCA Projection (didactic)
# -------------------------
with tabs[2]:
    st.header("1.5) Pre-PCA Projection (Didactic)")
    st.write(
        """
A true **PCA score plot** needs PCA (eigenvectors / loadings) to be computed.

Here we do something didactic:
- We show **arbitrary 2D projections** (before PCA) using:
  - two chosen variables (Vx vs Vy), or
  - simple constructed axes (sum / mean of variable subsets)
- Then we show the **real PCA score plot** (after preprocessing, if available)
"""
    )
    
    st.info(PARAM_HELP["pre_pca_projection"])

    if APP.raw is None or APP.X_cols is None or APP.X_raw is None or len(APP.X_cols) < 2:
        st.info("Load data and map features first (Import tab). You need at least 2 numeric features.")
        st.stop()

    df_full = APP.raw.copy()
    X_raw_df = _as_numeric_df(APP.X_raw.copy()).replace([np.inf, -np.inf], np.nan)

    # NOTE: we don't rely on APP.meta for hover here; we attach hover fields directly from df_full.
    # Keep meta only for the "true PCA" score plot (optional concat).
    meta = APP.meta.copy() if APP.meta is not None else pd.DataFrame(index=df_full.index)
    meta = meta.reset_index(drop=True)
    if meta.shape[0] != df_full.shape[0]:
        meta = pd.DataFrame({"row_index": np.arange(df_full.shape[0])})

    # ======================================================
    # Controls
    # ======================================================
    st.subheader("Choose how to create the 'pre-PCA' 2D projection")

    mode = st.radio(
        "Projection mode",
        [
            "Two variables (feature vs feature)",
            "Constructed axes (sum/mean of feature subsets)",
            "Random 2D projection (linear combination)",
        ],
        horizontal=True,
        key="pre_pca_mode",
    )

    st.divider()
    st.subheader("Choose the data stage")

    stage = st.radio(
        "Data stage",
        ["RAW (as loaded)", "PRE-SCALE (after norm/transform/alignment)", "PROCESSED (after scaling)"],
        horizontal=True,
        key="pre_pca_stage",
        help=(
            "RAW = original values (may differ in scale a lot). "
            "PRE-SCALE = after preprocessing steps but before scaling. "
            "PROCESSED = after scaling (same matrix used for PCA/modeling)."
        ),
    )

    # ======================================================
    # Pick matrix (ensure consistent row identity)
    # ======================================================
    if stage.startswith("RAW"):
        # RAW may contain NaN -> minimal impute so plots never crash
        X_stage = X_raw_df.copy()

        # ✅ critical: SimpleImputer drops all-NaN columns -> must pre-drop them
        X_stage, dropped_all_nan = impute_df_safe(X_stage, strategy="median")

        # Align index to df_full (same samples)
        X_stage = X_stage.reindex(index=df_full.index)

        if dropped_all_nan:
            st.caption(f"RAW stage: dropped {len(dropped_all_nan)} all-NaN features before imputation.")

    elif stage.startswith("PRE-SCALE"):
        if APP.X_pre_scale is None:
            st.warning("PRE-SCALE matrix not found. Run preprocessing first (tab 2).")
            st.stop()

        # APP.X_pre_scale is already your "final imputed, non-scaled" matrix from preprocessing.
        # Keep it as-is to avoid re-imputing and subtly changing values again.
        X_stage = APP.X_pre_scale.copy()
        X_stage = X_stage.replace([np.inf, -np.inf], np.nan)

        # Safety only (should rarely do anything if preprocessing is correct)
        if X_stage.isna().any().any():
            X_stage = X_stage.fillna(X_stage.median(numeric_only=True))

        # Ensure row index matches df_full
        X_stage = X_stage.reset_index(drop=True)
        X_stage.index = df_full.index

    else:
        if APP.X_proc is None or APP.feature_names is None:
            st.warning("PROCESSED matrix not found. Run preprocessing first (tab 2).")
            st.stop()

        # CRITICAL FIX: use df_full.index (not X_raw_df.index) to avoid shape/index mismatch crashes
        X_stage = pd.DataFrame(APP.X_proc, columns=APP.feature_names, index=df_full.index)

    feats = X_stage.columns.tolist()

    # speed controls
    st.divider()
    fast_mode = st.checkbox("Fast mode", value=True, key="pre_pca_fast")
    max_points = 3000 if fast_mode else 10000

    # ======================================================
    # Build "pre-PCA" 2D coordinates
    # ======================================================
    coords_df = pd.DataFrame(index=df_full.index)

    if mode.startswith("Two variables"):
        c1, c2 = st.columns(2)
        with c1:
            fx = st.selectbox("X feature", feats, index=0, key="pre_pca_fx")
        with c2:
            fy = st.selectbox("Y feature", feats, index=1, key="pre_pca_fy")

        coords_df["Axis 1"] = X_stage[fx].values
        coords_df["Axis 2"] = X_stage[fy].values
        subtitle = f"Pre-PCA view = {fx} vs {fy}"

    elif mode.startswith("Constructed axes"):
        st.caption(
            "We create two simple axes using feature subsets (not PCA):\n"
            "- Axis 1 = mean (or sum) of subset A\n"
            "- Axis 2 = mean (or sum) of subset B\n"
            "This shows that dimensionality reduction can be done arbitrarily, but PCA does it optimally."
        )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            agg = st.selectbox("Aggregation", ["mean", "sum"], index=0, key="pre_pca_agg")
        with c2:
            kA = st.slider("Subset A size", 2, min(50, len(feats)), min(10, len(feats)), key="pre_pca_kA")
        with c3:
            kB = st.slider("Subset B size", 2, min(50, len(feats)), min(10, len(feats)), key="pre_pca_kB")

        subset_mode = st.radio(
            "Subset selection",
            ["First features", "Random features (seeded)"],
            horizontal=True,
            key="pre_pca_subset_mode",
        )

        if subset_mode.startswith("First"):
            A = feats[:kA]
            B = feats[kA : kA + kB] if (kA + kB) <= len(feats) else feats[-kB:]
        else:
            seed = st.number_input("Random seed", value=0, step=1, key="pre_pca_subset_seed")
            rng = np.random.default_rng(int(seed))
            pick = rng.choice(feats, size=min(kA + kB, len(feats)), replace=False).tolist()
            A = pick[:kA]
            B = pick[kA : kA + kB]

        if agg == "mean":
            coords_df["Axis 1"] = X_stage[A].mean(axis=1).values
            coords_df["Axis 2"] = X_stage[B].mean(axis=1).values
        else:
            coords_df["Axis 1"] = X_stage[A].sum(axis=1).values
            coords_df["Axis 2"] = X_stage[B].sum(axis=1).values

        subtitle = f"Pre-PCA view = {agg}(subset A) vs {agg}(subset B)"

        with st.expander("Show subsets used", expanded=False):
            st.write("Subset A:", A)
            st.write("Subset B:", B)

    else:
        st.caption(
            "Random 2D projection = linear combination of all features:\n"
            "Axis 1 = X · w1, Axis 2 = X · w2 (random weights)\n"
            "This is *not* PCA, but shows that 'projecting to 2D' is easy — PCA chooses the best projection."
        )

        seed = st.number_input("Random seed", value=0, step=1, key="pre_pca_rand_seed")
        rng = np.random.default_rng(int(seed))
        W = rng.normal(size=(len(feats), 2))
        W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)

        Z = X_stage.values @ W
        coords_df["Axis 1"] = Z[:, 0]
        coords_df["Axis 2"] = Z[:, 1]
        subtitle = "Pre-PCA view = random linear projection (2D)"

    # Attach metadata for hover + color (from df_full, same row order/index)
    if APP.id_col and APP.id_col in df_full.columns:
        coords_df[APP.id_col] = df_full[APP.id_col].astype(str).values
    if APP.y_col and APP.y_col in df_full.columns and APP.y_col not in coords_df.columns:
        coords_df[APP.y_col] = df_full[APP.y_col].astype(str).values
    if APP.color_col and APP.color_col in df_full.columns and APP.color_col not in coords_df.columns:
        coords_df[APP.color_col] = df_full[APP.color_col].astype(str).values

    hover_cols = [c for c in coords_df.columns if c not in ["Axis 1", "Axis 2"]]

    # Optional downsampling for speed (keep hover_cols consistent)
    n = coords_df.shape[0]
    if n > max_points:
        coords_df_plot = coords_df.sample(n=max_points, random_state=0)
        st.caption(f"Showing {max_points} / {n} points (downsampled for speed).")
    else:
        coords_df_plot = coords_df.copy()

    # ======================================================
    # Plot: pre-PCA projection
    # ======================================================
    st.divider()
    st.subheader("A) Pre-PCA projection (not a score plot)")

    fig_pre = px.scatter(
        coords_df_plot,
        x="Axis 1",
        y="Axis 2",
        color=(APP.color_col if (APP.color_col and APP.color_col in coords_df_plot.columns) else None),
        hover_data=[c for c in coords_df_plot.columns if c not in ["Axis 1", "Axis 2"]],
        title=f"Pre-PCA 2D projection — {stage} — {subtitle}",
    )
    fig_pre.update_layout(dragmode="zoom", height=520)
    st.plotly_chart(fig_pre, use_container_width=True, config={"displaylogo": False})

    key_pre = f"pre_pca_projection_{stage.replace(' ', '_').lower()}_{mode.split('(')[0].strip().replace(' ', '_').lower()}"
    store_fig(key_pre, fig_pre)
    add_download_html_button(fig_pre, "Download HTML: pre-PCA projection", key_pre)

    # ======================================================
    # Plot: True PCA score plot (if available)
    # ======================================================
    st.divider()
    st.subheader("B) True PCA score plot (after PCA)")

    if APP.X_proc is None or APP.feature_names is None:
        st.info("Run preprocessing first to compute the PROCESSED matrix, then go to Exploration for PCA.")
        st.stop()

    Xp = APP.X_proc
    max_pca = min(10, Xp.shape[1])
    if max_pca < 2:
        st.warning("Not enough features for PCA (need >=2).")
        st.stop()

    n_comp = st.slider(
        "PCA components (for this tab)",
        min_value=2,
        max_value=max_pca,
        value=min(3, max_pca),
        key="pre_pca_true_pca_ncomp",
    )

    pca = PCA(n_components=n_comp, random_state=0)
    scores = pca.fit_transform(Xp)

    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_comp)])

    # Optional metadata concat (already row-aligned by construction)
    if meta is not None and not meta.empty:
        scores_df = pd.concat([scores_df, meta.reset_index(drop=True)], axis=1)

    # Ensure key hover/label columns exist
    if APP.id_col and APP.id_col in df_full.columns and APP.id_col not in scores_df.columns:
        scores_df[APP.id_col] = df_full[APP.id_col].astype(str).values
    if APP.color_col and APP.color_col in df_full.columns and APP.color_col not in scores_df.columns:
        scores_df[APP.color_col] = df_full[APP.color_col].astype(str).values
    if APP.y_col and APP.y_col in df_full.columns and APP.y_col not in scores_df.columns:
        scores_df[APP.y_col] = df_full[APP.y_col].astype(str).values

    pcx = st.selectbox("X axis (true PCA)", [f"PC{i+1}" for i in range(n_comp)], index=0, key="pre_pca_true_x")
    pcy = st.selectbox("Y axis (true PCA)", [f"PC{i+1}" for i in range(n_comp)], index=1, key="pre_pca_true_y")

    color_true = APP.color_col if (APP.color_col and APP.color_col in scores_df.columns) else None
    hover_true = [c for c in scores_df.columns if not c.startswith("PC")]

    fig_true = px.scatter(
        scores_df,
        x=pcx,
        y=pcy,
        color=color_true,
        hover_data=hover_true,
        title=f"TRUE PCA scores (computed): {pcx} vs {pcy}",
    )
    fig_true.update_layout(dragmode="zoom", height=520)
    st.plotly_chart(fig_true, use_container_width=True, config={"displaylogo": False})

    key_true = "pre_pca_true_pca_scores"
    store_fig(key_true, fig_true)
    add_download_html_button(fig_true, "Download HTML: true PCA scores", key_true)

    # Explained variance
    evr = pca.explained_variance_ratio_ * 100.0
 
    evr_df = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(n_comp)], "Explained_%": evr})
    fig_evr = px.bar(evr_df, x="PC", y="Explained_%", title="PCA explained variance (%) — (computed here)")
    st.plotly_chart(fig_evr, use_container_width=True, config={"displaylogo": False})

    key_evr = "pre_pca_true_pca_explained_variance"
    store_fig(key_evr, fig_evr)
    add_download_html_button(fig_evr, "Download HTML: explained variance (this tab)", key_evr)

    # Convenience ZIP download for this tab
    st.divider()
    st.subheader("Download this tab plots (ZIP of HTML)")
    figs_local = {
        key_pre: fig_pre,
        key_true: fig_true,
        key_evr: fig_evr,
    }
    st.download_button(
        "Download Pre-PCA tab plots (ZIP)",
        data=zip_html(figs_local),
        file_name="pre_pca_tab_plots_html.zip",
        mime="application/zip",
        use_container_width=True,
    )
    
    def pca_2d_step_by_step(X_stage: pd.DataFrame, fx: str, fy: str, eps: float = 1e-12):
        """
        PCA geometry demo on 2 selected features.
        Returns:
          - df_raw: original coords
          - df_cent: mean-centered coords
          - eigvecs: 2x2 matrix (columns = PC1, PC2 directions in original space)
          - df_rot: rotated coords (PC scores in 2D)
          - evr: explained variance ratio (2,)
          - mu: mean vector (2,)
        """
        X2 = X_stage[[fx, fy]].copy()
        X2 = X2.replace([np.inf, -np.inf], np.nan)

        # minimal impute just for this 2D demo
        X2 = X2.fillna(X2.median(numeric_only=True))

        A = X2.to_numpy(dtype=float)                 # (n,2)
        mu = A.mean(axis=0)                          # (2,)
        C = A - mu                                   # centered

        # Covariance (2x2)
        cov = np.cov(C.T, bias=False)

        # Eigen-decomposition (symmetric => eigh)
        eigvals, eigvecs = np.linalg.eigh(cov)       # eigvecs columns
        idx = np.argsort(eigvals)[::-1]              # descending
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        evr = eigvals / (eigvals.sum() + eps)

        # Rotate: scores = C · eigvecs  (PC coordinates)
        S = C @ eigvecs                               # (n,2)

        df_raw = pd.DataFrame(A, columns=[fx, fy], index=X2.index)
        df_cent = pd.DataFrame(C, columns=[f"{fx}_centered", f"{fy}_centered"], index=X2.index)
        df_rot = pd.DataFrame(S, columns=["PC1", "PC2"], index=X2.index)

        return df_raw, df_cent, eigvecs, df_rot, evr, mu


    # -------------------------
    # Inside your tab (after X_stage is defined)
    # -------------------------
    st.divider()
    st.subheader("C) PCA step-by-step (2 features only — geometric view)")
    with st.expander("Show PCA geometry (mean-centering → PC axis → rotation)", expanded=False):

        if len(feats) < 2:
            st.info("Need at least 2 features.")
            st.stop()

        c1, c2 = st.columns(2)
        with c1:
            fx_demo = st.selectbox("Feature X (demo)", feats, index=0, key="pca_demo_fx")
        with c2:
            fy_demo = st.selectbox("Feature Y (demo)", feats, index=1, key="pca_demo_fy")

        df_raw2, df_cent2, eigvecs, df_rot2, evr2, mu2 = pca_2d_step_by_step(X_stage, fx_demo, fy_demo)

        # Attach class/color for plotting
        color_col = None
        if APP.color_col and APP.color_col in df_full.columns:
            color_col = APP.color_col

        # A) Original
        figA = px.scatter(
            df_raw2.reset_index(drop=True).assign(**({color_col: df_full[color_col].astype(str).values} if color_col else {})),
            x=fx_demo, y=fy_demo,
            color=color_col,
            title="A) Original 2D scatter (selected features)",
        )
        figA.update_layout(dragmode="zoom", height=420)
        st.plotly_chart(figA, use_container_width=True, config={"displaylogo": False})

        # B) Mean-centered
        xC, yC = f"{fx_demo}_centered", f"{fy_demo}_centered"
        figB = px.scatter(
            df_cent2.reset_index(drop=True).assign(**({color_col: df_full[color_col].astype(str).values} if color_col else {})),
            x=xC, y=yC,
            color=color_col,
            title="B) Mean-centered scatter",
        )

        # C) Add PC1 axis line on centered plot
        # direction of PC1 in original coords is eigvecs[:,0] in centered space too
        v = eigvecs[:, 0]
        # build a symmetric line around origin for visibility
        L = np.nanmax(np.abs(df_cent2[[xC, yC]].to_numpy())) * 1.2
        if not np.isfinite(L) or L <= 0:
            L = 1.0
        line_x = np.array([-L * v[0], L * v[0]])
        line_y = np.array([-L * v[1], L * v[1]])
        figB.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="PC1 axis"))

        figB.update_layout(dragmode="zoom", height=420)
        st.plotly_chart(figB, use_container_width=True, config={"displaylogo": False})
        st.caption(f"C) PC1 direction shown. Explained variance: PC1={evr2[0]*100:.1f}%, PC2={evr2[1]*100:.1f}%")

        # D) Rotated coordinates (scores)
        figD = px.scatter(
            df_rot2.reset_index(drop=True).assign(**({color_col: df_full[color_col].astype(str).values} if color_col else {})),
            x="PC1", y="PC2",
            color=color_col,
            title="D) Rotated axes: PCA scores for the 2-feature demo (PC1 vs PC2)",
        )
        figD.update_layout(dragmode="zoom", height=420)
        st.plotly_chart(figD, use_container_width=True, config={"displaylogo": False})


# -------------------------
# 3) Exploration (PCA, correlations)
# -------------------------
with tabs[3]:
    st.header("3) Exploration")

    if APP.X_proc is None or APP.feature_names is None:
        st.info("Run preprocessing first (tab 2).")
    else:
        X = APP.X_proc

        if APP.feature_names is None or X.shape[1] != len(APP.feature_names):
            st.error(
                f"Internal mismatch in Exploration tab: X has {X.shape[1]} columns "
                f"but feature_names has {0 if APP.feature_names is None else len(APP.feature_names)} names. "
                f"Please click '🧹 Clear APP data (reset preprocessing/models)' in the sidebar "
                f"and run preprocessing again."
            )
            st.stop()

        max_pca = min(10, X.shape[1])

        if max_pca < 2:
            st.warning(f"Not enough features for PCA (need >=2). You currently have {X.shape[1]}.")
            st.stop()
        else:
            n_comp = st.slider("PCA components", 2, max_pca, min(3, max_pca))

        pca = PCA(n_components=n_comp, random_state=0)
        scores = pca.fit_transform(X)

        scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_comp)])
        scores_df["sample_index"] = np.arange(scores_df.shape[0])

        # Add metadata for coloring/labels
        meta = APP.meta.copy() if APP.meta is not None else pd.DataFrame(index=scores_df.index)
        if meta is not None and not meta.empty:
            meta = meta.reset_index(drop=True)
            scores_df = pd.concat([scores_df, meta], axis=1)

        color_by = APP.color_col if APP.color_col in scores_df.columns else None
        hover_cols = [c for c in scores_df.columns if c not in [f"PC{i+1}" for i in range(n_comp)]]

        c1, c2 = st.columns([2, 1])

        with c1:
            st.subheader("PCA score plot")
            pcx = st.selectbox("X axis", [f"PC{i+1}" for i in range(n_comp)], index=0)
            pcy = st.selectbox("Y axis", [f"PC{i+1}" for i in range(n_comp)], index=1)
        
            show_ellipse = st.checkbox("Show 95% confidence ellipse", value=True, key="explore_pca_ellipse")

            group_color_map = None
            if color_by is not None and color_by in scores_df.columns:
                groups_sorted = sorted(scores_df[color_by].dropna().astype(str).unique().tolist())
                palette = px.colors.qualitative.Plotly
                group_color_map = {
                    grp: palette[i % len(palette)]
                    for i, grp in enumerate(groups_sorted)
                }
         
            fig_scores = px.scatter(
                    scores_df,
                    x=pcx,
                    y=pcy,
                    color=color_by,
                    hover_data=hover_cols,
                    title=f"PCA Scores: {pcx} vs {pcy}",
                    color_discrete_map=group_color_map if group_color_map is not None else None,
                )
         
            fig_scores.update_traces(marker=dict(size=9, line=dict(width=0)))

            if show_ellipse:
                fig_scores = add_confidence_ellipse_to_fig(
                    fig_scores,
                    scores_df,
                    x_col=pcx,
                    y_col=pcy,
                    group_col=color_by,
                    level=0.95,
                    color_map=group_color_map,
                )
        
            fig_scores.update_layout(dragmode="zoom")
            st.plotly_chart(fig_scores, use_container_width=True, config={"displaylogo": False})
            key = "explore_pca_scores"
            store_fig(key, fig_scores)
            add_download_html_button(fig_scores, "Download HTML: PCA scores", key)

        with c2:
            st.subheader("Explained variance")
            evr = pca.explained_variance_ratio_ * 100.0
            evr_df = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(n_comp)], "Explained_%": evr})
            fig_evr = px.bar(evr_df, x="PC", y="Explained_%", title="Explained variance (%)")
            st.plotly_chart(fig_evr, use_container_width=True, config={"displaylogo": False})
            key = "explore_explained_variance"
            store_fig(key, fig_evr)
            add_download_html_button(fig_evr, "Download HTML: explained variance", key)

        st.divider()
        st.subheader("Correlation heatmap (processed X)")

        # Safety check: processed matrix must match stored feature names
        feats_all = list(APP.feature_names)
        if X.shape[1] != len(feats_all):
            st.error(
                f"Internal mismatch in Exploration tab: X has {X.shape[1]} columns "
                f"but feature_names has {len(feats_all)} names. "
                f"Please click '🧹 Clear APP data (reset preprocessing/models)' in the sidebar "
                f"and run preprocessing again."
            )
            st.stop()

        # Correlation on a subset if too many features
        max_features = st.slider("Max features for correlation heatmap", 10, 200, 60)
        rng = np.random.default_rng(0)

        if len(feats_all) > max_features:
            feats = list(rng.choice(feats_all, size=max_features, replace=False))
        else:
            feats = feats_all

        # Build a proper DataFrame with all feature columns, then subset by name
        if X.shape[1] != len(feats_all):
            st.error(
                f"Internal mismatch in Exploration tab: X has {X.shape[1]} columns "
                f"but feature_names has {len(feats_all)} names. "
                f"Please click '🧹 Clear APP data (reset preprocessing/models)' in the sidebar "
                f"and run preprocessing again."
            )
            st.stop()

        X_df = pd.DataFrame(X, columns=feats_all)
        X_sub = X_df[feats]
        corr = X_sub.corr()

        fig_corr = px.imshow(
            corr,
            title="Correlation heatmap (subset)",
            aspect="auto",
        )
        st.plotly_chart(fig_corr, use_container_width=True, config={"displaylogo": False})
        key = "explore_corr_heatmap"
        store_fig(key, fig_corr)
        add_download_html_button(fig_corr, "Download HTML: correlation heatmap", key)

        st.download_button(
            "Download ALL Exploration plots (ZIP of HTML)",
            data=zip_html({k: v for k, v in FIGS.items() if k.startswith("explore_")}),
            file_name="exploration_plots_html.zip",
            mime="application/zip",
            use_container_width=True,
        )

# -------------------------
# 4) Modeling (LogReg + PLS-DA)
# -------------------------
with tabs[4]:
    st.header("4) Modeling")

    if APP.X_proc is None:
        st.info("Run preprocessing first.")
    elif APP.y_raw is None:
        st.warning("No target y selected. Choose a categorical target column in the sidebar to model.")
    else:
        y_ser = APP.y_raw

        # basic cleanup: drop missing y
        mask = ~pd.isna(y_ser)
        X = APP.X_proc[mask.values, :]
        y = y_ser[mask].astype(str).values

        # --- determine max folds allowed (useful later / consistency) ---
        class_counts = pd.Series(y).value_counts()
        min_class_n = int(class_counts.min())
        if min_class_n < 2:
            st.error(
                f"Not enough samples per class for supervised modeling. "
                f"Counts: {class_counts.to_dict()} "
                f"(each class needs at least 2 samples)."
            )
            st.stop()
        max_allowed_folds = min(10, min_class_n)
        st.caption(f"Class counts: {class_counts.to_dict()} | max folds allowed: {max_allowed_folds}")

        feats = APP.feature_names

        # -------------------------
        # Model selector
        # -------------------------
        model_kind = st.selectbox(
            "Choose supervised model",
            ["Logistic Regression (baseline)", "PLS-DA (PLS regression on one-hot y)"],
            index=1,
        )

        figs_local = {}

        # =====================================================================
        # A) Logistic Regression (baseline)
        # =====================================================================
        if model_kind.startswith("Logistic"):
            st.subheader("Logistic Regression (baseline classifier)")

            c1, c2 = st.columns(2)
            with c1:
                C = st.slider("Inverse regularization (C)", 0.01, 10.0, 1.0, key="logreg_C")
            with c2:
                max_iter = st.slider("max_iter", 100, 5000, 1000, step=100, key="logreg_maxiter")

            model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver="lbfgs",
            )

            model.fit(X, y)

            st.write("Classes:", list(model.classes_))
            st.write("n_samples:", X.shape[0], " | n_features:", X.shape[1])

            st.divider()
            st.subheader("Coefficients (feature importance proxy)")
            coef = model.coef_

            if coef.shape[0] == 1:
                df_coef = pd.DataFrame({"feature": feats, "coef": coef[0]}).sort_values("coef", ascending=False)
                topn = st.slider("Top N", 5, min(50, len(feats)), 20, key="logreg_topn_bin")
                df_show = pd.concat([df_coef.head(topn), df_coef.tail(topn)], axis=0)
                fig_coef = px.bar(df_show, x="coef", y="feature", orientation="h", title="Top + Bottom coefficients")
                st.plotly_chart(fig_coef, use_container_width=True, config={"displaylogo": False})
                key = "model_logreg_coefficients"
                store_fig(key, fig_coef)
                add_download_html_button(fig_coef, "Download HTML: coefficients", key)
                figs_local[key] = fig_coef
            else:
                strength = np.linalg.norm(coef, axis=0)
                df_coef = pd.DataFrame({"feature": feats, "strength": strength}).sort_values("strength", ascending=False)
                topn = st.slider("Top N", 5, min(50, len(feats)), 30, key="logreg_topn_multi")
                df_show = df_coef.head(topn)
                fig_coef = px.bar(
                    df_show,
                    x="strength",
                    y="feature",
                    orientation="h",
                    title="Feature strength (L2 norm across classes)",
                )
                st.plotly_chart(fig_coef, use_container_width=True, config={"displaylogo": False})
                key = "model_logreg_feature_strength"
                store_fig(key, fig_coef)
                add_download_html_button(fig_coef, "Download HTML: feature strength", key)
                figs_local[key] = fig_coef

        # =====================================================================
        # B) PLS-DA (PLSRegression on one-hot y)
        # =====================================================================
        else:
            st.subheader("PLS-DA")

            st.info(
                "PLS-DA is implemented as PLS regression where **y is one-hot encoded**. "
                "Scores = latent variables; Loadings = variable contributions. "
                "Validation (CV / permutation) should be done in the Validation tab."
            )

            classes = sorted(pd.unique(y).tolist())
            y_cat = pd.Categorical(y, categories=classes)
            Y = pd.get_dummies(y_cat).values  # (n_samples x n_classes)

            max_comp = min(10, X.shape[1], X.shape[0] - 1)
            if max_comp < 2:
                st.warning(f"PLS-DA needs at least 2 components possible, but max_comp={max_comp}. "
                           f"(Check if you have too few samples/features after preprocessing.)")
                st.stop()  # <-- THIS st.stop IS OK HERE (top-level tab), not inside an expander
            else:
                n_comp = st.slider(
                    "PLS-DA components",
                    min_value=2,
                    max_value=max_comp,
                    value=2,
                    key="plsda_ncomp",
                    #help="Limited by n_samples and n_features.",
                    help=PARAM_HELP["plsda_components"],
                )

            # Fit PLS
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X, Y)

            # Scores (T): sample coordinates in LV space
            T = pls.x_scores_  # shape (n_samples, n_comp)

            scores_df = pd.DataFrame(T, columns=[f"LV{i+1}" for i in range(n_comp)])
            scores_df["class"] = y

            # Add SampleID if available (nice for hover)
            if APP.raw is not None and APP.id_col and APP.id_col in APP.raw.columns:
                # Align indices: use same mask used above
                sample_ids = APP.raw.loc[mask.values, APP.id_col].astype(str).values
                scores_df[APP.id_col] = sample_ids

            hover_cols = [c for c in scores_df.columns if c not in [f"LV{i+1}" for i in range(n_comp)]]

            c1, c2 = st.columns([2, 1])

            with c1:
                lvx = st.selectbox("X axis", [f"LV{i+1}" for i in range(n_comp)], index=0, key="plsda_lvx")
                lvy = st.selectbox("Y axis", [f"LV{i+1}" for i in range(n_comp)], index=1, key="plsda_lvy")
            
                show_pls_ellipse = st.checkbox(
                    "Show 95% confidence ellipse",
                    value=True,
                    key="plsda_score_ellipse",
                )
            
                pls_group_color_map = None
                if "class" in scores_df.columns:
                    groups_sorted = sorted(scores_df["class"].dropna().astype(str).unique().tolist())
                    palette = px.colors.qualitative.Plotly
                    pls_group_color_map = {
                        grp: palette[i % len(palette)]
                        for i, grp in enumerate(groups_sorted)
                    }
            
                fig_pls_scores = px.scatter(
                    scores_df,
                    x=lvx,
                    y=lvy,
                    color="class",
                    hover_data=hover_cols,
                    title=f"PLS-DA Scores: {lvx} vs {lvy}",
                    color_discrete_map=pls_group_color_map if pls_group_color_map is not None else None,
                )
            
                fig_pls_scores.update_traces(marker=dict(size=9, line=dict(width=0)))
            
                if show_pls_ellipse:
                    fig_pls_scores = add_confidence_ellipse_to_fig(
                        fig_pls_scores,
                        scores_df,
                        x_col=lvx,
                        y_col=lvy,
                        group_col="class",
                        level=0.95,
                        color_map=pls_group_color_map,
                    )
            
                fig_pls_scores.update_layout(dragmode="zoom")
                st.plotly_chart(fig_pls_scores, use_container_width=True, config={"displaylogo": False})
            
                key = "model_plsda_scores"
                store_fig(key, fig_pls_scores)
                add_download_html_button(fig_pls_scores, "Download HTML: PLS-DA scores", key)
                figs_local[key] = fig_pls_scores

            with c2:
                # Simple proxy: fraction of X variance captured per component
                # (PLS doesn't expose "explained variance" exactly like PCA; this is didactic)
                X_hat = pls.x_scores_ @ pls.x_loadings_.T
                ss_total = np.sum(X ** 2)
                ss_res = np.sum((X - X_hat) ** 2)
                r2x = 1.0 - (ss_res / ss_total) if ss_total > 0 else np.nan
                st.metric("R²X (overall, approx.)", f"{r2x:.3f}" if np.isfinite(r2x) else "NA")

                # Also show class distribution for context
                st.write("Classes:", classes)

                # -------------------------------------------------
                # What is Q²? (didactic dropdown explanation)
                # -------------------------------------------------
                with st.expander("What is Q² (cross-validated predictive ability)?", expanded=False):
                
                    st.markdown("""
                **Q² measures how well the model predicts unseen samples.**
                
                Unlike **R²**, which measures how well the model fits the training data,
                Q² evaluates predictive performance using **cross-validation**.
                
                ### Concept
                
                1. The dataset is split into several folds.
                2. The model is trained on part of the data.
                3. Predictions are made on the left-out samples.
                4. Prediction errors are accumulated.
                
                
                ### Interpretation
                
                | Q² value | Meaning |
                |--------|--------|
                | < 0 | model predicts worse than the mean (**overfitting**) |
                | 0 – 0.3 | weak predictive power |
                | 0.3 – 0.5 | moderate predictive ability |
                | > 0.5 | good predictive model |
                """)
             
                # -----------------------------
                # Q² (cross-validated predictive ability)
                # -----------------------------

                st.subheader("Cross-validated Q²",help="Q² measures how well the model predicts new data, usually using cross-validation.")

                # CV parameters
                cv_folds = st.slider(
                    "Folds for Q²",
                    min_value=2,
                    max_value=max_allowed_folds,
                    value=min(5, max_allowed_folds),
                    key="plsda_q2_folds",
                )

                cv_repeats = st.slider(
                    "Repeats for Q²",
                    min_value=1,
                    max_value=20,
                    value=3,
                    key="plsda_q2_repeats",
                )

                seed = st.number_input("Random seed (Q²)", value=0, step=1, key="plsda_q2_seed")

                from sklearn.model_selection import StratifiedKFold

                Y_true_all = []
                Y_pred_all = []

                for r in range(cv_repeats):
                    cv = StratifiedKFold(
                        n_splits=cv_folds,
                        shuffle=True,
                        random_state=int(seed) + r
                    )

                    for train_idx, test_idx in cv.split(X, y):
                        pls_cv = PLSRegression(n_components=n_comp)
                        pls_cv.fit(X[train_idx], Y[train_idx])

                        Y_pred = pls_cv.predict(X[test_idx])

                        Y_true_all.append(Y[test_idx])
                        Y_pred_all.append(Y_pred)

                Y_true_all = np.vstack(Y_true_all)
                Y_pred_all = np.vstack(Y_pred_all)

                # Compute Q²
                PRESS = np.sum((Y_true_all - Y_pred_all) ** 2)
                TSS = np.sum((Y_true_all - np.mean(Y_true_all, axis=0)) ** 2)

                Q2 = 1.0 - PRESS / TSS if TSS > 0 else np.nan

                st.metric("Q² (cross-validated)", f"{Q2:.3f}")

            st.divider()
            st.subheader("PLS-DA Loadings (which variables drive separation)")

            # Loadings: X-loadings (P) shape (n_features, n_comp)
            P = pls.x_loadings_
            comp_to_show = st.selectbox(
                "Component for loadings",
                [f"LV{i+1}" for i in range(n_comp)],
                index=0,
                key="plsda_loading_comp",
            )
            j = int(comp_to_show.replace("LV", "")) - 1

            load_df = pd.DataFrame({"feature": feats, "loading": P[:, j]})
            load_df = load_df.sort_values("loading", ascending=False)

            topn = st.slider("Top N (positive/negative)", 5, min(100, len(feats)), 30, key="plsda_topn_load")
            load_show = pd.concat([load_df.head(topn), load_df.tail(topn)], axis=0)

            fig_load = px.bar(
                load_show,
                x="loading",
                y="feature",
                orientation="h",
                title=f"Loadings for {comp_to_show} (Top + Bottom)",
            )
            st.plotly_chart(fig_load, use_container_width=True, config={"displaylogo": False})
            key = f"model_plsda_loadings_{comp_to_show}"
            store_fig(key, fig_load)
            add_download_html_button(fig_load, f"Download HTML: loadings {comp_to_show}", key)
            figs_local[key] = fig_load

            st.divider()
            st.subheader("VIP scores (Variable Importance in Projection)")
            
            st.caption(PARAM_HELP["vip"])

            # VIP calculation (standard PLS VIP)
            # X: (n x p), T: (n x a), W: (p x a), Q: (m x a) or (a x m) depending on sklearn
            # sklearn: x_weights_ is (p x a), y_loadings_ is (m x a)
            W = pls.x_weights_               # (p, a)
            Q = pls.y_loadings_              # (m, a)
            a = n_comp
            p = X.shape[1]

            # Sum of squares explained in Y by each component:
            # SSa = sum over responses of (t_a^2) * (q_a^2)
            # We'll compute using T and Q columns.
            SS = np.zeros(a)
            for k in range(a):
                t = T[:, k]
                q = Q[:, k]
                SS[k] = np.sum(t ** 2) * np.sum(q ** 2)

            # VIP_j = sqrt( p * sum_k (SS_k * (w_jk^2 / ||w_k||^2)) / sum_k SS_k )
            vip = np.zeros(p)
            SS_sum = np.sum(SS) if np.sum(SS) > 0 else np.nan
            for j in range(p):
                s = 0.0
                for k in range(a):
                    wk = W[:, k]
                    denom = np.sum(wk ** 2)
                    if denom > 0:
                        s += SS[k] * (W[j, k] ** 2 / denom)
                vip[j] = np.sqrt(p * s / SS_sum) if np.isfinite(SS_sum) and SS_sum > 0 else np.nan

            vip_df = pd.DataFrame({"feature": feats, "VIP": vip}).sort_values("VIP", ascending=False)
            APP.vip_df = vip_df.copy()
            APP.plsda_scores_df = scores_df.copy()
            topn_vip = st.slider("Top VIP features", 5, min(100, len(feats)), 30, key="plsda_topn_vip")
            vip_show = vip_df.head(topn_vip)

            fig_vip = px.bar(vip_show, x="VIP", y="feature", orientation="h", title="Top VIP features")
            st.plotly_chart(fig_vip, use_container_width=True, config={"displaylogo": False})
            key = "model_plsda_vip"
            store_fig(key, fig_vip)
            add_download_html_button(fig_vip, "Download HTML: VIP", key)
            figs_local[key] = fig_vip

            # Optional: show a table too
            with st.expander("Show VIP table"):
                st.dataframe(vip_df, use_container_width=True)

        # -------------------------
        # Download all modeling plots
        # -------------------------
        if figs_local:
            st.download_button(
                "Download ALL Modeling plots (ZIP of HTML)",
                data=zip_html(figs_local),
                file_name="modeling_plots_html.zip",
                mime="application/zip",
                use_container_width=True,
            )

# -------------------------
# 5) Validation (CV + confusion + ROC)
# -------------------------
with tabs[5]:
    st.header("5) Validation")
    with st.expander("What is the purpose of validation?", expanded=False):
        st.markdown(PARAM_HELP["validation_overview"])
    
    with st.expander("How does cross-validation work?", expanded=False):
        st.markdown(PARAM_HELP["validation_cv_overview"])

    if APP.X_proc is None:
        st.info("Run preprocessing first.")
    elif APP.y_raw is None:
        st.warning("No target y selected.")
    else:
        # -------------------------
        # Data
        # -------------------------
        y_ser = APP.y_raw
        mask = ~pd.isna(y_ser)
        X = APP.X_proc[mask.values, :]
        y = y_ser[mask].astype(str).values

        # Stable class order
        classes = np.array(sorted(pd.unique(y).tolist()))

        # Folds allowed by smallest class
        class_counts = pd.Series(y).value_counts()
        min_class_n = int(class_counts.min()) if len(class_counts) else 0
        if min_class_n < 2:
            st.error(f"Not enough samples per class for CV. Counts: {class_counts.to_dict()}")
            st.stop()

        max_allowed_folds = min(10, min_class_n)
        st.caption(f"Class counts: {class_counts.to_dict()} | max folds allowed: {max_allowed_folds}")

        # -------------------------
        # CV controls
        # -------------------------
        st.subheader("Cross-validation")
        with st.expander("Help — Cross-validation settings", expanded=False):
            st.markdown(PARAM_HELP["validation_cv_overview"])
            st.markdown(PARAM_HELP["validation_repeats"])
         
        cv_folds = st.slider(
            "Folds",
            min_value=2,
            max_value=max_allowed_folds,
            value=min(5, max_allowed_folds),
            key="val_folds",
            #help=f"Max allowed folds: {max_allowed_folds} (min class size = {min_class_n})",
            help=PARAM_HELP["cv_folds"],
        )
        n_repeats = st.slider("Repeats", 1, 20, 3, key="val_repeats",help=PARAM_HELP["cv_repeats"])
        seed = st.number_input("Random seed", value=0, step=1, key="val_seed")

        # Model controls
        C = st.slider("C (LogReg)", 0.01, 10.0, 1.0, key="val_C")
        max_iter = st.slider("max_iter", 100, 5000, 1000, step=100, key="val_max_iter")
        model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")

        # -------------------------
        # Repeated CV predictions
        # -------------------------
        y_true_all: List[np.ndarray] = []
        y_pred_all: List[np.ndarray] = []
        y_proba_all: List[np.ndarray] = []

        for r in range(int(n_repeats)):
            cv = StratifiedKFold(
                n_splits=int(cv_folds),
                shuffle=True,
                random_state=int(seed) + r,
            )

            y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
            y_true_all.append(y)
            y_pred_all.append(y_pred)

            # Probabilities only when available
            try:
                y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
                y_proba_all.append(y_proba)
            except Exception:
                pass

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)

        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        st.write(f"Accuracy: **{acc:.3f}**")
        st.write(f"Balanced accuracy: **{bacc:.3f}**")
        with st.expander("Help — Accuracy vs Balanced Accuracy", expanded=False):
            st.markdown(PARAM_HELP["validation_accuracy"])
            st.markdown("---")
            st.markdown(PARAM_HELP["validation_balanced_accuracy"])

        # -------------------------
        # Confusion matrix
        # -------------------------
        st.divider()
        st.subheader("Confusion matrix")
     
        with st.expander("Help — How to read the confusion matrix", expanded=False):
            st.markdown(PARAM_HELP["validation_confusion_matrix"])


        cm = confusion_matrix(y_true, y_pred, labels=classes)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true:{c}" for c in classes],
            columns=[f"pred:{c}" for c in classes],
        )
        fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion Matrix (repeated CV)")
        st.plotly_chart(fig_cm, use_container_width=True, config={"displaylogo": False})
        store_fig("validation_confusion_matrix", fig_cm)
        add_download_html_button(fig_cm, "Download HTML: confusion matrix", "validation_confusion_matrix")

        # -------------------------
        # ROC (binary only)
        # -------------------------
        st.divider()
        st.subheader("ROC (binary only)")
     
        with st.expander("Help — ROC curve and AUC", expanded=False):
            st.markdown(PARAM_HELP["validation_roc"])

        figs_local = {"validation_confusion_matrix": fig_cm}

        if len(classes) == 2 and len(y_proba_all) > 0:
            # Stack probabilities from the repeats that actually produced them
            proba = np.vstack(y_proba_all)

            # y order from cross_val_predict is aligned to the input y each time
            y_true_for_proba = np.tile(y, len(y_proba_all))

            # Sanity check: rows must match
            if proba.shape[0] != y_true_for_proba.shape[0]:
                st.warning("ROC skipped: probability rows do not match y_true length.")
            else:
                # IMPORTANT: get the true probability-column order from the estimator
                model_tmp = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
                model_tmp.fit(X, y)
                proba_classes = model_tmp.classes_  # column order used by predict_proba

                # Guard: ensure proba columns match the estimator's class order
                if proba.shape[1] != len(proba_classes):
                    st.warning("ROC skipped: probability output shape does not match class list.")
                else:
                    with st.expander("Help — Choosing the positive class", expanded=False):
                        st.markdown(PARAM_HELP["validation_positive_class"])
                    pos_label = st.selectbox(
                        "Positive class",
                        options=list(proba_classes),
                        index=1,
                        key="val_pos_label",
                    )
                    pos_idx = int(np.where(proba_classes == pos_label)[0][0])

                    y_bin = (y_true_for_proba == pos_label).astype(int)
                    y_score = proba[:, pos_idx]

                    auc = roc_auc_score(y_bin, y_score)
                    fpr, tpr, _ = roc_curve(y_bin, y_score)

                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
                    fig_roc.add_trace(
                        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash"))
                    )
                    fig_roc.update_layout(
                        title="ROC Curve (Repeated CV)",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        dragmode="zoom",
                    )

                    st.plotly_chart(fig_roc, use_container_width=True, config={"displaylogo": False})
                    store_fig("validation_roc", fig_roc)
                    add_download_html_button(fig_roc, "Download HTML: ROC curve", "validation_roc")
                    figs_local["validation_roc"] = fig_roc
        else:
            st.info("ROC is shown only for binary targets with probability predictions.")


        # -------------------------
        # Download all
        # -------------------------
        st.download_button(
            "Download ALL Validation plots (ZIP of HTML)",
            data=zip_html(figs_local),
            file_name="validation_plots_html.zip",
            mime="application/zip",
            use_container_width=True,
        )

        # -------------------------
        # Text report
        # -------------------------
        st.divider()
        st.subheader("Classification report (text)")

        with st.expander("Help — Classification report", expanded=False):
            st.markdown(PARAM_HELP["validation_classification_report"])

        st.code(classification_report(y_true, y_pred), language="text")

# -------------------------
# 6) Interpretation
# -------------------------
with tabs[6]:
    st.header("6) Interpretation")

    if APP.X_proc is None:
        st.info("Run preprocessing first.")
    else:
        st.subheader("Interpretation is *visual* + contextual")
        st.write(
            """
This tab is the place to teach:
- What a separation/prediction means in **real terms**
- Which variables matter **and why**
- How to avoid overclaiming (validation + domain knowledge)

For now, this starter app includes:
- PCA explained variance + scores (Exploration tab)
- Model coefficients / feature strength (Modeling tab)
- Confusion matrix + ROC (Validation tab)

Next upgrades for this tab (recommended):
- Contribution plots for selected samples/groups
- Permutation tests (PLS-DA style)
- SHAP (tree models) or permutation importance (any model)
- Report generator (HTML/PDF)
"""
        )

        # Provide "download all figures so far" convenience
        st.divider()
        st.subheader("Download everything (all stored figures)")
        if FIGS:
            st.download_button(
                "Download ALL figures from all tabs (ZIP of HTML)",
                data=zip_html(FIGS),
                file_name="all_figures_html.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.info("No figures stored yet. Generate plots in previous tabs first.")

    method_format = st.selectbox(
    "Export analysis description",
        [
        "Pipeline summary (short)",
        "Methods paragraph (paper ready)",
        "Detailed report (full parameters)"
        ]
    )


# -------------------------
# 7) Univariate Analysis
# -------------------------
with tabs[7]:
    st.header("7) Univariate Analysis")

    with st.expander("What is univariate analysis?", expanded=False):
        st.markdown(PARAM_HELP["univariate_overview"])

    if APP.X_proc is None or APP.feature_names is None:
        st.info("Run preprocessing first.")
    elif APP.y_raw is None:
        st.warning("A group/class column is required for univariate comparison.")
    else:
        y_ser = APP.y_raw.copy()
        mask = ~pd.isna(y_ser)
        y = y_ser[mask].astype(str).reset_index(drop=True)

        X_df = pd.DataFrame(
            APP.X_proc[mask.values, :],
            columns=APP.feature_names
        ).reset_index(drop=True)

        # Build working dataframe
        uni_df = X_df.copy()
        uni_df["Group"] = y.values

        if APP.id_col and APP.raw is not None and APP.id_col in APP.raw.columns:
            uni_df["SampleID"] = APP.raw.loc[mask.values, APP.id_col].astype(str).values
        else:
            uni_df["SampleID"] = [f"Sample_{i+1}" for i in range(len(uni_df))]

        st.subheader("Feature selection")

        c1, c2 = st.columns(2)

        with c1:
            feature_order_mode = st.selectbox(
                "Order features by",
                ["Alphabetical", "VIP (if available)"],
                index=1 if APP.vip_df is not None else 0,
            )

        with c2:
            top_n_candidates = st.slider(
                "Number of candidate features to show",
                min_value=5,
                max_value=min(200, len(APP.feature_names)),
                value=min(30, len(APP.feature_names)),
                step=5,
            )

        feature_options = APP.feature_names.copy()

        if feature_order_mode == "VIP (if available)" and APP.vip_df is not None:
            with st.expander("Help — VIP-based feature ranking", expanded=False):
                st.markdown(PARAM_HELP["vip_univariate_help"])

            vip_features = APP.vip_df["feature"].tolist()
            feature_options = [f for f in vip_features if f in APP.feature_names][:top_n_candidates]
        else:
            feature_options = sorted(APP.feature_names)[:top_n_candidates]

        selected_features = st.multiselect(
            "Select features for univariate analysis",
            options=feature_options,
            default=feature_options[:min(3, len(feature_options))],
        )

        if not selected_features:
            st.info("Select at least one feature.")
            st.stop()

        st.divider()
        st.subheader("Plot settings")

        c1, c2, c3 = st.columns(3)
        with c1:
            plot_points = st.checkbox("Show individual points", value=True)
        with c2:
            use_box = st.checkbox("Show boxplot", value=True)
        with c3:
            use_log_y = st.checkbox("Log Y axis", value=False)

        figs_local = {}

        for feat in selected_features:
            st.divider()
            st.subheader(f"Feature: {feat}")

            df_feat = uni_df[["Group", "SampleID", feat]].copy()
            df_feat = df_feat.rename(columns={feat: "Value"})
            df_feat = df_feat[np.isfinite(df_feat["Value"])]

            if df_feat.empty:
                st.warning(f"No valid numeric data for {feat}.")
                continue

            # summary table
            summary_df = (
                df_feat.groupby("Group")["Value"]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .reset_index()
            )

            c1, c2 = st.columns([2, 1])

            with c1:
                fig = go.Figure()

                groups_sorted = sorted(df_feat["Group"].astype(str).unique().tolist())
                palette = px.colors.qualitative.Plotly
                group_color_map = {
                    grp: palette[i % len(palette)]
                    for i, grp in enumerate(groups_sorted)
                }

                if use_box:
                    for grp in groups_sorted:
                        sub = df_feat[df_feat["Group"] == grp]
                        fig.add_trace(
                            go.Box(
                                x=sub["Group"],
                                y=sub["Value"],
                                name=str(grp),
                                marker_color=group_color_map[grp],
                                boxpoints=False,
                                showlegend=False,
                            )
                        )

                if plot_points:
                    fig_points = px.strip(
                        df_feat,
                        x="Group",
                        y="Value",
                        color="Group",
                        hover_data=["SampleID"],
                        color_discrete_map=group_color_map,
                    )
                    for tr in fig_points.data:
                        tr.showlegend = False
                        fig.add_trace(tr)

                fig.update_layout(
                    title=f"Boxplot / group comparison — {feat}",
                    xaxis_title="Group",
                    yaxis_title=feat,
                    dragmode="zoom",
                )

                if use_log_y:
                    fig.update_yaxes(type="log")

                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

                key = f"univariate_boxplot_{feat}"
                store_fig(key, fig)
                add_download_html_button(fig, f"Download HTML: {feat}", key)
                figs_local[key] = fig

            with c2:
                st.write("Group summary")
                st.dataframe(summary_df, use_container_width=True)

            # statistical tests
            st.markdown("**Statistical test**")

            groups_data = [
                df_feat.loc[df_feat["Group"] == grp, "Value"].dropna().values
                for grp in groups_sorted
            ]
            groups_data = [g for g in groups_data if len(g) > 0]

            if len(groups_data) >= 2:
                with st.expander("Help — ANOVA and t-test", expanded=False):
                    st.markdown(PARAM_HELP["anova_help"])
                    st.markdown("---")
                    st.markdown(PARAM_HELP["ttest_help"])

                if len(groups_sorted) == 2:
                    g1, g2 = groups_sorted[0], groups_sorted[1]
                    x1 = df_feat.loc[df_feat["Group"] == g1, "Value"].dropna().values
                    x2 = df_feat.loc[df_feat["Group"] == g2, "Value"].dropna().values

                    if len(x1) >= 2 and len(x2) >= 2:
                        t_stat, p_val = stats.ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
                        st.write(f"Welch t-test: **t = {t_stat:.4f}**, **p = {p_val:.4e}**")
                    else:
                        st.info("Not enough observations in one of the two groups for t-test.")

                if len(groups_sorted) >= 3:
                    enough_groups = all(len(g) >= 2 for g in groups_data)
                    if enough_groups:
                        f_stat, p_val = stats.f_oneway(*groups_data)
                        st.write(f"One-way ANOVA: **F = {f_stat:.4f}**, **p = {p_val:.4e}**")
                    else:
                        st.info("At least one group has too few values for ANOVA.")
            else:
                st.info("At least two groups are needed.")

            # optional raw table
            with st.expander(f"Show raw values for {feat}", expanded=False):
                st.dataframe(df_feat, use_container_width=True)

        if figs_local:
            st.divider()
            st.download_button(
                "Download ALL Univariate plots (ZIP of HTML)",
                data=zip_html(figs_local),
                file_name="univariate_plots_html.zip",
                mime="application/zip",
                use_container_width=True,
            )
