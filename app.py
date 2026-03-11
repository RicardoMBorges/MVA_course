
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
for logo_name in ["LAABio.png", "MVA_Course.png"]:
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

# --------- Math funtions for Normalization
def _as_numeric_df(X_df: pd.DataFrame) -> pd.DataFrame:
    return X_df.apply(pd.to_numeric, errors="coerce")

def quantile_normalize_rows(X: pd.DataFrame) -> pd.DataFrame:
    """
    Quantile normalization across samples (rows), treating each sample distribution equally.
    X is samples x features.
    Returns samples x features.
    """
    # work as numpy for speed
    A = X.to_numpy(dtype=float, copy=True)      # (n_samples, n_features)
    At = A.T                                    # (n_features, n_samples)

    # argsort each sample (column in At)
    order = np.argsort(At, axis=0)              # (n_features, n_samples)
    sorted_vals = np.take_along_axis(At, order, axis=0)

    # average across samples at each rank
    mean_sorted = np.nanmean(sorted_vals, axis=1)   # (n_features,)

    # put the mean_sorted back into original positions (inverse sort)
    out = np.empty_like(At)
    np.put_along_axis(out, order, mean_sorted[:, None], axis=0)

    return pd.DataFrame(out.T, index=X.index, columns=X.columns)


def sample_normalize(
    X: pd.DataFrame,
    method: str,
    sample_factor: Optional[pd.Series] = None,
    ref_sample: Optional[pd.Series] = None,
    ref_feature: Optional[str] = None,
    group_labels: Optional[pd.Series] = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    X: samples x features
    Returns X normalized by the chosen method.
    """
    Xn = X.copy()

    if method == "None":
        return Xn

    # -----------------------------------------------------
    # Sample-specific factor normalization
    # -----------------------------------------------------
    if method == "Sample-specific normalization (factor)":
        if sample_factor is None:
            raise ValueError("sample_factor is required for sample-specific normalization.")
        f = pd.to_numeric(sample_factor, errors="coerce").astype(float)
        if f.isna().any():
            raise ValueError("Sample factor has missing/non-numeric values.")
        return Xn.div(f.values, axis=0)

    # -----------------------------------------------------
    # Simple row scalings
    # -----------------------------------------------------
    if method == "Normalization by sum":
        s = Xn.sum(axis=1).replace(0, np.nan)
        return Xn.div(s.values, axis=0)

    if method == "Normalization by median":
        m = Xn.median(axis=1).replace(0, np.nan)
        return Xn.div(m.values, axis=0)

    # -----------------------------------------------------
    # PQN (reference sample)
    # -----------------------------------------------------
    if method == "Normalization by a reference sample (PQN)":
        if ref_sample is None:
            raise ValueError("ref_sample is required for PQN.")

        ref = pd.to_numeric(ref_sample, errors="coerce").astype(float)

        # Xn: (samples x features)
        # ref: (features,)
        quot = Xn.div(ref.values + eps, axis=1)
        factors = quot.median(axis=1).replace(0, np.nan)

        return Xn.div(factors.values, axis=0)

    # -----------------------------------------------------
    # Group PQN
    # -----------------------------------------------------
    if method == "Normalization by a pooled sample from group (group PQN)":
        if group_labels is None:
            raise ValueError("group_labels is required for group PQN.")

        gl = group_labels.astype(str)
        Xout = Xn.copy()

        for g in gl.unique():
            idx = gl == g
            ref = Xn.loc[idx].median(axis=0)  # pooled reference per group
            quot = Xn.loc[idx].div(ref.values + eps, axis=1)
            factors = quot.median(axis=1).replace(0, np.nan)
            Xout.loc[idx] = Xn.loc[idx].div(factors.values, axis=0)

        return Xout

    # -----------------------------------------------------
    # Reference feature normalization
    # -----------------------------------------------------
    if method == "Normalization by reference feature":
        if ref_feature is None:
            raise ValueError("ref_feature is required for reference-feature normalization.")
        if ref_feature not in Xn.columns:
            raise ValueError(f"Reference feature '{ref_feature}' not found in X.")

        f = Xn[ref_feature].replace(0, np.nan)
        return Xn.div(f.values, axis=0)

    # -----------------------------------------------------
    # Quantile normalization
    # -----------------------------------------------------
    if method == "Quantile normalization":
        n_samp, n_feat = Xn.shape
        if n_samp * n_feat > 5_000_000:
            raise ValueError(
                f"Quantile normalization is too heavy for this size ({n_samp}×{n_feat}). "
                "Reduce features/samples or use another normalization."
            )
        return quantile_normalize_rows(Xn)

    raise ValueError(f"Unknown sample normalization method: {method}")


def transform_data(X: pd.DataFrame, method: str, eps: float = 1e-12) -> pd.DataFrame:
    Xt = X.copy()

    if method == "None":
        return Xt

    if method in {"Log transformation (base 10)", "Log transformation (base 2)"}:
        if (Xt.values < 0).any():
            raise ValueError("Log transform selected but data contains negative values (often from batch centering).")

    if method == "Log transformation (base 10)":
        return np.log10(Xt + eps)

    if method == "Log transformation (base 2)":
        return np.log2(Xt + eps)
    if method == "Square root transformation":
        return np.sqrt(np.clip(Xt, a_min=0, a_max=None))

    if method == "Cube root transformation":
        # supports negatives too
        return np.cbrt(Xt)

    if method == "Variance stabilizing normalization (VSN)":
        # Simple, robust VSN-like transform for teaching:
        # asinh(x / s), with s = median of nonzero values (global)
        vals = Xt.values.flatten()
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]
        s = np.median(vals) if vals.size else 1.0
        s = float(s) if s > 0 else 1.0
        return np.arcsinh(Xt / s)

    raise ValueError(f"Unknown transformation method: {method}")


def batch_align(X: pd.DataFrame, batch: Optional[pd.Series], method: str) -> pd.DataFrame:
    """
    Very simple "alignment" = batch correction / centering.
    X: samples x features
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

# -------------------------
# Sidebar: Data import + column mapping
# -------------------------
st.sidebar.title("Data Import")

uploaded = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
    help="""
Accepted format (your example):

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


# -------------------------
# Mapping UI for this special format
# -------------------------
df_u = st.session_state.get("raw_uploaded_df", None)

if df_u is not None and not df_u.empty:
    st.sidebar.subheader("Format mapping (columns=samples, rows=variables)")

    # 1) Choose which column holds the row labels (often the first unnamed column)
    # Pandas may name it "Unnamed: 0" or it might be an empty string depending on file.
    candidate_label_cols = df_u.columns.tolist()
    default_label_col = candidate_label_cols[0]

    row_label_col = st.sidebar.selectbox(
        "Row-label column (contains ATTRIBUTE_class / Feature_1 / ...)",
        options=candidate_label_cols,
        index=0,
        help="This is usually the first column (often named 'Unnamed: 0' in CSV).",
    )

    # Create a preview list of row labels for selection
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
    # Default: all rows except the chosen class row
    default_feature_rows = [r for r in row_labels if r != class_row_label]
    feature_rows = st.sidebar.multiselect(
        "Feature rows (variables used as X)",
        options=row_labels,
        default=default_feature_rows,
        help="Pick the rows that represent numeric features (Feature_1, Feature_2, ...).",
    )

    # 5) Parse + store into your APP.* variables (sklearn-friendly orientation)
    if st.sidebar.button("Apply mapping", type="primary"):
        try:
            X_df, y = parse_course_table(
                df=df_u,
                row_label_col=row_label_col,
                sample_cols=sample_cols,
                class_row_label=class_row_label,
                feature_rows=feature_rows,
            )

            # Store in your app state (match your earlier design)
            APP.raw = X_df.reset_index()  # has SampleID column
            APP.id_col = "SampleID"

            # Add y as a column if present, so the rest of your pipeline can use selectboxes
            if y is not None:
                APP.raw["ATTRIBUTE_class"] = y.values
                APP.y_col = "ATTRIBUTE_class"
                APP.color_col = "ATTRIBUTE_class"
            else:
                APP.y_col = None
                APP.color_col = None

            # X columns are numeric features (after transpose)
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
        "3) Exploration",
        "4) Modeling",
        "5) Validation",
        "6) Interpretation",
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

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Preview")
            st.dataframe(df.head(50), use_container_width=True)
        with c2:
            st.subheader("Shape")
            st.write(f"Rows: **{df.shape[0]}**")
            st.write(f"Cols: **{df.shape[1]}**")
            st.subheader("Missingness report")
            rep = build_missing_report(df)
            st.dataframe(rep.head(30), use_container_width=True, height=420)

        # Build X/y/meta snapshots
        if APP.X_cols:
            APP.X_raw = df[APP.X_cols].copy()
            APP.feature_names = APP.X_cols.copy()
        else:
            APP.X_raw = None
            APP.feature_names = None

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
            imp = SimpleImputer(strategy="median")
            X_raw_imp = imp.fit_transform(X_raw_df.values)

            if X_raw_imp.shape[0] < 3 or X_raw_imp.shape[1] < 2:
                st.warning("Not enough data for raw PCA (need >=3 samples and >=2 features).")
            else:
                n_comp = st.slider(
                    "Raw PCA components",
                    min_value=2,
                    max_value=min(10, X_raw_imp.shape[1]),
                    value=min(3, X_raw_imp.shape[1]),
                    key="import_raw_pca_ncomp",
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

    if APP.raw is None or APP.X_raw is None or not APP.X_cols:
        st.info("Select numeric feature columns (X) in the sidebar.")
    else:
        df_full = APP.raw.copy()
        X_df = _as_numeric_df(APP.X_raw.copy())  # samples x features (raw numeric)

        st.subheader("Preprocessing choices")

        # ---------------------------------------
        # Controls (organized)
        # ---------------------------------------
        c1, c2, c3 = st.columns(3)

        with c1:
            impute_strategy = st.selectbox(
                "Missing value imputation",
                ["median", "mean", "most_frequent", "constant (0)"],
                index=0,
            )
            if impute_strategy == "constant (0)":
                imp = SimpleImputer(strategy="constant", fill_value=0.0)
            else:
                imp = SimpleImputer(strategy=impute_strategy)

            missing_col_thresh = st.slider(
                "Drop features with missing % above",
                0, 100, 90,
                help="If a feature has too many missing values, you can drop it.",
            )

        with c2:
            sample_norm = st.selectbox(
                "Sample normalization",
                [
                    "None",
                    "Sample-specific normalization (factor)",
                    "Normalization by sum",
                    "Normalization by median",
                    "Normalization by a reference sample (PQN)",
                    "Normalization by a pooled sample from group (group PQN)",
                    "Normalization by reference feature",
                    "Quantile normalization",
                ],
                index=0,
            )

            transform = st.selectbox(
                "Data transformation",
                [
                    "None",
                    "Log transformation (base 10)",
                    "Log transformation (base 2)",
                    "Square root transformation",
                    "Cube root transformation",
                    "Variance stabilizing normalization (VSN)",
                ],
                index=0,
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
                help="Optional. Requires a batch column in your table (metadata column in APP.raw).",
            )

            scaling = st.selectbox(
                "Scaling (feature-wise)",
                ["none", "standard (z-score)"],
                index=1,
                help="Typical for PCA / many models.",
            )
            scaler = None if scaling == "none" else StandardScaler(with_mean=True, with_std=True)

            drop_zero_var = st.checkbox("Drop zero-variance features", value=True)

        st.divider()
        st.subheader("Extra parameters (only when needed)")

        # Identify metadata candidates in APP.raw (anything not in X_cols)
        meta_candidates = [c for c in df_full.columns if c not in (APP.X_cols or [])]

        # sample-specific factor column (weight/volume/etc.)
        factor_col = None
        if sample_norm == "Sample-specific normalization (factor)":
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
        if sample_norm == "Normalization by a reference sample (PQN)":
            if APP.id_col and APP.id_col in df_full.columns:
                ref_sample_id = st.selectbox(
                    "Reference sample (for PQN)",
                    options=df_full[APP.id_col].astype(str).tolist(),
                    index=0,
                )
            else:
                st.warning("PQN needs SampleID available (APP.id_col). Add/keep SampleID in your mapped data.")

        # group PQN requires y / class column
        group_labels = None
        if sample_norm == "Normalization by a pooled sample from group (group PQN)":
            if APP.y_col and APP.y_col in df_full.columns:
                group_labels = df_full[APP.y_col].astype(str)
            else:
                st.warning("Group PQN requires a group/class column (APP.y_col).")

        # reference feature normalization
        ref_feature = None
        if sample_norm == "Normalization by reference feature":
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
            X_imp = imp.fit_transform(X_df2.values)
            X_imp_df = pd.DataFrame(X_imp, index=X_df2.index, columns=X_df2.columns)

            # 2) Sample normalization (row-wise)
            sample_factor = df_full[factor_col] if factor_col else None

            ref_sample_series = None
            if sample_norm == "Normalization by a reference sample (PQN)" and ref_sample_id is not None:
                idx = df_full[APP.id_col].astype(str) == str(ref_sample_id)
                if idx.sum() != 1:
                    st.error("Could not uniquely identify the reference sample for PQN.")
                    st.stop()
                ref_sample_series = X_imp_df.loc[idx].iloc[0]

            try:
                X_norm_df = sample_normalize(
                    X_imp_df,
                    method=sample_norm,
                    sample_factor=sample_factor,
                    ref_sample=ref_sample_series,
                    ref_feature=ref_feature,
                    group_labels=group_labels,
                )
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
            X_al_df = X_al_df.replace([np.inf, -np.inf], np.nan)

            # final imputation to guarantee PCA/models never see NaN
            final_imp = SimpleImputer(strategy="median")
            X_al_df = pd.DataFrame(
                final_imp.fit_transform(X_al_df),
                index=X_al_df.index,
                columns=X_al_df.columns,
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

            # 6) Scaling (feature-wise)
            if scaler is not None:
                X_proc = scaler.fit_transform(X_al_df.values)
            else:
                X_proc = X_al_df.values

            # Store to app state
            APP.X_proc = np.asarray(X_proc, dtype=float)
            APP.feature_names = X_al_df.columns.tolist()

            st.success(f"Processed X: {APP.X_proc.shape[0]} samples × {APP.X_proc.shape[1]} features")
            if dropped_missing:
                st.warning(f"Dropped (missingness): {len(dropped_missing)} features")
            if dropped_zero:
                st.warning(f"Dropped (zero variance): {len(dropped_zero)} features")

        # ---------------------------------------
        # Visualization: before vs after
        # ---------------------------------------
        st.divider()
        st.subheader("Before vs After (visual checks)")
        fast_mode = st.checkbox("Fast mode", value=True,
                                help="Reduces plot complexity so the app stays responsive.")

        MAX_SAMPLES_TRACES = 25 if fast_mode else 80
        MAX_FEATURES_TRACES = 25 if fast_mode else 80
        MAX_POINTS_PER_TRACE = 400 if fast_mode else 2000
        figs_local = {}

        # Matrices aligned to final feature list
        feat_labels = APP.feature_names
        raw_mat = _as_numeric_df(APP.X_raw.copy()).reindex(columns=feat_labels)
        proc_mat = pd.DataFrame(APP.X_proc, index=raw_mat.index, columns=feat_labels)

        if APP.X_proc.shape[1] != len(feat_labels):
            st.error("Internal mismatch: X_proc columns do not match feature_names. Please run preprocessing again.")
            st.stop()

        # Sample labels
        if APP.id_col and APP.id_col in df_full.columns:
            sample_names_all = df_full[APP.id_col].astype(str).tolist()
        else:
            sample_names_all = [f"Sample_{i}" for i in range(raw_mat.shape[0])]

        # ======================================================
        # A) DISTRIBUTION FOR EVERY SAMPLE (across features)
        # ======================================================
        with st.expander("Distributions: GROUP OVERLAY (across features) — raw vs processed", expanded=False):
            st.caption(
                "Fast view: for each group, we pool ALL values (samples×features) and overlay distributions by group. "
                "Use this to visually check sample-wise normalization effects without plotting each sample."
            )

            # --- need a group label column ---
            if not (APP.y_col and APP.y_col in df_full.columns):
                st.warning("No group column available (APP.y_col). Add/keep a class column (e.g., ATTRIBUTE_class).")
            else:
                group_series = df_full[APP.y_col].astype(str)

                # optional: limit features for speed (but still not per-feature plotting)
                n_feats = int(len(feat_labels))
                if n_feats < 2:
                    st.warning("Not enough features to plot pooled distributions.")
                else:
                    min_feat = 2
                    max_feat_allowed = min(5000, n_feats)
                    default_feat = min(800, max_feat_allowed)

                    if min_feat == max_feat_allowed:
                        max_feat = max_feat_allowed
                        st.caption(f"Max features used: {max_feat} (only option)")
                    else:
                        step_feat = 20 if (max_feat_allowed - min_feat) >= 20 else 1
                        max_feat = st.slider(
                            "Max features used for pooled distributions (speed control)",
                            min_value=min_feat,
                            max_value=max_feat_allowed,
                            value=default_feat,
                            step=step_feat,
                            help="This only limits how many feature columns are pooled. No individual feature plots.",
                            key="group_overlay_maxfeat",
                        )
                    feat_use = feat_labels[:max_feat]

                    # optional: sample cap for speed (again, pooled)
                    n_samples = int(raw_mat.shape[0])
                    if n_samples < 2:
                        st.warning("Not enough samples to plot pooled distributions.")
                    else:
                        min_samp = 2
                        max_samp_allowed = n_samples
                        default_samp = min(300, max_samp_allowed)

                        if min_samp == max_samp_allowed:
                            max_samp = max_samp_allowed
                            st.caption(f"Max samples used: {max_samp} (only option)")
                        else:
                            step_samp = 10 if (max_samp_allowed - min_samp) >= 10 else 1
                            max_samp = st.slider(
                                "Max samples used (speed control)",
                                min_value=min_samp,
                                max_value=max_samp_allowed,
                                value=default_samp,
                                step=step_samp,
                                key="group_overlay_maxsamp",
                            )

                        idx_use = list(range(min(max_samp, n_samples)))

                        # aligned matrices (subset)
                        raw_sub = raw_mat.iloc[idx_use][feat_use]
                        proc_sub = proc_mat.iloc[idx_use][feat_use]
                        grp_sub = group_series.iloc[idx_use]

                        # ---------- build long dataframe (pooled values) ----------
                        raw_long = pd.DataFrame(
                            {
                                "value": raw_sub.to_numpy().ravel(),
                                "group": np.repeat(grp_sub.values, len(feat_use)),
                                "stage": "raw",
                            }
                        )
                        proc_long = pd.DataFrame(
                            {
                                "value": proc_sub.to_numpy().ravel(),
                                "group": np.repeat(grp_sub.values, len(feat_use)),
                                "stage": "processed",
                            }
                        )

                        df_long = pd.concat([raw_long, proc_long], ignore_index=True)
                        df_long = df_long[np.isfinite(df_long["value"].values)]

                        # ---------- choose plot type ----------
                        plot_kind = st.radio(
                            "Plot type",
                            ["Histogram (fastest)", "Violin (still fast)"],
                            horizontal=True,
                            key="group_overlay_plotkind",
                        )

                        if plot_kind.startswith("Histogram"):
                            nbins = st.slider("Bins", 20, 200, 80, key="group_overlay_bins")

                            fig = px.histogram(
                                df_long,
                                x="value",
                                color="group",
                                facet_col="stage",
                                barmode="overlay",
                                nbins=nbins,
                                histnorm="probability density",
                                title="GROUP overlay distribution (pooled samples×features): RAW vs PROCESSED",
                            )
                            fig.update_layout(dragmode="zoom", height=520)
                            fig.update_traces(opacity=0.45)
                        else:
                            fig = px.violin(
                                df_long,
                                x="group",
                                y="value",
                                color="group",
                                facet_col="stage",
                                box=True,
                                points=False,
                                title="GROUP overlay distribution (pooled samples×features): RAW vs PROCESSED",
                            )
                            fig.update_layout(dragmode="zoom", height=520)

                        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                        key = "preprocess_group_overlay_raw_vs_processed"
                        store_fig(key, fig)
                        add_download_html_button(fig, "Download HTML: group overlay (raw vs processed)", key)
                        figs_local[key] = fig

        # ======================================================
        # B) DISTRIBUTION FOR EVERY FEATURE (across samples)
        #   NOW WITH 3 STAGES:
        #     1) RAW (as loaded; may have NaN)
        #     2) PRE-SCALE (after impute + norm + transform + alignment + final impute; BUT BEFORE scaling)
        #     3) PROCESSED (after scaling, if chosen)
        # ======================================================
        with st.expander("Distributions: EVERY FEATURE (across samples) — raw vs pre-scale vs processed", expanded=False):
            st.caption(
                "Each violin is one feature. Values are all samples for that feature.\n"
                "Stages:\n"
                "• RAW = original values\n"
                "• PRE-SCALE = after normalization/transform/alignment (no scaling)\n"
                "• PROCESSED = final matrix (after scaling, if enabled)"
            )

            max_feat_show = st.slider(
                "How many features to show (violin view)",
                min_value=5,
                max_value=min(300, len(feat_labels)),
                value=min(60, len(feat_labels)),
                help="Start with 40–100 for teaching. Thousands become heavy.",
                key="dist_features_n",
            )

            mode = st.radio(
                "Feature selection mode",
                ["First N features", "Choose features manually"],
                horizontal=True,
                key="dist_features_mode",
            )

            if mode == "First N features":
                feat_show = feat_labels[:max_feat_show]
            else:
                feat_show = st.multiselect(
                    "Pick features",
                    options=feat_labels,
                    default=feat_labels[: min(max_feat_show, len(feat_labels))],
                    key="dist_features_pick",
                )
                if len(feat_show) > max_feat_show:
                    feat_show = feat_show[:max_feat_show]
                    st.info(f"Showing first {max_feat_show} of your selection (performance cap).")

            # ---------- MATRICES ----------
            # RAW (may contain NaN/inf)
            raw_mat = _as_numeric_df(APP.X_raw.copy()).reindex(columns=feat_labels)
            raw_mat = raw_mat.replace([np.inf, -np.inf], np.nan)

            # PRE-SCALE: X_al_df is your "final non-scaled" dataframe (right before scaler.fit_transform)
            # Ensure it exists in this scope: it is created above in your preprocessing flow.
            if APP.X_pre_scale is None:
                st.warning("Pre-scale matrix not found. Click 'Run preprocessing' again.")
                st.stop()

            pre_scale_mat = APP.X_pre_scale.reindex(index=raw_mat.index)
            pre_scale_mat = pre_scale_mat.replace([np.inf, -np.inf], np.nan)

            # PROCESSED (scaled if selected, else equals pre-scale numerically)
            proc_mat = pd.DataFrame(APP.X_proc, index=raw_mat.index, columns=feat_labels)
            proc_mat = proc_mat.replace([np.inf, -np.inf], np.nan)

            # ---------- VIOLIN PLOT (each feature) ----------
            fig = go.Figure()

            # RAW (left)
            for f in feat_show:
                vals = raw_mat[f].values
                vals = vals[np.isfinite(vals)]
                fig.add_trace(
                    go.Violin(
                        y=vals,
                        name=f"{f} (raw)",
                        side="negative",
                        width=0.9,
                        points=False,
                        showlegend=False,
                        meanline_visible=True,
                    )
                )

            # PRE-SCALE (middle overlay)
            # Note: Plotly violin doesn't have a true "middle"; we'll overlay with smaller width and higher opacity.
            for f in feat_show:
                vals = pre_scale_mat[f].values
                vals = vals[np.isfinite(vals)]
                fig.add_trace(
                    go.Violin(
                        y=vals,
                        name=f"{f} (pre-scale)",
                        side="both",
                        width=0.35,
                        points=False,
                        showlegend=False,
                        meanline_visible=True,
                    )
                )

            # PROCESSED (right)
            for f in feat_show:
                vals = proc_mat[f].values
                vals = vals[np.isfinite(vals)]
                fig.add_trace(
                    go.Violin(
                        y=vals,
                        name=f"{f} (processed)",
                        side="positive",
                        width=0.9,
                        points=False,
                        showlegend=False,
                        meanline_visible=True,
                    )
                )

            fig.update_layout(
                title="Every feature distribution (across samples): RAW (left) vs PRE-SCALE (center) vs PROCESSED (right)",
                yaxis_title="Sample intensity (within-feature distribution)",
                xaxis_title="Features (stacked violins)",
                violingap=0.02,
                violinmode="overlay",
                height=650,
                dragmode="zoom",
            )

            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            key = "preprocess_dist_every_feature_violin_3stage"
            store_fig(key, fig)
            add_download_html_button(fig, "Download HTML: distributions per feature (3 stages)", key)
            figs_local[key] = fig

            # ---- RESULT distributions (global + feature-summary) ----
            st.divider()
            st.subheader("Resulting distributions (EVERY FEATURE)")

            # 1) GLOBAL distribution of all values (flattened): RAW vs PRE-SCALE vs PROCESSED
            raw_vals = raw_mat[feat_show].to_numpy().ravel()
            pre_vals = pre_scale_mat[feat_show].to_numpy().ravel()
            proc_vals = proc_mat[feat_show].to_numpy().ravel()

            raw_vals = raw_vals[np.isfinite(raw_vals)]
            pre_vals = pre_vals[np.isfinite(pre_vals)]
            proc_vals = proc_vals[np.isfinite(proc_vals)]

            df_global = pd.DataFrame(
                {
                    "value": np.concatenate([raw_vals, pre_vals, proc_vals]),
                    "stage": (["raw"] * len(raw_vals)) + (["pre-scale"] * len(pre_vals)) + (["processed"] * len(proc_vals)),
                }
            )

            fig_global = px.histogram(
                df_global,
                x="value",
                color="stage",
                barmode="overlay",
                nbins=80,
                histnorm="probability density",
                title="GLOBAL distribution of all values (samples×features): RAW vs PRE-SCALE vs PROCESSED",
            )
            fig_global.update_layout(dragmode="zoom")
            st.plotly_chart(fig_global, use_container_width=True, config={"displaylogo": False})
            key = "preprocess_result_global_values_features_3stage"
            store_fig(key, fig_global)
            add_download_html_button(fig_global, "Download HTML: global values (3 stages)", key)
            figs_local[key] = fig_global

            # 2) Distribution across features of feature-wise summaries (3 stages)
            def _feature_stats(mat: pd.DataFrame) -> pd.DataFrame:
                arr = mat.to_numpy()
                mean = np.nanmean(arr, axis=0)
                std = np.nanstd(arr, axis=0)
                med = np.nanmedian(arr, axis=0)
                cv = std / np.where(np.abs(mean) < 1e-12, np.nan, np.abs(mean))
                return pd.DataFrame({"mean": mean, "median": med, "std": std, "cv": cv})

            f_raw = _feature_stats(raw_mat[feat_show]);       f_raw["stage"] = "raw"
            f_pre = _feature_stats(pre_scale_mat[feat_show]); f_pre["stage"] = "pre-scale"
            f_pro = _feature_stats(proc_mat[feat_show]);      f_pro["stage"] = "processed"

            df_stats = pd.concat([f_raw, f_pre, f_pro], ignore_index=True).melt(
                id_vars=["stage"], var_name="stat", value_name="value"
            )

            fig_stats = px.violin(
                df_stats,
                x="stat",
                y="value",
                color="stage",
                box=True,
                points=False,
                title="Feature-wise summaries (distribution across features): RAW vs PRE-SCALE vs PROCESSED",
            )
            fig_stats.update_layout(dragmode="zoom")
            st.plotly_chart(fig_stats, use_container_width=True, config={"displaylogo": False})
            key = "preprocess_result_feature_summaries_3stage"
            store_fig(key, fig_stats)
            add_download_html_button(fig_stats, "Download HTML: feature summaries (3 stages)", key)
            figs_local[key] = fig_stats

        # ======================================================
        # C) Per-feature selected histograms (RAW vs PRE-SCALE vs PROCESSED)
        # ======================================================
        st.subheader("Selected feature distributions (histograms)")

        feat_pick = st.multiselect(
            "Pick features to compare",
            APP.feature_names,
            default=APP.feature_names[: min(3, len(APP.feature_names))],
            key="preprocess_hist_pick",
        )

        for f in feat_pick:
            # raw (may have NaN)
            raw_vals = pd.to_numeric(raw_mat[f], errors="coerce").values
            raw_vals = raw_vals[np.isfinite(raw_vals)]

            # pre-scale (no scaling; already cleaned/imputed)
            pre_vals = pd.to_numeric(pre_scale_mat[f], errors="coerce").values
            pre_vals = pre_vals[np.isfinite(pre_vals)]

            # processed (final)
            proc_vals = pd.to_numeric(proc_mat[f], errors="coerce").values
            proc_vals = proc_vals[np.isfinite(proc_vals)]

            if len(raw_vals) < 2 or len(proc_vals) < 2:
                st.warning(f"Not enough data for feature: {f}")
                continue

            # shared binning/range so RAW doesn't disappear
            all_vals = np.concatenate([raw_vals, pre_vals, proc_vals]) if len(pre_vals) else np.concatenate([raw_vals, proc_vals])
            xmin, xmax = float(np.min(all_vals)), float(np.max(all_vals))
            nbins = 50

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=raw_vals, name="raw", opacity=0.45,
                nbinsx=nbins, histnorm="probability density",
                xbins=dict(start=xmin, end=xmax),
            ))
            fig.add_trace(go.Histogram(
                x=pre_vals, name="pre-scale", opacity=0.45,
                nbinsx=nbins, histnorm="probability density",
                xbins=dict(start=xmin, end=xmax),
            ))
            fig.add_trace(go.Histogram(
                x=proc_vals, name="processed", opacity=0.45,
                nbinsx=nbins, histnorm="probability density",
                xbins=dict(start=xmin, end=xmax),
            ))

            fig.update_layout(
                barmode="overlay",
                title=f"{f}: RAW vs PRE-SCALE vs PROCESSED",
                xaxis_title="Value",
                yaxis_title="Density",
                dragmode="zoom",
                height=420,
            )

            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            key = f"preprocess_hist_{f}"
            store_fig(key, fig)
            add_download_html_button(fig, f"Download HTML: {f}", key)
            figs_local[key] = fig


# -------------------------
# 3) Exploration (PCA, correlations)
# -------------------------
with tabs[2]:
    st.header("3) Exploration")

    if APP.X_proc is None or APP.feature_names is None:
        st.info("Run preprocessing first (tab 2).")
    else:
        X = APP.X_proc
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

            fig_scores = px.scatter(
                scores_df,
                x=pcx,
                y=pcy,
                color=color_by,
                hover_data=hover_cols,
                title=f"PCA Scores: {pcx} vs {pcy}",
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
        # Correlation on a subset if too many features
        max_features = st.slider("Max features for correlation heatmap", 10, 200, 60)
        rng = np.random.default_rng(0)
        feats_all = list(APP.feature_names)
        if len(feats_all) > max_features:
            feats = list(rng.choice(feats_all, size=max_features, replace=False))
        else:
            feats = feats_all

        # Build a proper DataFrame with all feature columns, then subset by name
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
# IMPORTANT: add this import at the top of your file:
# from sklearn.cross_decomposition import PLSRegression

with tabs[3]:
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
                multi_class="auto",
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
                    help="Limited by n_samples and n_features.",
                )

            # Fit PLS
            from sklearn.cross_decomposition import PLSRegression
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

                fig_pls_scores = px.scatter(
                    scores_df,
                    x=lvx,
                    y=lvy,
                    color="class",
                    hover_data=hover_cols,
                    title=f"PLS-DA Scores: {lvx} vs {lvy}",
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

                # -----------------------------
                # Q² (cross-validated predictive ability)
                # -----------------------------

                st.subheader("Cross-validated Q²")

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
with tabs[4]:
    st.header("5) Validation")

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

        cv_folds = st.slider(
            "Folds",
            min_value=2,
            max_value=max_allowed_folds,
            value=min(5, max_allowed_folds),
            key="val_folds",
            help=f"Max allowed folds: {max_allowed_folds} (min class size = {min_class_n})",
        )
        n_repeats = st.slider("Repeats", 1, 20, 3, key="val_repeats")
        seed = st.number_input("Random seed", value=0, step=1, key="val_seed")

        # Model controls
        C = st.slider("C (LogReg)", 0.01, 10.0, 1.0, key="val_C")
        max_iter = st.slider("max_iter", 100, 5000, 1000, step=100, key="val_max_iter")
        model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", multi_class="auto")

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

        # -------------------------
        # Confusion matrix
        # -------------------------
        st.divider()
        st.subheader("Confusion matrix")

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
                model_tmp = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", multi_class="auto")
                model_tmp.fit(X, y)
                proba_classes = model_tmp.classes_  # column order used by predict_proba

                # Guard: ensure proba columns match the estimator's class order
                if proba.shape[1] != len(proba_classes):
                    st.warning("ROC skipped: probability output shape does not match class list.")
                else:
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
        st.code(classification_report(y_true, y_pred), language="text")

# -------------------------
# 6) Interpretation
# -------------------------
with tabs[5]:
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

