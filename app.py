
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
for logo_name in ["LAABio.png"]: #"logo_massQL.png", 
    p = STATIC_DIR / logo_name
    try:
        from PIL import Image
        st.sidebar.image(Image.open(p), use_container_width=True)
    except Exception:
        pass


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
    # self-contained HTML with plotly.js included
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
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
    sample_cols: list[str],
    class_row_label: str | None,
    feature_rows: list[str],
) -> tuple[pd.DataFrame, pd.Series | None]:
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


if "app" not in st.session_state:
    st.session_state["app"] = AppData()

APP: AppData = st.session_state["app"]

# Keep figures for "download all"
if "figs" not in st.session_state:
    st.session_state["figs"] = {}
FIGS: Dict[str, go.Figure] = st.session_state["figs"]


def store_fig(key: str, fig: go.Figure):
    FIGS[key] = fig


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

        # Requirements: need X columns selected (numeric features)
        if APP.X_cols and APP.raw is not None:
            df = APP.raw.copy()

            # Build raw X and drop rows with any missing values (raw PCA should be simple/transparent)
            X_raw_df = df[APP.X_cols].apply(pd.to_numeric, errors="coerce")
            n_before = X_raw_df.shape[0]
            X_raw_df = X_raw_df.dropna(axis=0, how="any")
            n_after = X_raw_df.shape[0]

            if n_after < 3:
                st.warning("Not enough complete samples for raw PCA after dropping missing values.")
            else:
                if n_after < n_before:
                    st.info(
                        f"Raw PCA: dropped {n_before - n_after} samples due to missing values "
                        "(no imputation in this view)."
                    )

                # PCA with no scaling/normalization
                n_comp = st.slider(
                    "Raw PCA components",
                    min_value=2,
                    max_value=min(10, X_raw_df.shape[1]),
                    value=min(3, X_raw_df.shape[1]),
                    key="import_raw_pca_ncomp",
                )

                pca_raw = PCA(n_components=n_comp, random_state=0)
                scores = pca_raw.fit_transform(X_raw_df.values)

                scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_comp)])
                scores_df["sample_index"] = np.arange(scores_df.shape[0])

                # add sample id + metadata (if available)
                if APP.id_col and APP.id_col in df.columns:
                    scores_df[APP.id_col] = df.loc[X_raw_df.index, APP.id_col].astype(str).values
                if APP.color_col and APP.color_col in df.columns:
                    scores_df[APP.color_col] = df.loc[X_raw_df.index, APP.color_col].astype(str).values
                if APP.y_col and APP.y_col in df.columns and APP.y_col not in scores_df.columns:
                    scores_df[APP.y_col] = df.loc[X_raw_df.index, APP.y_col].astype(str).values

                color_by = APP.color_col if (APP.color_col and APP.color_col in scores_df.columns) else None
                hover_cols = [c for c in scores_df.columns if not c.startswith("PC")]

                c1, c2 = st.columns([2, 1])

                with c1:
                    pcx = st.selectbox(
                        "X axis",
                        [f"PC{i+1}" for i in range(n_comp)],
                        index=0,
                        key="import_raw_pca_x",
                    )
                    pcy = st.selectbox(
                        "Y axis",
                        [f"PC{i+1}" for i in range(n_comp)],
                        index=1,
                        key="import_raw_pca_y",
                    )

                    fig_raw_scores = px.scatter(
                        scores_df,
                        x=pcx,
                        y=pcy,
                        color=color_by,
                        hover_data=hover_cols,
                        title=f"RAW PCA Scores (no scaling): {pcx} vs {pcy}",
                    )
                    fig_raw_scores.update_layout(dragmode="zoom")
                    st.plotly_chart(fig_raw_scores, use_container_width=True, config={"displaylogo": False})

                    key = "import_raw_pca_scores"
                    store_fig(key, fig_raw_scores)
                    add_download_html_button(fig_raw_scores, "Download HTML: raw PCA scores", key)

                with c2:
                    evr = pca_raw.explained_variance_ratio_ * 100.0
                    evr_df = pd.DataFrame(
                        {"PC": [f"PC{i+1}" for i in range(n_comp)], "Explained_%": evr}
                    )

                    fig_raw_evr = px.bar(
                        evr_df,
                        x="PC",
                        y="Explained_%",
                        title="RAW PCA explained variance (%)",
                    )
                    st.plotly_chart(fig_raw_evr, use_container_width=True, config={"displaylogo": False})

                    key = "import_raw_pca_explained_variance"
                    store_fig(key, fig_raw_evr)
                    add_download_html_button(fig_raw_evr, "Download HTML: raw PCA explained variance", key)

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
        X_df = APP.X_raw.copy()

        st.subheader("Preprocessing choices")
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

        with c2:
            scaling = st.selectbox(
                "Scaling",
                ["none", "standard (z-score)"],
                index=1,
                help="Start with standard scaling for PCA / many models.",
            )
            scaler = None if scaling == "none" else StandardScaler(with_mean=True, with_std=True)

        with c3:
            drop_zero_var = st.checkbox("Drop zero-variance features", value=True)
            missing_col_thresh = st.slider(
                "Drop features with missing % above",
                0, 100, 90,
                help="If a feature has too many missing values, you can drop it.",
            )

        # Drop high-missing columns
        miss_pct = X_df.isna().mean() * 100.0
        keep_cols = miss_pct[miss_pct <= missing_col_thresh].index.tolist()
        dropped_missing = [c for c in X_df.columns if c not in keep_cols]
        X_df = X_df[keep_cols]

        # Drop zero-variance columns (after imputation/scaling we will check again later, but do now too)
        if drop_zero_var:
            vari = X_df.var(axis=0, skipna=True)
            keep2 = vari[vari > 0].index.tolist()
            dropped_zero = [c for c in X_df.columns if c not in keep2]
            X_df = X_df[keep2]
        else:
            dropped_zero = []

        # Pipeline to produce processed X
        steps = [("imputer", imp)]
        if scaler is not None:
            steps.append(("scaler", scaler))
        pipe = Pipeline(steps)
        X_proc = pipe.fit_transform(X_df.values)
        APP.X_proc = X_proc
        APP.feature_names = X_df.columns.tolist()

        st.success(f"Processed X: {X_proc.shape[0]} samples × {X_proc.shape[1]} features")
        if dropped_missing:
            st.warning(f"Dropped (missingness): {len(dropped_missing)} features")
        if dropped_zero:
            st.warning(f"Dropped (zero variance): {len(dropped_zero)} features")

        st.divider()
        st.subheader("Before vs After: example feature distributions")
        feat_pick = st.multiselect(
            "Pick features to compare",
            APP.feature_names,
            default=APP.feature_names[: min(3, len(APP.feature_names))],
        )

        figs_local = {}
        for f in feat_pick:
            idx = APP.feature_names.index(f)

            before = APP.X_raw[f].astype(float)
            after = pd.Series(APP.X_proc[:, idx], name=f)

            df_long = pd.DataFrame({"value": pd.concat([before, after], ignore_index=True),
                                    "stage": ["raw"] * len(before) + ["processed"] * len(after)})

            fig = px.histogram(df_long, x="value", color="stage", barmode="overlay", nbins=40,
                               title=f"Raw vs Processed: {f}")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            key = f"preprocess_hist_{f}"
            store_fig(key, fig)
            add_download_html_button(fig, f"Download HTML: {f}", key)
            figs_local[key] = fig

        if figs_local:
            st.download_button(
                "Download ALL Preprocessing plots (ZIP of HTML)",
                data=zip_html(figs_local),
                file_name="preprocessing_plots_html.zip",
                mime="application/zip",
                use_container_width=True,
            )


# -------------------------
# 3) Exploration (PCA, correlations)
# -------------------------
with tabs[2]:
    st.header("3) Exploration")

    if APP.X_proc is None or APP.feature_names is None:
        st.info("Run preprocessing first (tab 2).")
    else:
        X = APP.X_proc
        n_comp = st.slider("PCA components", 2, min(10, X.shape[1]), 3)

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
        feats = APP.feature_names[: min(max_features, len(APP.feature_names))]
        X_sub = pd.DataFrame(X[:, : len(feats)], columns=feats)
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
            n_comp = st.slider(
                "PLS-DA components",
                min_value=2,
                max_value=max(2, max_comp),
                value=min(2, max_comp),
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
        y = APP.y_raw
        mask = ~pd.isna(y)
        X = APP.X_proc[mask.values, :]
        y = y[mask].astype(str).values

        # Choose CV strategy
        st.subheader("Cross-validation")
        # --- CHANGED: folds slider max depends on data ---
        cv_folds = st.slider(
            "Folds",
            min_value=2,
            max_value=max_allowed_folds,
            value=min(5, max_allowed_folds),
            key="val_folds",
            help=f"Max allowed folds for your data: {max_allowed_folds} (min class size = {min_class_n})"
        )
        n_repeats = st.slider("Repeats", 1, 20, 3)
        seed = st.number_input("Random seed", value=0, step=1)

        # Model
        C = st.slider("C (LogReg)", 0.01, 10.0, 1.0, key="val_C")
        max_iter = st.slider("max_iter", 100, 5000, 1000, step=100, key="val_max_iter")
        model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", multi_class="auto")

        # Repeated CV predictions
        preds_all = []
        probs_all = []
        y_all = []
        fold_id = []

        classes = np.unique(y)

        for r in range(n_repeats):
            # --- CHANGED: safety clamp ---
            effective_folds = min(cv_folds, max_allowed_folds)
            cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=int(seed) + r)
            # Predict labels
            y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
            preds_all.append(y_pred)
            y_all.append(y)

            # Predict probabilities (for ROC AUC where possible)
            try:
                y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
                probs_all.append(y_proba)
            except Exception:
                probs_all.append(None)

        y_pred = np.concatenate(preds_all)
        y_true = np.concatenate(y_all)

        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)

        st.write(f"Accuracy: **{acc:.3f}**")
        st.write(f"Balanced accuracy: **{bacc:.3f}**")

        st.divider()
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        cm_df = pd.DataFrame(cm, index=[f"true:{c}" for c in classes], columns=[f"pred:{c}" for c in classes])
        fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion Matrix (repeated CV predictions)")
        st.plotly_chart(fig_cm, use_container_width=True, config={"displaylogo": False})
        store_fig("validation_confusion_matrix", fig_cm)
        add_download_html_button(fig_cm, "Download HTML: confusion matrix", "validation_confusion_matrix")

        st.divider()
        st.subheader("ROC (binary only)")
        figs_local = {"validation_confusion_matrix": fig_cm}
        if len(classes) == 2 and probs_all and probs_all[0] is not None:
            # average over repeats by concatenation
            proba = np.vstack([p for p in probs_all if p is not None])
            # proba order corresponds to model.classes_ but cross_val_predict preserves it per fold.
            # We'll assume consistent ordering; use class index of positive label chosen by user.
            pos_label = st.selectbox("Positive class", list(classes), index=1)
            pos_idx = list(classes).index(pos_label)

            y_bin = (y_true == pos_label).astype(int)
            y_score = proba[:, pos_idx]

            auc = roc_auc_score(y_bin, y_score)
            fpr, tpr, _ = roc_curve(y_bin, y_score)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
            fig_roc.update_layout(
                title="ROC Curve (Repeated CV)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
            )
            st.plotly_chart(fig_roc, use_container_width=True, config={"displaylogo": False})
            store_fig("validation_roc", fig_roc)
            add_download_html_button(fig_roc, "Download HTML: ROC curve", "validation_roc")
            figs_local["validation_roc"] = fig_roc
        else:
            st.info("ROC curve is shown only for binary targets with probability predictions.")

        st.download_button(
            "Download ALL Validation plots (ZIP of HTML)",
            data=zip_html(figs_local),
            file_name="validation_plots_html.zip",
            mime="application/zip",
            use_container_width=True,
        )

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
