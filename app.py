# ==========================================================
# DoE Toolkit (NumPy-only designs)
# + Quadratic model + visualization + response surfaces
# ==========================================================

import itertools
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.spatial import Delaunay
import math
import streamlit as st
from pathlib import Path
from PIL import Image

# MUST be the first Streamlit call
st.set_page_config(page_title="DoE Toolkit", layout="wide")

# -----------------------------
# LOGOS — AFTER page config
# -----------------------------

STATIC_DIR = Path(__file__).parent / "static"

laabio_logo = STATIC_DIR / "LAABio.png"
doe_logo = STATIC_DIR / "logo_DoE.png"

col_left, col_center, col_right = st.columns([1.2, 2, 1.2])

# Center logo (Main DoE branding)
if doe_logo.exists():
    try:
        st.sidebar.image(Image.open(doe_logo), use_container_width=True)
    except Exception:
        pass

if laabio_logo.exists():
    try:
        st.sidebar.image(Image.open(laabio_logo), use_container_width=True)
    except Exception:
        pass

st.sidebar.markdown("---")

st.sidebar.link_button(
    "📘 Tutorial (GitHub)",
    "https://github.com/RicardoMBorges/DoE_pipeline_st/blob/main/tutorial.md"
)


st.title("Design of Experiments (DoE) — Toolkit")

st.markdown("""
This app helps you plan and analyze experiments in **any domain** (chemistry, biology, engineering, optimization, etc.).

You can:

1. Generate experimental designs (factorial, CCD, Box–Behnken, fractional, LHS)
2. Export a run sheet for the lab/bench
3. Upload the completed table with measured responses
4. Fit a **quadratic** model (main effects + interactions + curvature)
5. Visualize response surfaces (2D contour + 3D surface)
6. Search for **best predicted conditions** and validate experimentally
""")

# ==========================================================
# DESIGN GENERATORS (Pure NumPy)
# ==========================================================

def full_factorial_2level(k: int) -> np.ndarray:
    return np.array(list(itertools.product([-1, 1], repeat=k)), dtype=float)

def box_behnken(k: int) -> np.ndarray:
    if k < 3:
        raise ValueError("Box-Behnken requires at least 3 factors.")
    design = []
    for i in range(k):
        for j in range(i + 1, k):
            for pair in itertools.product([-1, 1], repeat=2):
                row = [0] * k
                row[i] = pair[0]
                row[j] = pair[1]
                design.append(row)
    design.append([0] * k)  # center point
    return np.array(design, dtype=float)

def central_composite(k: int) -> np.ndarray:
    factorial = full_factorial_2level(k)
    axial = []
    alpha = float(np.sqrt(k))  # rotatable-ish default
    for i in range(k):
        row_pos = [0] * k
        row_neg = [0] * k
        row_pos[i] = alpha
        row_neg[i] = -alpha
        axial.append(row_pos)
        axial.append(row_neg)
    center = np.zeros((1, k), dtype=float)
    return np.vstack([factorial, np.array(axial, dtype=float), center])


def full_factorial_3level(k: int) -> np.ndarray:
    """
    3-level full factorial in coded space: {-1, 0, +1}
    Runs = 3^k (can explode quickly).
    """
    levels = [-1.0, 0.0, 1.0]
    return np.array(list(itertools.product(levels, repeat=k)), dtype=float)


def lhs_design(n_runs: int, k: int, seed: int = 123) -> np.ndarray:
    """
    Latin Hypercube Sampling in coded space approximately in [-1, +1].
    Good for low-resolution exploration when k is large.
    """
    rng = np.random.default_rng(seed)
    # Stratify [0,1] into n bins; sample one from each bin per factor
    H = np.zeros((n_runs, k), dtype=float)
    for j in range(k):
        cut = np.linspace(0, 1, n_runs + 1)
        u = rng.random(n_runs)
        pts = cut[:-1] + u * (cut[1:] - cut[:-1])
        rng.shuffle(pts)
        H[:, j] = pts
    # map [0,1] -> [-1,1]
    return 2.0 * H - 1.0


def _parse_generator(gen: str, base_names: list[str]) -> list[int]:
    g = gen.strip().upper().replace(" ", "")
    if g == "":
        raise ValueError("Empty generator.")

    # must be at least 2 letters (avoid "A" which duplicates base factor)
    if len(g) < 2:
        raise ValueError(f"Generator '{gen}' must have at least 2 base letters (e.g., AB).")

    # no repeated letters (avoid AA -> constant)
    if len(set(g)) != len(g):
        raise ValueError(f"Generator '{gen}' repeats a letter (e.g., AA). Not allowed.")

    idx = []
    for ch in g:
        if ch not in base_names:
            raise ValueError(f"Generator '{gen}' uses unknown base factor '{ch}'. Base={base_names}")
        idx.append(base_names.index(ch))
    return idx


def fractional_factorial_2level_regular(
    base_k: int,
    generators: list[str],
) -> np.ndarray:
    """
    Regular 2-level fractional factorial.

    You build a 2^(base_k) full factorial in base factors A,B,C,...
    Then add derived factors defined by generators, e.g.:
      generators=["AB", "AC"] means:
        D = A*B
        E = A*C 
    For simplicity: generators must use ONLY base letters A.. (no derived letters).

    Total factors = base_k + len(generators)
    Runs = 2^(base_k)
    """
    if base_k < 2:
        raise ValueError("base_k must be >= 2")

    base_names = [chr(ord("A") + i) for i in range(base_k)]
    base = full_factorial_2level(base_k)  # shape: (2^base_k, base_k)

    derived_cols = []
    for gen in generators:
        idx = _parse_generator(gen, base_names)
        col = np.prod(base[:, idx], axis=1).reshape(-1, 1)
        derived_cols.append(col)

    if len(derived_cols) > 0:
        return np.hstack([base] + derived_cols).astype(float)

    return base.astype(float)

# ==========================================================
# MODEL HELPERS
# ==========================================================

def build_quadratic_matrix(df_coded: pd.DataFrame, factor_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Quadratic model basis:
      1
      x_i
      x_i*x_j  (i<j)
      x_i^2
    """
    n = len(df_coded)
    X_parts = [np.ones((n, 1), dtype=float)]
    names = ["Intercept"]

    # linear
    for c in factor_cols:
        X_parts.append(df_coded[[c]].to_numpy(dtype=float))
        names.append(c)

    # interactions
    for i in range(len(factor_cols)):
        for j in range(i + 1, len(factor_cols)):
            xi = df_coded[factor_cols[i]].to_numpy(dtype=float)
            xj = df_coded[factor_cols[j]].to_numpy(dtype=float)
            X_parts.append((xi * xj).reshape(-1, 1))
            names.append(f"{factor_cols[i]}*{factor_cols[j]}")

    # squares
    for c in factor_cols:
        xi = df_coded[c].to_numpy(dtype=float)
        X_parts.append((xi ** 2).reshape(-1, 1))
        names.append(f"{c}^2")

    X = np.hstack(X_parts)
    return X, names

def fit_lstsq(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    return coef.flatten(), yhat.flatten()

def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def predict_quadratic(point: dict, coef: np.ndarray, factor_cols: list[str]) -> float:
    """
    point: dict factor->coded float
    coef: fitted coefficients aligned with build_quadratic_matrix()
    """
    x = np.array([point[c] for c in factor_cols], dtype=float)
    row = [1.0]                       # intercept
    row.extend(list(x))               # linear
    for i in range(len(x)):           # interactions
        for j in range(i + 1, len(x)):
            row.append(x[i] * x[j])
    for i in range(len(x)):           # squares
        row.append(x[i] ** 2)
    return float(np.dot(np.array(row, dtype=float), coef))

def coded_to_real_value(coded_val: float, spec: dict) -> float:
    """
    Map coded value to real value by linear interpolation using low/center/high.
    Handles coded values beyond [-1,1] (e.g., CCD axial points).
    """
    low, center, high = float(spec["low"]), float(spec["center"]), float(spec["high"])
    if coded_val == 0:
        return center
    # piecewise linear: [-1..0] and [0..+1], extend linearly beyond
    if coded_val < 0:
        # between low (-1) and center (0)
        return center + coded_val * (center - low)
    else:
        # between center (0) and high (+1)
        return center + coded_val * (high - center)

# ----------------------------------------------------------
# Utility: Download Plotly figure as standalone HTML
# ----------------------------------------------------------
def download_plotly_html(fig, filename: str, button_label: str):
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    st.download_button(
        label=button_label,
        data=html,
        file_name=filename,
        mime="text/html",
        use_container_width=True,
    )

def run_mixture_design_extraction():
    #st.title("Mixture Design – Extraction Solvent Optimization")

    n = st.selectbox("Number of solvents", [2, 3, 4], index=1)
    resolution = st.slider("Grid resolution", 3, 40, 5)

    st.markdown("### Solvent names + bounds (fractions)")
    names, mins, maxs = [], [], []

    for i in range(n):
        name = st.text_input(
            f"Solvent {i+1} name",
            f"S{i+1}",
            key=f"solv_name_{i}",
        )

        c1, c2 = st.columns(2)
        mn = c1.number_input(
            f"Min fraction (Solvent {i+1}: {name})",
            0.0, 1.0, 0.0, 0.01,
            key=f"min_{i}",
        )
        mx = c2.number_input(
            f"Max fraction (Solvent {i+1}: {name})",
            0.0, 1.0, 1.0, 0.01,
            key=f"max_{i}",
        )

        names.append(name.strip() if name else f"S{i+1}")
        mins.append(float(mn))
        maxs.append(float(mx))

    # Feasibility checks
    if any(mn > mx for mn, mx in zip(mins, maxs)):
        st.error("At least one solvent has min > max. Fix the bounds.")
        return

    if sum(mins) > 1.0 + 1e-9:
        st.error("Sum of minimum fractions is > 1. Reduce mins.")
        return

    if sum(maxs) < 1.0 - 1e-9:
        st.error("Sum of maximum fractions is < 1. Increase maxs so fractions can sum to 1.")
        return

    X = generate_constrained_simplex_grid(n, resolution, mins=mins, maxs=maxs)

    if X.size == 0:
        st.error("No feasible mixture points found with these min/max constraints and grid resolution.")
        return

    st.subheader("Generated design points")
    df = pd.DataFrame(X, columns=names)

    # store mixture design in session state (so other tabs can use it)
    st.session_state["mixture_df"] = df
    st.session_state["mixture_names"] = names
    st.session_state["mixture_n"] = n
    st.session_state["mixture_resolution"] = resolution

    st.dataframe(df, use_container_width=True)

    # Download design
    st.download_button(
        "Download design CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="mixture_design_extraction.csv",
        mime="text/csv"
    )

    # Visualization of the design space
    st.subheader("Design space visualization")
    if n == 2:
        fig = px.scatter(df, x=names[0], y=names[1], title="Binary mixture design")
        st.plotly_chart(fig, use_container_width=True)

    elif n == 3:
        fig = px.scatter_ternary(df, a=names[0], b=names[1], c=names[2],
                                 title="Ternary mixture design")
        st.plotly_chart(fig, use_container_width=True)

    elif n == 4:
        fig = px.scatter_3d(df, x=names[0], y=names[1], z=names[2], color=names[3],
                            title="Quaternary mixture design (color = 4th solvent)")
        st.plotly_chart(fig, use_container_width=True)

def generate_simplex_grid(n, resolution):

    if n == 2:
        x = np.linspace(0, 1, resolution + 1)
        return np.column_stack([x, 1 - x])

    elif n == 3:
        points = []
        for i in range(resolution + 1):
            for j in range(resolution + 1 - i):
                k = resolution - i - j
                points.append([i, j, k])
        points = np.array(points) / resolution
        return points

    elif n == 4:
        points = []
        for i in range(resolution + 1):
            for j in range(resolution + 1 - i):
                for k in range(resolution + 1 - i - j):
                    l = resolution - i - j - k
                    points.append([i, j, k, l])
        points = np.array(points) / resolution
        return points


def generate_constrained_simplex_grid(n, resolution, mins=None, maxs=None):
    """
    Generates mixture points on a simplex lattice and filters by min/max constraints.
    All fractions sum to 1.

    mins/maxs: lists of length n with bounds in [0,1].
    """
    if mins is None:
        mins = [0.0] * n
    if maxs is None:
        maxs = [1.0] * n

    mins = np.array(mins, dtype=float)
    maxs = np.array(maxs, dtype=float)

    pts = []

    if n == 2:
        for i in range(resolution + 1):
            x1 = i / resolution
            x2 = 1 - x1
            p = np.array([x1, x2])
            if np.all(p >= mins) and np.all(p <= maxs):
                pts.append(p)

    elif n == 3:
        for i in range(resolution + 1):
            for j in range(resolution + 1 - i):
                k = resolution - i - j
                p = np.array([i, j, k], dtype=float) / resolution
                if np.all(p >= mins) and np.all(p <= maxs):
                    pts.append(p)

    elif n == 4:
        for i in range(resolution + 1):
            for j in range(resolution + 1 - i):
                for k in range(resolution + 1 - i - j):
                    l = resolution - i - j - k
                    p = np.array([i, j, k, l], dtype=float) / resolution
                    if np.all(p >= mins) and np.all(p <= maxs):
                        pts.append(p)

    return np.array(pts)



def visualize_mixture(design, names):

    n = len(names)

    if n == 2:
        fig = px.line(
            x=design[:,0],
            y=design[:,1],
            labels={"x": names[0], "y": names[1]},
            title="Binary Mixture Design"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif n == 3:
        df = pd.DataFrame(design, columns=names)

        fig = px.scatter_ternary(
            df,
            a=names[0],
            b=names[1],
            c=names[2],
            title="Ternary Mixture Simplex"
        )

        st.plotly_chart(fig, use_container_width=True)
        
    elif n == 4:
        df = pd.DataFrame(design, columns=names)

        fig = px.scatter_3d(
            df,
            x=names[0],
            y=names[1],
            z=names[2],
            color=names[3],
            title="Quaternary Mixture (Color-coded 4th component)"
        )

        st.plotly_chart(fig, use_container_width=True)        
        
def build_scheffe_quadratic_matrix(df_mix: pd.DataFrame, mix_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Scheffé quadratic mixture model (no intercept):
      y = sum(b_i x_i) + sum(b_ij x_i x_j)  for i<j
    No intercept because mixtures satisfy sum(x)=1 (intercept is redundant).
    """
    n = len(df_mix)
    X_parts = []
    names = []

    # linear terms
    for c in mix_cols:
        X_parts.append(df_mix[[c]].to_numpy(dtype=float))
        names.append(c)

    # pairwise interactions
    for i in range(len(mix_cols)):
        for j in range(i + 1, len(mix_cols)):
            xi = df_mix[mix_cols[i]].to_numpy(dtype=float)
            xj = df_mix[mix_cols[j]].to_numpy(dtype=float)
            X_parts.append((xi * xj).reshape(-1, 1))
            names.append(f"{mix_cols[i]}*{mix_cols[j]}")

    X = np.hstack(X_parts) if X_parts else np.zeros((n, 0), dtype=float)
    return X, names


def predict_scheffe_quadratic(point: dict, coef: np.ndarray, mix_cols: list[str]) -> float:
    """
    point: dict solvent->fraction (should sum ~1)
    coef aligns with build_scheffe_quadratic_matrix()
    """
    x = np.array([float(point[c]) for c in mix_cols], dtype=float)

    row = []
    row.extend(list(x))  # linear
    for i in range(len(x)):  # pairwise
        for j in range(i + 1, len(x)):
            row.append(x[i] * x[j])

    return float(np.dot(np.array(row, dtype=float), coef))

def _read_csv_flexible(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file, sep=";")
    except Exception:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        return pd.read_csv(uploaded_file)

def ternary_to_xy(a, b, c):
    """
    Map ternary fractions (a,b,c) with a+b+c=1 to 2D coordinates of an equilateral triangle.
    Vertices:
      A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    """
    x = b + 0.5 * c
    y = (math.sqrt(3) / 2.0) * c
    return x, y


# ----------------------------------------------------------
# Session state initialization (prevents KeyError)
# ----------------------------------------------------------
defaults = {
    "coded": None,
    "design": None,
    "factor_specs": [],
    "results_df": None,

    "response_specs_classic": [],
    "response_specs_mixture": [],

    "mixture_df": None,
    "mixture_names": None,
    "mixture_n": None,
    "mixture_resolution": None,
    "results_processed_key": None,
}


for k0, v0 in defaults.items():
    if k0 not in st.session_state:
        st.session_state[k0] = v0


# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Design Type")

design_mode = st.sidebar.radio(
    "Choose Design Type:",
    [
        "Factorial Design",
        "Mixture Design (Solvent Optimization)"
    ]
)


# ==========================================================
# UI TABS
# ==========================================================

tab1, tab2, tab3 = st.tabs(["STEP 1 — Design", 
    "STEP 2 — Results", 
    "STEP 3 — Model (Quadratic) & Plots"])

# ----------------------------------------------------------
# STEP 1 — DESIGN
# ----------------------------------------------------------
with tab1:
    if design_mode == "Mixture Design (Solvent Optimization)":
        st.header("STEP 1 — Mixture Design (Extraction Solvent Optimization)")
        run_mixture_design_extraction()
    else:
        # ---- Classic DoE Step 1 (only when NOT mixture) ----
        st.header("STEP 1 — Create Experimental Design")

        st.info(
        """
**Pick a design**, define factor names and LOW/CENTER/HIGH values, then generate and download the table.

Tips:
- If you are not sure: **Box–Behnken**
- If you want axial points and better curvature estimation: **Central Composite**
- If you want fast screening with fewer runs: **Fractional factorial** or **LHS**
"""
    )

        st.markdown("### Define Response Variables")

        n_responses = st.number_input(
            "Number of response variables",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            key="n_responses"
        )

        response_specs = []

        for i in range(n_responses):
            col1, col2 = st.columns([2, 1])
            with col1:
                rname = st.text_input(
                    f"Response name #{i+1}",
                    value=f"Response_{i+1}",
                    key=f"resp_name_{i}"
                )
            with col2:
                goal = st.selectbox(
                    f"Goal",
                    ["Maximize", "Minimize"],
                    key=f"resp_goal_{i}"
                )

            response_specs.append({"name": rname, "goal": goal})

        design_type = st.selectbox(
            "Choose Design Type",
            [
                "Box-Behnken (recommended)",
                "Central Composite",
                "2-Level Full Factorial (2^k)",
                "3-Level Full Factorial (3^k)",
                "2-Level Fractional Factorial (regular)",
                "Latin Hypercube (low-resolution)",
            ],
            index=0,
        )

        k = st.slider("Number of Factors", 2, 6, 3)

        # ---------------------------------------
        # Extra settings (only used by some designs)
        # ---------------------------------------
        base_k = None
        frac_generators: list[str] = []
        lhs_runs = None
        lhs_seed = 123

        # ---------------------------------------
        # Fractional factorial settings
        # ---------------------------------------
        if design_type.startswith("2-Level Fractional"):
            st.markdown("### Fractional Factorial — Settings")

            st.info(
                """
    A fractional factorial reduces the number of experiments by **aliasing** (confounding) some effects.

    How it works:
    - Choose **base factors** (independent: A, B, C, ...)
    - Define the remaining factors as **products** of base factors (generators)

    This can shrink the design dramatically.

    Example:
    k = 5 total factors, base_k = 3  →  runs = 2³ = 8 (instead of 2⁵ = 32)
    """
            )

            # --- Base factor selection (safe version)

            if k <= 2:
                # Only possible option is full factorial
                base_k = k
                st.info("With only 2 factors, fractional design is identical to full factorial (2² = 4 runs).")
            else:
                base_k = st.slider(
                    "Number of base factors (independent A, B, C, ...)",
                    min_value=2,
                    max_value=k - 1,   # IMPORTANT FIX
                    value=min(3, k - 1),
                    key="frac_base_k",
                )

            st.success(
                f"Runs (fractional) = **{2**base_k}**   |   Runs (full 2-level) = {2**k}"
            )

            derived_k = k - base_k

            if derived_k == 0:
                st.info("No derived factors (k = base_k) → this becomes a full 2-level factorial.")
            else:
                st.markdown("#### Generators for derived factors")
                st.caption(
                    """
    Each derived factor must be defined as a product of base factors.

    Examples (valid generators):
    - AB   → A × B
    - AC   → A × C
    - ABC  → A × B × C

    Rules:
    - Use only the base letters (A..)
    - No spaces
    - Order does not matter (AB = BA)
    """
                )

                frac_generators = []
                for gi in range(derived_k):
                    frac_generators.append(
                        st.text_input(
                            f"Generator for derived factor #{gi+1}",
                            value="AB",
                            key=f"frac_gen_{gi}",
                        ).strip().upper()
                    )
                # ✅ enforce generator completeness
                frac_generators = [g for g in frac_generators if g.strip() != ""]

                frac_ok = (len(frac_generators) == derived_k)

                if not frac_ok:
                    st.error(f"You must provide exactly {derived_k} generator(s). Fill all generator boxes.")

                st.warning(
                    """
    ⚠️ Interpretation warning (aliasing):
    - In lower-resolution fractionals, some **main effects** can be confounded with **2-factor interactions**.
    - Use fractional designs mainly for **screening** (finding what matters), not final optimization.
    """
                )

        # ---------------------------------------
        # LHS settings
        # ---------------------------------------
        if design_type.startswith("Latin Hypercube"):
            st.markdown("### Latin Hypercube Sampling (LHS) — Settings")

            st.info(
                """
    LHS is a **space-filling** design.

    It is great when you have many factors and need **fewer runs** than factorial designs.
    It is NOT a classic “effect-estimation” design like factorials.

    Good for:
    - many factors (k ≥ 5)
    - expensive experiments
    - broad exploration / response surfaces
    - ML training data

    Not ideal for:
    - clean main-effect / interaction interpretation
    """
            )

            lhs_runs = st.slider(
                "Number of runs (LHS)",
                min_value=max(8, 2 * k),
                max_value=200,
                value=4 * k,
                key="lhs_runs",
            )

            st.caption(
                f"""
    Rule of thumb:
    - Minimum: 2 × k = {2*k}
    - Better: 4 × k = {4*k}
    - Very good coverage: 6–8 × k = {6*k}–{8*k}
    """
            )

            lhs_seed = st.number_input(
                "Random seed (reproducibility)",
                value=123,
                step=1,
                key="lhs_seed",
            )

            st.warning(
                """
    ⚠️ LHS does NOT guarantee:
    - orthogonality
    - balanced interaction estimation

    It guarantees:
    - uniform coverage of each factor range
    """
            )

        # ---------------------------------------
        # Factor definitions (names + low/center/high)
        # ---------------------------------------
        factor_specs = []
        for i in range(k):
            with st.expander(f"Factor {i+1}", expanded=True):
                name = st.text_input("Factor Name", value=f"Factor_{i+1}", key=f"name_{i}")
                low = st.number_input("Low (-1)", value=5.0, key=f"low_{i}")
                center = st.number_input("Center (0)", value=15.0, key=f"center_{i}")
                high = st.number_input("High (+1)", value=25.0, key=f"high_{i}")

                if name.strip() == "":
                    st.error("Factor name cannot be empty.")
                else:
                    factor_specs.append({
                        "name": name.strip(),
                        "low": low,
                        "center": center,
                        "high": high
                    })

                st.caption("Coded values: -1=Low, 0=Center, +1=High (CCD may include ±sqrt(k)).")

        # ---------------------------------------
        # Generate design
        # ---------------------------------------
        derived_k = (k - base_k) if (design_type.startswith("2-Level Fractional") and base_k is not None) else 0
        frac_ok = True

        if design_type.startswith("2-Level Fractional"):
            # must have exactly derived_k generators
            frac_ok = (base_k is not None) and (len(frac_generators) == derived_k)

        can_generate = True
        if design_type.startswith("2-Level Fractional") and not frac_ok:
            can_generate = False

        btn = st.button("Generate Design", type="primary", disabled=not can_generate)

        if btn:
            # Final validation only at click-time (keeps other tabs from going blank)
            if design_type.startswith("2-Level Fractional") and not frac_ok:
                st.error(f"Please provide exactly {derived_k} generator(s) before generating the design.")
            else:
                # Build coded design
                if design_type.startswith("Box"):
                    coded = box_behnken(k)

                elif design_type.startswith("Central"):
                    coded = central_composite(k)

                elif design_type.startswith("3-Level Full"):
                    coded = full_factorial_3level(k)

                elif design_type.startswith("2-Level Fractional"):
                    coded = fractional_factorial_2level_regular(
                        base_k=base_k,
                        generators=frac_generators
                    )

                elif design_type.startswith("Latin Hypercube"):
                    coded = lhs_design(n_runs=lhs_runs, k=k, seed=lhs_seed)

                else:
                    coded = full_factorial_2level(k)

                factor_names = [f["name"] for f in factor_specs]

                # Safety: coded columns must match factor count
                if coded.shape[1] != len(factor_names):
                    st.error(
                        f"Design produced {coded.shape[1]} columns but you defined {len(factor_names)} factors. "
                        "Check base_k and generators."
                    )
                else:
                    coded_df = pd.DataFrame(coded, columns=factor_names)

                    # Real-valued table for the lab
                    real_df = coded_df.copy()
                    for f in factor_specs:
                        col = f["name"]
                        real_df[col] = real_df[col].apply(lambda v: coded_to_real_value(float(v), f))

                    real_df.insert(0, "Experiment#", np.arange(1, len(real_df) + 1))

                    # Add response columns dynamically
                    for r in response_specs:
                        real_df[r["name"]] = ""

                    # Add combined result column (modeling target)
                    real_df["Results"] = ""

                    # Store state
                    st.session_state["response_specs_classic"] = response_specs
                    st.session_state["coded"] = coded_df
                    st.session_state["design"] = real_df
                    st.session_state["factor_specs"] = factor_specs

                    st.success(f"Design created with {len(real_df)} runs.")

        # ==========================================================
        # DESIGN VISUALIZATION
        # ==========================================================
        st.divider()
        st.subheader("Design Visualization")

        coded_df = st.session_state.get("coded")
        real_df = st.session_state.get("design")
        factor_specs_state = st.session_state.get("factor_specs", [])

        if coded_df is None or real_df is None or len(factor_specs_state) == 0:
            st.info("Generate a design first (click **Generate Design**) to see the design plots.")
        else:
            factor_cols = [f["name"] for f in factor_specs_state]

            space = st.radio(
                "Plot design in:",
                ["Coded Space (-1 to +1)", "Real Units"],
                horizontal=True,
                key="design_space",
            )

            plot_df = coded_df.copy() if space.startswith("Coded") else real_df[factor_cols].copy()

            col1, col2 = st.columns(2)
            with col1:
                fx = st.selectbox("X axis", factor_cols, key="design_fx")
            with col2:
                fy = st.selectbox("Y axis", factor_cols, index=1 if len(factor_cols) > 1 else 0, key="design_fy")

            # (optional) Add a simple 2D plot — useful even with 3+ factors
            #st.markdown("### 2D Projection")
            #fig2d = px.scatter(plot_df, x=fx, y=fy, text=np.arange(1, len(plot_df) + 1),
            #                   title=f"Design Points — {fx} vs {fy}")
            #fig2d.update_traces(marker=dict(size=10))
            #fig2d.update_layout(height=520)
            #st.plotly_chart(fig2d, use_container_width=True)

            if len(factor_cols) >= 3:
                st.markdown("### 3D Projection")
                fz = st.selectbox("Z axis", factor_cols, index=2, key="design_fz")
                fig3d = px.scatter_3d(plot_df, x=fx, y=fy, z=fz, text=np.arange(1, len(plot_df) + 1),
                                      title=f"3D Design — {fx}, {fy}, {fz}")
                fig3d.update_traces(marker=dict(size=6))
                fig3d.update_layout(height=650)
                st.plotly_chart(fig3d, use_container_width=True)

        # Experimental Table
        design_df = st.session_state.get("design")

        if design_df is not None:
            st.subheader("Experimental Table (Real units for the lab)")
            st.dataframe(design_df, use_container_width=True, height=380)

            st.download_button(
                "Download Experimental Table (CSV ;)",
                design_df.to_csv(index=False, sep=";"),
                file_name="Experimental_Table.csv",
                mime="text/csv",
            )
        else:
            st.info("Generate a design to see the experimental table.")


# ----------------------------------------------------------
# STEP 2 — RESULTS (TAB 2)  [PREVIEW ONLY — NO UPLOAD HERE]
# ----------------------------------------------------------
with tab2:
    st.header("STEP 2 — Results")

    df = st.session_state.get("results_df")

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Upload your completed CSV in the **sidebar** (Upload Results).")
    else:
        st.subheader("Preview uploaded data")
        st.dataframe(df, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            if "Results" in df.columns:
                fig = px.histogram(df, x="Results", nbins=20, title="Results distribution")
                st.plotly_chart(fig, use_container_width=True)
                download_plotly_html(fig, "results_distribution.html", "Download as HTML")
            else:
                st.info("No 'Results' column found yet (compute it in the sidebar).")

        with c2:
            if "Results" not in df.columns:
                st.info("No 'Results' column found yet.")
            else:
                resp_specs = (
                    st.session_state.get("response_specs_mixture", [])
                    if design_mode == "Mixture Design (Solvent Optimization)"
                    else st.session_state.get("response_specs_classic", [])
                )
                candidates = [r["name"] for r in resp_specs if r.get("name") and r["name"] in df.columns]

                if not candidates:
                    st.info("No response columns detected for plotting (check response names and file columns).")
                else:
                    xcol = st.selectbox("X axis (response)", candidates, index=0, key="tab2_xcol")
                    fig = px.scatter(df, x=xcol, y="Results", color="Results", title=f"{xcol} vs Results")
                    st.plotly_chart(fig, use_container_width=True)
                    download_plotly_html(fig, "response_vs_results.html", "Download as HTML")               
                    
# ----------------------------------------------------------
# STEP 2 — RESULTS (SIDEBAR)
# ----------------------------------------------------------
st.sidebar.header("Upload Results")



# =========================
# Mixture path (SIDEBAR ONLY)
# =========================
if design_mode == "Mixture Design (Solvent Optimization)":
    mix_df = st.session_state.get("mixture_df")
    mix_names = st.session_state.get("mixture_names")

    if mix_df is None or mix_names is None:
        st.sidebar.info("Generate the mixture design in STEP 1 first.")
    else:
        uploaded = st.sidebar.file_uploader(
            "Upload Completed CSV (separator ';' recommended)",
            type=["csv"],
            key="results_upload_mixture",
        )

        if uploaded is None:
            st.session_state["results_processed_key"] = None
            st.sidebar.info("Upload your completed mixture results CSV here.")
        else:
            df = _read_csv_flexible(uploaded)

            # -------------------------------------------------
            # Validate mixture columns (no return; no st.stop)
            # -------------------------------------------------
            mix_ready = True
            missing_mix = [c for c in mix_names if c not in df.columns]
            if missing_mix:
                st.sidebar.error(
                    "Missing mixture columns in file:\n"
                    f"{missing_mix}\n\n"
                    f"Your file must include the solvent fraction columns: {mix_names}"
                )
                mix_ready = False

            # -------------------------------------------------
            # Define responses (still allow UI, but block compute)
            # -------------------------------------------------
            st.sidebar.markdown("### Define Response Variables (Mixture)")

            n_responses = st.sidebar.number_input(
                "Number of response variables",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                key="mix_n_responses",
            )

            response_specs = []
            for i in range(int(n_responses)):
                rname = st.sidebar.text_input(
                    f"Response name #{i+1}",
                    value=f"Response_{i+1}",
                    key=f"mix_resp_name_{i}",
                ).strip()

                goal = st.sidebar.selectbox(
                    f"Goal #{i+1}",
                    ["Maximize", "Minimize"],
                    key=f"mix_resp_goal_{i}",
                )

                response_specs.append({"name": rname, "goal": goal})

            required = [r["name"] for r in response_specs if r["name"]]

            if not required:
                st.sidebar.warning("Please provide at least one response name.")
                mix_ready = False

            missing_resp = [c for c in required if c not in df.columns]
            if missing_resp:
                st.sidebar.error(f"Missing response columns: {missing_resp}")
                mix_ready = False

            # -------------------------------------------------
            # Compute Results (Mixture) only if everything is OK
            # -------------------------------------------------
            if mix_ready:
                st.sidebar.markdown("### Compute Results (Mixture)")

                score = np.zeros(len(df), dtype=float)

                for r in response_specs:
                    col = r["name"]
                    values = pd.to_numeric(df[col], errors="coerce")

                    if values.notna().sum() < 3:
                        st.sidebar.error(f"Response '{col}' has too few numeric values.")
                        mix_ready = False
                        break

                    mu = float(values.mean(skipna=True))
                    sd = float(values.std(skipna=True))
                    if not np.isfinite(sd) or sd == 0.0:
                        sd = 1.0

                    z = (values - mu) / sd
                    zv = z.fillna(0.0).to_numpy(dtype=float)

                    weight = st.sidebar.number_input(
                        f"Weight for {col}",
                        value=1.0,
                        key=f"mix_weight_{col}",
                    )

                    score += (weight * zv) if r["goal"] == "Maximize" else (-weight * zv)

                if mix_ready:
                    df_out = df.copy()
                    df_out["Results"] = score

                    st.session_state["response_specs_mixture"] = response_specs
                    st.session_state["results_df"] = df_out
                    # ✅ build a stable key for this uploaded file
                    processed_key = f"mixture::{uploaded.name}::{uploaded.size}::{response_specs}"

                    # ✅ only rerun once per file (prevents infinite loop)
                    if st.session_state.get("results_processed_key") != processed_key:
                        st.session_state["response_specs_mixture"] = response_specs
                        st.session_state["results_df"] = df_out
                        st.session_state["results_processed_key"] = processed_key

                        st.sidebar.success("Mixture results stored ✅ Go to STEP 2 tab to preview.")
                        st.rerun()
                    else:
                        # already processed; do NOT rerun again
                        st.session_state["response_specs_mixture"] = response_specs
                        st.session_state["results_df"] = df_out
                        st.sidebar.success("Mixture results already computed for this file ✅")
            else:
                st.sidebar.info("Fix the issues above to compute Results.")

# =========================
# Classic DoE path (your current logic)
# =========================
else:
    # Guard: design must exist
    if st.session_state.get("design") is None:
        st.sidebar.info("Generate the design in STEP 1 first.")
    else:
        uploaded = st.sidebar.file_uploader(
            "Upload Completed CSV (separator ';')",
            type=["csv"],
            key="results_upload",
        )

        if uploaded is None:
            st.session_state["results_processed_key"] = None
            st.sidebar.info("Upload your completed results CSV here.")
        else:
            df = _read_csv_flexible(uploaded)

            response_specs = st.session_state.get("response_specs_classic", [])
            required = [r["name"] for r in response_specs]
            missing = [c for c in required if c not in df.columns]

            if missing:
                st.sidebar.error(f"Missing columns: {missing}")
            else:
                st.sidebar.markdown("### Compute Results")
                score = np.zeros(len(df), dtype=float)

                for r in response_specs:
                    values = pd.to_numeric(df[r["name"]], errors="coerce")
                    std = float(values.std()) if float(values.std()) != 0.0 else 1.0
                    z = (values - values.mean()) / std

                    weight = st.sidebar.number_input(
                        f"Weight for {r['name']}",
                        value=1.0,
                        key=f"weight_{r['name']}",
                    )

                    zv = z.fillna(0.0).to_numpy(dtype=float)
                    score += (weight * zv) if r["goal"] == "Maximize" else (-weight * zv)

                df["Results"] = score
                st.session_state["results_df"] = df
                # ✅ build a stable key for this uploaded file
                processed_key = f"classic::{uploaded.name}::{uploaded.size}::{response_specs}"

                # ✅ only rerun once per file (prevents infinite loop)
                if st.session_state.get("results_processed_key") != processed_key:
                    st.session_state["results_df"] = df
                    st.session_state["results_processed_key"] = processed_key

                    st.sidebar.success("Results computed and stored.")
                    st.rerun()
                else:
                    st.session_state["results_df"] = df
                    st.sidebar.success("Results already computed for this file ✅")

        

# ----------------------------------------------------------
# STEP 3 — MODEL & VISUALIZE
# ----------------------------------------------------------
with tab3:
    # ======================================================
    # MIXTURE MODE (Scheffé Quadratic)
    # ======================================================
    if design_mode == "Mixture Design (Solvent Optimization)":
        st.header("STEP 3 — Model & Visualize (Mixture — Scheffé Quadratic)")

        df_state = st.session_state.get("results_df")
        mix_df = st.session_state.get("mixture_df")
        mix_cols = st.session_state.get("mixture_names")

        if df_state is None or not isinstance(df_state, pd.DataFrame) or df_state.empty:
            st.info("Upload mixture results in STEP 2 (sidebar) first.")
            st.stop()

        if mix_df is None or mix_cols is None:
            st.error("Missing mixture design. Generate the mixture design in STEP 1 first.")
            st.stop()

        if "Results" not in df_state.columns:
            st.info("No 'Results' column yet. Compute Results in the sidebar first.")
            st.stop()

        df = df_state.copy()

        # keep only rows with finite Results + finite mixture fractions
        y = pd.to_numeric(df["Results"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y)
        for c in mix_cols:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            ok = ok & np.isfinite(v)

        if ok.sum() < max(6, 2 * len(mix_cols)):
            st.error("Not enough valid rows to fit the mixture model.")
            st.stop()

        df_fit = df.loc[ok, :].reset_index(drop=True)

        row_sum = df_fit[mix_cols].sum(axis=1)
        bad = (row_sum - 1.0).abs() > 1e-6
        if bad.any():
            st.warning(f"{int(bad.sum())} row(s) have mixture fractions not summing to 1. Check your CSV.")

        # --- fit Scheffé quadratic
        X, term_names = build_scheffe_quadratic_matrix(df_fit, mix_cols)
        coef, yhat = fit_lstsq(X, df_fit["Results"].to_numpy(dtype=float))
        r2v = r2_score(df_fit["Results"].to_numpy(dtype=float), yhat)

        st.subheader("Model quality")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("R² (in-sample)", f"{r2v:.3f}")
        with c2:
            st.caption("Mixture model: linear + pairwise interaction terms (Scheffé quadratic).")

        st.subheader("Observed vs Predicted")
        ovp = pd.DataFrame({"Observed": df_fit["Results"].to_numpy(dtype=float), "Predicted": yhat})
        fig_ovp = px.scatter(
            ovp, x="Observed", y="Predicted", trendline="ols",
            title="Observed vs Predicted (Mixture)"
        )
        st.plotly_chart(fig_ovp, use_container_width=True)
        download_plotly_html(fig_ovp, "mixture_observed_vs_predicted.html", "Download as HTML")

        st.subheader("Coefficient magnitudes (Pareto-like view)")
        coef_df = pd.DataFrame({"Term": term_names, "Coefficient": coef})
        coef_df["Abs"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("Abs", ascending=False).reset_index(drop=True)
        fig_coef = px.bar(
            coef_df.head(20), x="Term", y="Abs",
            title="Top coefficient magnitudes (|coef|) — Mixture"
        )
        st.plotly_chart(fig_coef, use_container_width=True)
        download_plotly_html(fig_coef, "mixture_coefficients_pareto.html", "Download as HTML")

        st.divider()
        st.subheader("Mixture space visualization (predicted)")

        # use the same lattice points you generated in Step 1 for visualization
        grid_df = mix_df[mix_cols].copy()
        Zp = []
        for _, row in grid_df.iterrows():
            pt = {c: float(row[c]) for c in mix_cols}
            Zp.append(predict_scheffe_quadratic(pt, coef, mix_cols))
        grid_df["Predicted"] = np.array(Zp, dtype=float)

        n = len(mix_cols)

        if n == 2:
            fig = px.line(
                grid_df.sort_values(mix_cols[0]),
                x=mix_cols[0], y="Predicted",
                title="Predicted response across binary mixture"
            )
            st.plotly_chart(fig, use_container_width=True)
            download_plotly_html(fig, "mixture_predicted_binary.html", "Download as HTML")

        elif n == 3:
            # --- map ternary -> XY
            xs, ys = [], []
            for _, r in grid_df.iterrows():
                a = float(r[mix_cols[0]])
                b = float(r[mix_cols[1]])
                c = float(r[mix_cols[2]])
                x, y = ternary_to_xy(a, b, c)
                xs.append(x)
                ys.append(y)

            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            zs = grid_df["Predicted"].to_numpy(dtype=float)

            # --- triangulation (INTEGER indices)
            pts2 = np.column_stack([xs, ys])
            tri = Delaunay(pts2)
            simplices = tri.simplices.astype(int)

            st.markdown("### 3D Response Surface (Ternary)")
            fig3d = ff.create_trisurf(
                x=xs, y=ys, z=zs,
                simplices=simplices,
                title="Predicted response surface (ternary mixture)",
                show_colorbar=True,
            )
            fig3d.update_layout(
                scene=dict(
                    xaxis_title=mix_cols[1],
                    yaxis_title=mix_cols[2],
                    zaxis_title="Predicted",
                ),
                height=720,
                margin=dict(l=0, r=0, t=50, b=0),
            )
            st.plotly_chart(fig3d, use_container_width=True)
            download_plotly_html(fig3d, "mixture_surface3D_ternary.html", "Download 3D Surface as HTML")

            st.markdown("### 2D Contour (on ternary triangle)")
            fig2d = go.Figure()
            fig2d.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="markers",
                    marker=dict(size=8, color=zs, colorscale="Viridis", showscale=True),
                    text=[
                        f"{mix_cols[0]}={grid_df.iloc[i][mix_cols[0]]:.3f}<br>"
                        f"{mix_cols[1]}={grid_df.iloc[i][mix_cols[1]]:.3f}<br>"
                        f"{mix_cols[2]}={grid_df.iloc[i][mix_cols[2]]:.3f}<br>"
                        f"Pred={zs[i]:.4f}"
                        for i in range(len(grid_df))
                    ],
                    hoverinfo="text",
                    name="Grid points",
                )
            )

            # draw triangle boundary
            A = (0.0, 0.0)
            B = (1.0, 0.0)
            C = (0.5, math.sqrt(3) / 2.0)
            fig2d.add_trace(
                go.Scatter(
                    x=[A[0], B[0], C[0], A[0]],
                    y=[A[1], B[1], C[1], A[1]],
                    mode="lines",
                    line=dict(width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            fig2d.update_layout(
                title="Predicted response (ternary) — point-colored map",
                xaxis_title="(B + 0.5·C)",
                yaxis_title="(√3/2 · C)",
                height=620,
            )
            fig2d.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig2d, use_container_width=True)
            download_plotly_html(fig2d, "mixture_contour2D_ternary.html", "Download 2D Map as HTML")

        # IMPORTANT: do not fall through to Classic DoE
        st.stop()

    # ======================================================
    # CLASSIC DOE MODE (Quadratic)
    # ======================================================
    st.header("STEP 3 — Model & Visualize (Quadratic)")

    df_state = st.session_state.get("results_df")
    coded_df = st.session_state.get("coded")
    factor_specs = st.session_state.get("factor_specs", [])

    if df_state is None or not isinstance(df_state, pd.DataFrame) or df_state.empty:
        st.info("Upload results in STEP 2 (sidebar) first.")
        st.stop()

    if coded_df is None or len(factor_specs) == 0:
        st.error("Missing design metadata. Generate a design in STEP 1 first.")
        st.stop()

    if "Results" not in df_state.columns:
        st.info("No 'Results' column yet. Compute Results in the sidebar first.")
        st.stop()

    df = df_state.copy()
    coded_df = coded_df.copy()

    factor_cols = [f["name"] for f in factor_specs]

    if len(df) != len(coded_df):
        st.warning(
            "Uploaded table row count differs from the coded design. "
            "Fitting will use matching first N rows."
        )
        n = min(len(df), len(coded_df))
        df = df.iloc[:n].reset_index(drop=True)
        coded_df = coded_df.iloc[:n].reset_index(drop=True)

    y = pd.to_numeric(df["Results"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y)

    if ok.sum() < max(8, 2 * len(factor_cols)):
        st.error("Not enough valid Results values to fit a quadratic model.")
        st.stop()

    df_fit = df.loc[ok].reset_index(drop=True)
    coded_fit = coded_df.loc[ok].reset_index(drop=True)

    X, term_names = build_quadratic_matrix(coded_fit, factor_cols)
    coef, yhat = fit_lstsq(X, df_fit["Results"].to_numpy(dtype=float))
    r2v = r2_score(df_fit["Results"].to_numpy(dtype=float), yhat)

    st.subheader("Model quality")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("R² (in-sample)", f"{r2v:.3f}")
    with c2:
        st.caption("If R² is low: either the response is noisy or your factor ranges are too narrow/too wide.")

    st.subheader("Observed vs Predicted")
    ovp = pd.DataFrame({"Observed": df_fit["Results"].to_numpy(dtype=float), "Predicted": yhat})
    fig = px.scatter(ovp, x="Observed", y="Predicted", trendline="ols", title="Observed vs Predicted")
    st.plotly_chart(fig, use_container_width=True)
    download_plotly_html(fig, "observed_vs_predicted.html", "Download as HTML")

    st.subheader("Residuals (with ±1σ and ±2σ bands)")
    resid = (ovp["Observed"] - ovp["Predicted"]).to_numpy(dtype=float)
    pred = ovp["Predicted"].to_numpy(dtype=float)
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    order = np.argsort(pred)
    pred_s = pred[order]

    fig_res = go.Figure()

    fig_res.add_trace(
        go.Scatter(
            x=np.concatenate([pred_s, pred_s[::-1]]),
            y=np.concatenate([2 * sigma * np.ones_like(pred_s), (-2 * sigma) * np.ones_like(pred_s)[::-1]]),
            fill="toself",
            name="±2σ band",
            hoverinfo="skip",
            line=dict(width=0),
            opacity=0.18,
            showlegend=True,
        )
    )
    fig_res.add_trace(
        go.Scatter(
            x=np.concatenate([pred_s, pred_s[::-1]]),
            y=np.concatenate([1 * sigma * np.ones_like(pred_s), (-1 * sigma) * np.ones_like(pred_s)[::-1]]),
            fill="toself",
            name="±1σ band",
            hoverinfo="skip",
            line=dict(width=0),
            opacity=0.28,
            showlegend=True,
        )
    )

    for yline, nm in [(0.0, "0"), (sigma, "+1σ"), (-sigma, "-1σ"), (2 * sigma, "+2σ"), (-2 * sigma, "-2σ")]:
        fig_res.add_trace(
            go.Scatter(
                x=[pred_s.min(), pred_s.max()],
                y=[yline, yline],
                mode="lines",
                name=nm,
                hoverinfo="skip",
                line=dict(dash="dash", width=1),
                showlegend=(nm in ["0", "+1σ", "+2σ"]),
            )
        )

    fig_res.add_trace(
        go.Scatter(
            x=pred,
            y=resid,
            mode="markers",
            name="Residuals",
            marker=dict(size=8),
        )
    )

    fig_res.update_layout(
        title="Residuals vs Predicted (shaded ±1σ and ±2σ)",
        xaxis_title="Predicted",
        yaxis_title="Residual",
        height=520,
    )

    st.plotly_chart(fig_res, use_container_width=True)
    download_plotly_html(fig_res, "residuals_plot.html", "Download as HTML")

    st.subheader("Coefficient magnitudes (Pareto-like view)")
    coef_df = pd.DataFrame({"Term": term_names, "Coefficient": coef})
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs", ascending=False).reset_index(drop=True)
    fig = px.bar(coef_df.head(20), x="Term", y="Abs", title="Top coefficient magnitudes (|coef|)")
    st.plotly_chart(fig, use_container_width=True)
    download_plotly_html(fig, "coefficients_pareto.html", "Download as HTML")

    st.divider()
    st.subheader("Response surfaces (2D contour + 3D surface)")

    colA, colB, colC = st.columns([1.2, 1.2, 1.6])
    with colA:
        fx = st.selectbox("X factor", factor_cols, index=0)
    with colB:
        fy = st.selectbox("Y factor", factor_cols, index=1 if len(factor_cols) > 1 else 0)
    with colC:
        space = st.radio("Plot axis units", ["Coded (-1..+1)", "Real units"], horizontal=True)

    fixed = {}
    for f in factor_cols:
        if f not in (fx, fy):
            fixed[f] = st.slider(f"Fix {f} (coded)", -1.5, 1.5, 0.0, 0.05)

    grid_n = st.slider("Surface grid resolution", 25, 90, 55)
    gx = np.linspace(-1.5, 1.5, grid_n)
    gy = np.linspace(-1.5, 1.5, grid_n)

    Z = np.zeros((grid_n, grid_n), dtype=float)
    for i, xv in enumerate(gx):
        for j, yv in enumerate(gy):
            point = {f: fixed.get(f, 0.0) for f in factor_cols}
            point[fx] = float(xv)
            point[fy] = float(yv)
            Z[j, i] = predict_quadratic(point, coef, factor_cols)

    if space == "Real units":
        spec_x = next(s for s in factor_specs if s["name"] == fx)
        spec_y = next(s for s in factor_specs if s["name"] == fy)
        gx_plot = np.array([coded_to_real_value(v, spec_x) for v in gx], dtype=float)
        gy_plot = np.array([coded_to_real_value(v, spec_y) for v in gy], dtype=float)
        x_label = fx
        y_label = fy
    else:
        gx_plot = gx
        gy_plot = gy
        x_label = f"{fx} (coded)"
        y_label = f"{fy} (coded)"

    fig2d = go.Figure(
        data=go.Contour(x=gx_plot, y=gy_plot, z=Z, contours_coloring="heatmap", showscale=True)
    )
    fig2d.update_layout(
        title=f"2D Contour — Predicted Results — {fx} vs {fy}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=560,
    )
    st.plotly_chart(fig2d, use_container_width=True)
    download_plotly_html(fig2d, f"contour_{fx}_vs_{fy}.html", "Download Contour as HTML")

    fig3d = go.Figure(data=[go.Surface(x=gx_plot, y=gy_plot, z=Z)])
    fig3d.update_layout(
        title=f"3D Surface — Predicted Results — {fx} vs {fy}",
        scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title="Predicted Results"),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)
    download_plotly_html(fig3d, f"surface3D_{fx}_vs_{fy}.html", "Download 3D Surface as HTML")

    st.divider()
    st.subheader("Optimization (grid search in coded space)")
    search_min, search_max = st.slider("Coded search range", -2.0, 2.0, (-1.0, 1.0), 0.1)
    step = st.slider("Grid step (smaller = slower)", 0.02, 0.3, 0.08, 0.01)

    if st.button("Find best conditions", type="primary"):
        grid = np.arange(search_min, search_max + 1e-9, step)
        total = int(len(grid) ** len(factor_cols))
        if total > 2_000_000:
            st.error(
                f"Grid search too large: {total:,} evaluations. "
                "Increase step, reduce range, or reduce number of factors."
            )
            st.stop()

        best_val = -np.inf
        best_point = None

        for point_tuple in itertools.product(grid, repeat=len(factor_cols)):
            point = {factor_cols[i]: float(point_tuple[i]) for i in range(len(factor_cols))}
            val = predict_quadratic(point, coef, factor_cols)
            if val > best_val:
                best_val = val
                best_point = point

        st.success(f"Best predicted Results: {best_val:.4f}")
        st.write("Best coded point:", best_point)

        real_best = {}
        for spec in factor_specs:
            real_best[spec["name"]] = coded_to_real_value(best_point[spec["name"]], spec)
        st.write("Best real conditions:", real_best)