# 📘 Multivariate Data Analysis – Detailed Teaching Guide

> This document is designed as a conceptual and practical guide for students learning multivariate data analysis in metabolomics and related biological sciences.

---

# 1️⃣ What Is Multivariate Data?

## Definition: Variable

A **variable** (also called a feature) is a measurable quantity.

Examples:

* Metabolite intensity
* Peak area
* Gene expression value
* NMR bucket

---

## Definition: Sample

A **sample** is one observation containing many variables.

In metabolomics:

* One injection
* One biological replicate
* One individual

---

## Definition: Multivariate Data

Data are **multivariate** when each sample contains multiple variables.

Mathematically:

If we have:

* *n* samples
* *p* variables

Then our dataset is an **n × p matrix**.

Each sample is a point in **p-dimensional space**.

---

## Why Multivariate Analysis?

Univariate analysis:

* Tests one variable at a time.

Multivariate analysis:

* Considers all variables simultaneously.
* Accepts the fact that different variable affects the system in combinations.
* Interactions among different variables.
* Detects patterns that only emerge in combination.

In biology:

> Most biological phenomena are not driven by a single variable, but by coordinated changes.

---

# 2️⃣ Data Structure and Format

## Data Matrix Structure

We typically organize data as:

| Feature ↓ / Sample → | S1  | S2  | S3  | S4  |
| -------------------- | --- | --- | --- | --- |
| Metabolite 1         | x11 | x12 | x13 | x14 |
| Metabolite 2         | x21 | x22 | x23 | x24 |

---

## Definition: Data Matrix

A **data matrix** is a rectangular table of numeric values representing measurements.

Notation:

* X = data matrix
* xᵢⱼ = value of feature i in sample j

---

## Group Labels

When working with experimental groups:

* Control
* Treated
* Disease
* Healthy

These labels are stored separately (e.g., `ATTRIBUTE_class`).

Important:

> Group labels are not used in unsupervised methods.

---

# 3️⃣ Data Inspection

Before any transformation:

## Definition: Exploratory Data Analysis (EDA)

EDA is the process of examining data to understand:

* Distribution
* Missingness
* Outliers
* Variability

---

## Definition: Distribution

A **distribution** describes how values are spread across a range.

Common properties:

* Symmetry
* Skewness
* Presence of extreme values

Metabolomics data are often:

* Right-skewed
* Heteroscedastic

---

## Definition: Heteroscedasticity

Heteroscedasticity means:

> The variance changes with the mean.

Large peaks tend to have larger variance.

This is one reason transformations are applied.

---

# 4️⃣ Missing Values

## Definition: Missing Value

A missing value is an observation that has no recorded numeric value.

Represented as:

* NA
* null
* empty cell

---

## Why Missing Values Occur

* Below detection limit
* Peak detection failure
* Alignment failure
* True biological absence

---

## Definition: Missingness Mechanisms

There are three conceptual types:

1. MCAR – Missing Completely At Random
2. MAR – Missing At Random
3. MNAR – Missing Not At Random

In metabolomics, missingness is often MNAR (low abundance).

---

## Imputation

### Definition: Imputation

Imputation is the process of replacing missing values with estimated values.

---

### Common Imputation Methods

#### Mean Imputation

Replace missing values with the average of observed values.

Effect:

* Reduces variance artificially.

#### Median Imputation

Replace missing values with the median.

More robust to outliers.

#### Constant Imputation

Replace missing values with zero or small value.

Assumes missing = absence.

---

# 5️⃣ Transformation

## Definition: Data Transformation

A mathematical operation applied to data to change its scale or distribution.

---

## Log Transformation

### Definition

Log transformation replaces each value x with log(x).

Common base:

* log10
* natural log (ln)

---

### Why Use Log?

Log compresses large values more than small values.

Example:

* log(1000) vs log(10)
* Difference becomes smaller.

Effect:

* Reduces skewness
* Reduces heteroscedasticity

---

## Important Concept: Monotonic Transformation

Log transformation is monotonic:

* Order of values is preserved.
* Relative ranking remains the same.

---

# 6️⃣ Normalization

## Definition: Normalization

Normalization adjusts values across samples to make them comparable.

It corrects systematic sample-to-sample differences.

---

## Types of Normalization

### Total Sum Normalization

Each sample is divided by its total intensity.

Purpose:

* Correct injection differences.

---

### Internal Standard Normalization

Each feature is divided by the signal of a reference compound.

More chemically meaningful.

---

### PQN (Probabilistic Quotient Normalization)

Common in NMR.
Adjusts based on median fold change.

---

## Important Distinction

Normalization acts across samples.

Scaling acts across variables.

These are different operations.

---

# 7️⃣ Scaling

## Definition: Scaling

Scaling adjusts the variance of variables so that they contribute equally (or more equally) to the model.

---

## Mean-Centering

### Definition

Subtract the mean of each variable.

Formula:
xᵢⱼ → xᵢⱼ − mean(xᵢ)

Effect:

* Data centered around zero.

---

## Autoscaling (Unit Variance Scaling)

### Definition

Subtract mean and divide by standard deviation.

Formula:
xᵢⱼ → (xᵢⱼ − mean(xᵢ)) / sd(xᵢ)

Effect:

* All variables have variance = 1.

---

## Pareto Scaling

Divide by square root of standard deviation.

Compromise between no scaling and autoscaling.

---

## Why Scaling Matters

Without scaling:

* High-intensity metabolites dominate PCA.

With scaling:

* Small metabolites gain influence.

Scaling is a modeling decision.

---

# 8️⃣ PCA (Principal Component Analysis)

## Definition

PCA is an unsupervised dimensionality reduction method.

It finds linear combinations of variables that capture maximal variance.

---

## Definition: Principal Component

A principal component is:

> A weighted linear combination of original variables.

PC1 = w1x1 + w2x2 + ... + wpxp

Weights are chosen to maximize variance.

---

## Definition: Variance

Variance measures how spread out data are.

High variance = large dispersion.

PCA captures directions of maximum variance.

---

## Scores and Loadings

### Score

Coordinates of samples in PC space.

Represent:

* Samples

---

### Loading

Weights that define each principal component.

Represent:

* Variable contributions

---

## Orthogonality

Principal components are orthogonal:

* They are statistically independent.
* Dot product = 0.

---

# 9️⃣ Supervised Methods

## Definition: Supervised Learning

A method that uses known class labels during model training.

Example:

* PLS-DA

---

## PLS-DA

### Definition

Partial Least Squares Discriminant Analysis.

A regression-based method adapted for classification.

It finds components that:

* Maximize covariance between X (data) and Y (class labels).

---

## Covariance

Covariance measures how two variables vary together.

High covariance:

* When one increases, the other tends to increase.

PLS-DA maximizes covariance between features and class membership.

---

# 🔟 Model Evaluation

## R²

Explained variance.

Measures how well model fits training data.

---

## Q²

Predictive ability estimated by cross-validation.

Measures generalizability.

---

## Overfitting

### Definition

Overfitting occurs when a model learns noise instead of signal.

Characteristics:

* Excellent training performance
* Poor prediction performance

---

## Cross-Validation

Repeatedly splitting data into training and test subsets.

Purpose:

* Estimate model robustness.

---

## Permutation Test

Randomly shuffle class labels and refit model.

If shuffled model performs similarly:

* Original model not meaningful.

---

# 1️⃣1️⃣ VIP (Variable Importance in Projection)

## Definition

VIP quantifies how much each variable contributes to class separation in PLS-DA.

Common rule:
VIP > 1 = important.

But this is heuristic, not absolute truth.

---

# 1️⃣2️⃣ Outliers

## Definition

An outlier is a sample that deviates strongly from others.

Possible causes:

* Biological uniqueness
* Experimental error
* Instrumental issue

Outliers must be investigated, not blindly removed.

---

# 1️⃣3️⃣ Dimensionality Reduction

## Definition

Reducing number of variables while retaining essential information.

Original dimension = p
Reduced dimension = k (k << p)

Benefits:

* Visualization
* Noise reduction
* Computational efficiency

---

# 1️⃣4️⃣ Biological Interpretation

Statistics detect structure.

Interpretation assigns meaning.

Ask:

* Are discriminant features chemically related?
* Do they share biosynthetic pathways?
* Do they match phenotype?

Multivariate analysis:

> Is a structured way of asking better biological questions.

---

# 📌 Final Teaching Principle

Students must understand:

* Every preprocessing step is a modeling decision.
* Every transformation changes the geometry of the data.
* Every supervised model must be validated.
* Multivariate analysis reveals patterns — it does not create truth.

---
---

# 📐 Block 1 — Mathematical Foundations (Matrix Notation)

> This section formalizes the main operations used in multivariate analysis using matrix notation.

---

## 1️⃣ The Data Matrix

Let:

* ( n ) = number of samples
* ( p ) = number of variables

We define the data matrix:

$$
\mathbf{X} \in \mathbb{R}^{n \times p}
$$

$$
\mathbf{X} =
\begin{bmatrix}
x_{11} & x_{12} & \dots & x_{1p} \\\\
x_{21} & x_{22} & \dots & x_{2p} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
x_{n1} & x_{n2} & \dots & x_{np}
\end{bmatrix}
$$

* Rows = samples
* Columns = variables

Each row is a vector in ( \mathbb{R}^p ).

---

## 2️⃣ Mean-Centering

The mean of variable ( j ):

$$
\bar{x}_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}
$$

Mean-centered data:

$$
x'*{ij} = x*{ij} - \bar{x}_j
$$

Matrix form:

$$
\mathbf{X}_c = \mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^T
$$

where:

* ( \mathbf{1} ) is a column vector of ones
* ( \bar{\mathbf{x}} ) is the vector of variable means

---

## 3️⃣ Autoscaling (Unit Variance Scaling)

Standard deviation of variable ( j ):

$$
s_j = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_{ij} - \bar{x}_j)^2}
$$

Scaled value:

$$
x''*{ij} = \frac{x*{ij} - \bar{x}_j}{s_j}
$$

Matrix form:

$$
\mathbf{X}_{scaled} = \mathbf{X}_c \mathbf{D}^{-1}
$$

Where:

* ( \mathbf{D} ) = diagonal matrix of standard deviations

---

## 4️⃣ Covariance Matrix

The covariance matrix:

$$
\mathbf{S} = \frac{1}{n-1} \mathbf{X}_c^T \mathbf{X}_c
$$

Dimensions:

$$
\mathbf{S} \in \mathbb{R}^{p \times p}
$$

Interpretation:

* Diagonal elements → variances
* Off-diagonal elements → covariances

---

## 5️⃣ PCA as Eigen Decomposition

PCA finds eigenvectors of the covariance matrix:

$$
\mathbf{S} \mathbf{v}_k = \lambda_k \mathbf{v}_k
$$

Where:

* ( \mathbf{v}_k ) = eigenvector (loading vector)
* ( \lambda_k ) = eigenvalue (variance explained)

Principal components are ordered:

$$
\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p
$$

---

## 6️⃣ Scores

Scores are projections of data onto eigenvectors:

$$
\mathbf{T} = \mathbf{X}_c \mathbf{V}
$$

Where:

* ( \mathbf{V} ) = matrix of eigenvectors
* ( \mathbf{T} ) = score matrix

Each column of ( \mathbf{T} ) is a principal component.

---

## 7️⃣ Variance Explained

Proportion of variance explained by PC k:

$$
\frac{\lambda_k}{\sum_{j=1}^{p} \lambda_j}
$$

---

## 8️⃣ PLS-DA Concept (Matrix Form)

PLS models:

$$
\mathbf{X} = \mathbf{T}\mathbf{P}^T + \mathbf{E}
$$
$$
\mathbf{Y} = \mathbf{T}\mathbf{C}^T + \mathbf{F}
$$

Where:

* ( \mathbf{T} ) = latent scores
* ( \mathbf{P} ) = loadings for X
* ( \mathbf{C} ) = loadings for Y
* ( \mathbf{E}, \mathbf{F} ) = residuals

PLS maximizes covariance between ( \mathbf{X} ) and ( \mathbf{Y} ).

---
---

# 📐 Block 2 — Conceptual Geometry of Multivariate Analysis

> Understanding geometry helps students truly grasp PCA and PLS-DA.

---

## 1️⃣ Samples as Points in Space

If we have:

* 2 variables → 2D space
* 3 variables → 3D space
* p variables → p-dimensional space

Each sample is a point.

Distance between samples represents similarity.

---

## 2️⃣ What PCA Really Does (Geometrically)

PCA:

* Rotates the coordinate system
* Finds new axes
* Aligns axes with directions of maximum variance

It does NOT move the data.
It rotates the space.

---

📌 *[Insert 2D rotation illustration here]*

---

## 3️⃣ Projection

### Definition: Projection

Projection means dropping a perpendicular from a point onto a line or plane.

In PCA:

Each sample is projected onto a principal component axis.

The projection value = score.

---

## 4️⃣ Hyperplanes

### Definition: Hyperplane

A hyperplane is:

* A line in 2D
* A plane in 3D
* A (p−1)-dimensional surface in p dimensions

In supervised learning:

PLS-DA tries to find a hyperplane separating classes.

---

## 5️⃣ Separation as Geometry

If two groups are separable:

* There exists a direction in space
* Along which projections differ

If no such direction exists:

* Groups overlap in multidimensional space

---

## 6️⃣ Orthogonality

Principal components are perpendicular (orthogonal).

This ensures:

* No redundant variance
* Independent directions

---

## 7️⃣ Distance and Similarity

Common distance measure:

[
d(x_i, x_j) = \sqrt{\sum_{k=1}^{p}(x_{ik} - x_{jk})^2}
]

This is Euclidean distance.

Small distance → similar samples.

---

## 8️⃣ Overfitting Geometrically

Overfitting means:

* Drawing a very complex boundary
* That perfectly separates training data
* But fails on new data

Geometrically:

The hyperplane becomes too sensitive to noise.

---

---

# 🚨 Block 3 — Common Misconceptions in Multivariate Analysis

> This section is critical for teaching.

---

## ❌ Misconception 1: “If PCA separates groups, it proves biology.”

Reality:

PCA shows variance structure, not causality.

Separation may be due to:

* Batch effect
* Instrument drift
* Scaling artifact

---

## ❌ Misconception 2: “PLS-DA separation means strong predictive power.”

Reality:

PLS-DA always tries to separate groups.

Without cross-validation:

* It is meaningless.

---

## ❌ Misconception 3: “VIP > 1 means biomarker.”

Reality:

VIP indicates contribution to model.

It does NOT guarantee:

* Statistical significance
* Biological relevance
* Reproducibility

---

## ❌ Misconception 4: “Scaling is just cosmetic.”

Reality:

Scaling changes the geometry of the data space.

Different scaling → different PCA directions.

---

## ❌ Misconception 5: “Removing outliers improves model quality.”

Reality:

Outliers may contain real biology.

Removing them artificially improves separation.

---

## ❌ Misconception 6: “Higher R² always means better model.”

Reality:

R² measures fit.

Q² measures prediction.

A high R² with low Q² = overfitting.

---

## ❌ Misconception 7: “Multivariate analysis finds truth.”

Reality:

It finds patterns.

Interpretation requires:

* Biological reasoning
* Experimental validation
* Replication

---

## ❌ Misconception 8: “More components = better model.”

Reality:

Each additional component risks fitting noise.

Model complexity must be justified.

---

# 🎓 Final Teaching Message

Multivariate analysis is:

* Linear algebra
* Geometry
* Statistics
* Modeling philosophy

Students must understand:

> Every preprocessing choice reshapes the mathematical space.
> Every model is a projection of reality.
> Interpretation is a scientific responsibility.


