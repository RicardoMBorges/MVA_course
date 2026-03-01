# Multivariate Data Analysis Course – Streamlit App

An interactive web application for teaching and applying multivariate data analysis in metabolomics, analytical chemistry, and bioinformatics.

This app was designed for:

* Teaching multivariate statistics step-by-step
* Transparent model building
* Proper validation (Q², cross-validation)
* Fully interactive visualizations (hover + zoom)
* Downloadable figures as standalone HTML files

---

# 🔬 Implemented Methods

## Unsupervised

* PCA (raw and preprocessed)
* Correlation heatmaps

## Supervised

* Logistic Regression (baseline classifier)
* PLS-DA (PLS regression on one-hot encoded y)

## Validation

* Stratified Cross-Validation
* Accuracy / Balanced Accuracy
* Confusion Matrix
* ROC curve (binary)
* Q² (cross-validated predictive ability)

## Interpretation

* Model coefficients
* PLS loadings
* VIP (Variable Importance in Projection)
* Explained variance

---

# 📂 Accepted Data Format

The app follows a metabolomics-style format:

### Columns = Samples (Observations)

### Rows = Variables (Features)

Example:

```csv
,Sample1,Sample2,Sample3,Sample4
ATTRIBUTE_class,Control,Control,Treated,Treated
Feature_1,12.5,11.8,18.4,19.1
Feature_2,102,98,150,145
Feature_3,0.55,0.60,0.90,0.88
```

### Required Structure

* First column → row labels
* Columns → sample names
* One optional row → `ATTRIBUTE_class` (group labels)
* Remaining rows → numeric features

---

# 🚀 How to Use the App

The workflow is structured in 6 tabs:

---

## 1️⃣ Import

* Upload CSV or Excel file
* Map:

  * Row label column
  * Sample columns
  * Classification row
  * Feature rows

After mapping:

* Data is automatically transposed into sklearn-friendly format
* Quick diagnostics are shown
* Raw PCA (no scaling) is available inside an expander

Raw PCA is useful to:

* Detect strong magnitude bias
* Observe scaling effects
* Inspect obvious separation before preprocessing

---

## 2️⃣ Preprocessing

Options include:

* Missing value imputation
* Scaling (Z-score)
* Removing zero-variance variables
* Removing highly missing variables

Comparison plots show:

* Raw vs Processed distributions

This tab prepares X for modeling.

---

## 3️⃣ Exploration

Unsupervised structure analysis:

* PCA score plot
* Explained variance
* Correlation heatmap

This is the stage to ask:

> Is there natural structure without supervision?

---

## 4️⃣ Modeling

Choose between:

### Logistic Regression

* Baseline classifier
* Feature coefficient visualization

### PLS-DA

* Latent variable score plot
* Loadings plot
* VIP scores
* R²X (explained variance in X)
* Q² (cross-validated predictive ability)

PLS-DA is implemented using:

```python
PLSRegression
```

with one-hot encoded class labels.

---

## 5️⃣ Validation

Model assessment includes:

* Stratified Cross-Validation
* Repeated CV
* Confusion Matrix
* ROC curve (binary case)
* Accuracy & Balanced Accuracy

Q² is calculated as:

##
Q^2 = 1 - \frac{PRESS}{TSS}
##

Where:

* PRESS = prediction residual sum of squares (CV)
* TSS = total sum of squares of Y

This ensures predictive ability is assessed properly.

---

## 6️⃣ Interpretation

This section reinforces:

* Separation ≠ prediction
* Prediction ≠ causation
* Important variables require domain validation

Figures from all tabs can be downloaded as:

* Individual HTML
* ZIP of all figures

All visualizations are:

* Interactive
* Zoomable
* Hover-enabled
* Fully self-contained

---

# 📦 Installation (Local)

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

Create environment:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

---

# 🌐 Deploy on Streamlit Cloud

1. Push repository to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repository
4. Select `app.py`
5. Deploy

---

# 🎓 Teaching Philosophy

This app was built for structured learning:

1. Observe structure (PCA)
2. Apply supervision (PLS-DA)
3. Validate rigorously (Q², CV)
4. Interpret responsibly (VIP, loadings)

Separation without validation is meaningless.

---

# ⚠️ Important Notes

* PLS-DA can overfit easily.
* Always interpret Q² alongside R².
* Small sample sizes limit cross-validation folds automatically.

---

# 🧠 Recommended Extensions

Future upgrades could include:

* Permutation testing for PLS-DA
* CV-ANOVA
* SHAP (tree models)
* OPLS-DA
* Automated report generation

---

# 👨‍🔬 Author

[Ricardo M. Borges](https://orcid.org/0000-0002-7662-6734)
Instituto de Pesquisas de Produtos Naturais Walter Mors(IPPN-UFRJ), Rio de Janeiro, Brazil.



