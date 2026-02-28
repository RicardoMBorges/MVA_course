# 📊 Multivariate Data Analysis – Follow-Up Tutorial

## From Raw Data to Biological Interpretation

---

## 1️⃣ Why Multivariate Analysis?

In metabolomics (and in many biological datasets), we rarely deal with a single variable.

Instead, we measure:

* Dozens to thousands of metabolites
* Across multiple samples
* Belonging to different experimental groups

This means:

> Each sample is not a single value.
> Each sample is a **vector in multidimensional space**.

Multivariate analysis allows us to:

* Reduce dimensionality
* Detect patterns
* Identify group separation
* Discover discriminant variables
* Interpret biological meaning

---

## 2️⃣ Dataset Structure (MetaboAnalyst Format)

For this course, we follow the **MetaboAnalyst CSV format**.

### Structure rules:

* First row → `ATTRIBUTE_class` (group labels)
* First column → Feature names (metabolites)
* Remaining columns → Sample intensities

Example:

```csv
,Sample1,Sample2,Sample3,Sample4
ATTRIBUTE_class,Control,Control,Treated,Treated
Glucose,120,115,140,150
Lactate,80,75,95,100
Alanine,60,58,70,73
```

Interpretation:

* Rows = variables (metabolites)
* Columns = samples
* Class labels define biological grouping

---

## 3️⃣ Preprocessing: The Most Important Step

Before PCA or PLS-DA, data must be cleaned.

### 3.1 Missing Values

Common strategies:

* Median imputation
* Mean imputation
* Constant (0)
* Remove features above missing threshold

⚠️ Important:
Removing too many variables may remove biological meaning.

---

### 3.2 Scaling

Metabolites have different magnitudes.

Example:

* Glucose: 100–1000
* Cytokine: 0.01–2

Without scaling, large variables dominate PCA.

Common scaling:

| Method         | Effect                     |
| -------------- | -------------------------- |
| None           | Raw magnitude preserved    |
| Mean-centering | Centers data               |
| Autoscaling    | Mean-center + divide by SD |
| Pareto scaling | Divide by √SD              |

In metabolomics, **autoscaling** is often used.

---

## 4️⃣ Principal Component Analysis (PCA)

PCA is an **unsupervised method**.

It answers:

> Do samples naturally cluster?

### Conceptually:

* Original data = high dimensional space
* PCA finds new axes (principal components)
* These axes maximize variance

PC1 → largest variance
PC2 → second largest variance

---

### Interpretation of PCA Score Plot

* Each point = one sample
* Distance between points = similarity
* Separation suggests systematic differences

If groups separate without supervision:

👉 Strong biological signal.

---

### Loading Plot

Loadings tell us:

> Which variables drive separation?

High absolute loading values → strong influence

Interpretation must consider:

* Biological plausibility
* Analytical reliability
* Scaling method used

---

📌 *[Insert PCA score plot image here]*
📌 *[Insert PCA loading plot image here]*

---

## 5️⃣ Supervised Methods: PLS-DA

Unlike PCA:

PLS-DA uses group information.

It tries to:

> Maximize separation between predefined classes.

PLS-DA is powerful, but dangerous if misused.

---

### Why Dangerous?

Because it will **always try to separate groups**, even if separation is weak.

Therefore:

* Cross-validation is mandatory
* Permutation testing is recommended
* Avoid overfitting

---

### Model Evaluation Metrics

| Metric   | Meaning                    |
| -------- | -------------------------- |
| R²       | Explained variance         |
| Q²       | Predictive ability         |
| Accuracy | Classification performance |

If:

* R² high
* Q² low

👉 Model likely overfitted.

---

📌 *[Insert PLS-DA score plot here]*

---

## 6️⃣ VIP Scores (Variable Importance in Projection)

VIP identifies variables most responsible for class separation.

General rule:

* VIP > 1 → important variable

But:

VIP must be interpreted together with:

* Fold change
* p-value
* Biological context

Never rely on VIP alone.

---

## 7️⃣ Biological Interpretation

Statistics do not give biology automatically.

After identifying important features:

Ask:

* Are these metabolites biochemically related?
* Are they in the same pathway?
* Do they make physiological sense?

Multivariate analysis is:

> A hypothesis generator, not a final answer.

---

## 8️⃣ Common Mistakes in Multivariate Analysis

❌ Using raw data without scaling
❌ Not checking missing values
❌ Trusting PLS-DA without validation
❌ Ignoring batch effects
❌ Overinterpreting small separations

---

## 9️⃣ Recommended Workflow

1. Inspect raw data
2. Remove obvious artifacts
3. Handle missing values
4. Scale appropriately
5. Run PCA
6. Interpret structure
7. Run supervised method (if justified)
8. Validate model
9. Interpret biologically

---

## 🔟 Final Conceptual Takeaway

Multivariate analysis is about:

* Structure
* Variance
* Patterns
* Biological meaning

It is not about:

* Pretty plots
* Forced separation
* Statistical decoration

The goal is:

> Transform multidimensional complexity into interpretable biological insight.

---

# 🚀 What’s Next?

In the next module we will explore:

* Model validation strategies
* Cross-validation in practice
* Permutation testing
* Avoiding overfitting
* Real metabolomics case study

