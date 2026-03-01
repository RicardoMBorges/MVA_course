# 1️⃣ NORMALIZATION

*(Sample-wise adjustment)*

## 🔎 What it corrects

Normalization corrects for:

* Different sample concentrations
* Injection volume differences
* Total signal intensity variation
* Dilution effects

It works **row-wise** (per sample).

---

## 🧪 Ideal “Before”

Imagine metabolomics samples:

| Sample | Total Intensity |
| ------ | --------------- |
| A      | 1,000,000       |
| B      | 2,000,000       |
| C      | 500,000         |

But biologically they are identical.

The only difference is concentration.

On a PCA score plot (before normalization):

* Samples separate along PC1
* But separation is driven by total intensity
* Not by biology

That’s artificial clustering.

---

## 🧪 Ideal “After” Normalization

After Total Sum Normalization:

Each sample is scaled so:

$$
\sum_{j=1}^{p} x_{ij} = 1
$$

Now:

* Concentration effect disappears
* Only relative feature differences remain
* Samples cluster based on biology

---

### ✅ Ideal Outcome of Normalization

✔ Total intensity comparable
✔ No separation driven by dilution
✔ Biological patterns preserved

---

# 🎯 2️⃣ SCALING

*(Variable-wise adjustment)*

Scaling works **column-wise** (per variable).

It corrects for:

* Different magnitude ranges
* Dominance of large peaks
* Variance imbalance

---

## 🧪 Ideal “Before”

Imagine:

| Variable         | Range     |
| ---------------- | --------- |
| Glucose          | 1000–5000 |
| Minor metabolite | 0.01–0.05 |

Without scaling:

* PCA is dominated by glucose
* Small metabolites contribute almost nothing
* Subtle biology is masked

---

## 🧪 Ideal “After” Autoscaling

Autoscaling:

$$
x''*{ij} = \frac{x*{ij} - \bar{x}_j}{s_j}
$$

Now:

* Each variable has mean = 0
* Each variable has variance = 1

Result:

* All variables contribute equally
* Subtle differences become visible

---

### ✅ Ideal Outcome of Scaling

✔ Variables equally weighted
✔ Large peaks no longer dominate
✔ Smaller features become interpretable

---

# 🧠 Key Conceptual Difference

|          | Normalization             | Scaling                        |
| -------- | ------------------------- | ------------------------------ |
| Acts on  | Samples (rows)            | Variables (columns)            |
| Fixes    | Concentration differences | Variance magnitude differences |
| Prevents | Artificial clustering     | Variable dominance             |
| Changes  | Relative intensity        | Variance structure             |

---

# 🎓 In a Close-to-Ideal World

### Before any preprocessing:

Score plot separation should reflect:

* Biology
* Chemistry
* Experimental design

Not:

* Injection volume
* Instrument drift
* Variable magnitude

---

## Ideal workflow

1. Remove technical bias → Normalization
2. Equalize variable influence → Scaling
3. Then PCA

---

# 🚨 Important Point

Normalization and scaling solve **orthogonal problems**.

Normalization fixes sample-to-sample comparability.
Scaling fixes variable-to-variable comparability.

They are not interchangeable.

---

# 📐 Geometric View

Normalization:

→ Rescales the length of each sample vector.

Scaling:

→ Rescales the axes of the coordinate system.

That distinction is extremely powerful when students see it.


