# Concrete Compressive Strength Prediction Using Machine Learning

**B.Tech CSE Semester V — Machine Learning Case Study Project**

---

## Problem Statement

A construction materials testing laboratory routinely measures the compressive strength of concrete mixes using destructive physical crush tests (casting standard 150 mm cubes or 150 × 300 mm cylinders, curing in water for up to 28 or 90 days, and testing to failure in a hydraulic compression machine). This traditional laboratory testing workflow is labor-intensive, time-consuming, expensive in materials, and slow for iterative mix pre-qualification.

The objective of this case study is to develop, evaluate, and deploy a supervised machine learning regression pipeline capable of predicting concrete compressive strength directly from batch mix proportions and curing duration. The model can potentially support preliminary concrete mix screening and reduce unnecessary experimental trials by providing rapid strength estimates. However, physical laboratory testing remains necessary for engineering validation, quality assurance, and compliance.

---

## Dataset

This study utilizes the authentic **Concrete Compressive Strength dataset from the UCI Machine Learning Repository**, originally contributed by Prof. I-Cheng Yeh (1998).

* **Dataset Filename:** `Concrete_Compressive_Strength.csv`
* **Dataset Size:** 1,030 experimental formulations (1,005 distinct experimental records after removing duplicate laboratory replicates during preprocessing to prevent identical observations from being distributed across training and testing partitions).
* **Missing Values:** 0 (clean dataset).
* **Target Variable:** `Concrete Compressive Strength` (Continuous, measured in MegaPascals — MPa).

---

## Features

The dataset comprises 8 input features and 1 continuous target:

| Feature Name | Role in Concrete Mix | Units | UCI Range |
| :--- | :--- | :--- | :--- |
| **Cement** | Primary hydraulic binder; hydrates to form C-S-H gel | kg/m³ | 102.0 – 540.0 |
| **Blast Furnace Slag** | Supplementary cementitious material; latent hydraulic binder | kg/m³ | 0.0 – 359.4 |
| **Fly Ash** | Supplementary cementitious material (pozzolan); secondary binder | kg/m³ | 0.0 – 200.1 |
| **Water** | Essential reactant for hydration; excess increases capillary porosity | kg/m³ | 121.75 – 247.0 |
| **Superplasticizer** | Chemical admixture (high-range water reducer) | kg/m³ | 0.0 – 32.2 |
| **Coarse Aggregate** | Crushed stone/gravel (>4.75 mm); primary structural skeleton | kg/m³ | 801.0 – 1145.0 |
| **Fine Aggregate** | Natural sand (<4.75 mm); fills voids between coarse aggregate | kg/m³ | 594.0 – 992.6 |
| **Age** | Curing time elapsed prior to compressive testing | days | 1 – 365 |
| **Concrete Compressive Strength** *(Target)* | Ultimate uniaxial compressive resistance | MPa | 2.33 – 82.60 |

---

## Algorithms

Exactly five regression algorithms were implemented, trained, and comparatively evaluated:

1. **Linear Regression**
2. **Polynomial Regression** (Degree 2 feature expansion)
3. **Decision Tree Regressor**
4. **Random Forest Regressor** (Ensemble Bagging)
5. **Gradient Boosting Regressor** (Sequential Boosting)

All models incorporate feature standardization using `StandardScaler` inside Scikit-Learn `Pipeline` architectures to guarantee zero data leakage between training and validation/test folds.

---

## Evaluation Metrics & Comparative Study

All models were evaluated on the held-out test partition (20% split, $N = 201$) and through 5-Fold Cross-Validation on the training partition ($N = 804$):

* **$R^2$ Score (Coefficient of Determination)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **Mean Absolute Error (MAE)**
* **5-Fold Cross-Validation** (Mean $\pm$ Standard Deviation)

### Measured Results Comparison Table

| Model | Test R² | Test MSE | Test RMSE | Test MAE | CV R² Mean | CV RMSE Mean | CV MAE Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.5801 | 125.2653 | 11.1922 | 8.8960 | 0.5895 (±0.0287) | 10.2027 (±0.1535) | 8.1003 (±0.0681) |
| **Polynomial Regression (deg 2)** | 0.7686 | 69.0364 | 8.3088 | 6.2205 | 0.7730 (±0.0088) | 7.5984 (±0.2858) | 5.8601 (±0.2810) |
| **Decision Tree Regressor** | 0.8671 | 39.6385 | 6.2959 | 3.9451 | 0.7907 (±0.0215) | 7.2945 (±0.5489) | 4.9944 (±0.3996) |
| **Random Forest Regressor (Selected Best)** | **0.9108** | **26.6161** | **5.1591** | **3.4168** | **0.8863 (±0.0226)** | **5.3483 (±0.4885)** | **3.8707 (±0.2948)** |
| **Gradient Boosting Regressor** | 0.8979 | 30.4535 | 5.5185 | 4.0958 | 0.8906 (±0.0182) | 5.2556 (±0.4279) | 3.8981 (±0.2813) |

---

## Expected Prediction Error Methodology

To avoid misrepresenting error statistics:

1. **Mean Absolute Error (MAE):** The MAE represents the average absolute difference between predicted and observed compressive strength values ($3.87\text{ MPa}$ in out-of-fold training cross-validation; $3.42\text{ MPa}$ on the unseen test set). It does **not** represent a guaranteed confidence interval or boundary.
2. **95th Percentile Absolute Error:** The 95th percentile absolute error indicates the error magnitude below which approximately 95% of the observed absolute out-of-fold prediction errors fall in the training cross-validation analysis ($10.56\text{ MPa}$ for out-of-fold training predictions; $9.54\text{ MPa}$ for test-set residuals).
3. **Out-of-Fold Estimation:** The empirical error distribution is estimated primarily using out-of-fold cross-validation predictions on the training data ($N = 804$) to avoid using the held-out test partition for uncertainty estimation.

> **Methodological Disclaimer:**  
> This empirical error statistic describes the observed model error distribution and should not be interpreted as a guaranteed prediction interval or engineering confidence interval.

---

## Key Analytical Insights

1. **Polynomial Regression vs Linear Regression:** The improved performance of polynomial regression ($R^2$ increases from 0.5801 to 0.7686; RMSE decreases from 11.19 MPa to 8.31 MPa) indicates that nonlinear relationships and interaction cross-products among the input variables are useful for predicting compressive strength in this dataset.
2. **Curing Age Nonlinearity:** The exploratory analysis indicates a nonlinear relationship between curing age and compressive strength, with larger strength gains at earlier ages (averaging ~1.0 MPa/day up to 28 days) and slower increases at later ages (~0.02 MPa/day from 28 to 365 days).
3. **Feature Influence:** According to the Random Forest feature-importance measure, curing age (36.1%) and cement content (30.3%) were among the most influential predictors used by the model. Feature importance indicates how strongly the trained model relies on each feature for prediction; it does not establish direct physical causation.
4. **Water-Cement Ratio:** Shows an empirical negative association ($r = -0.49$) with compressive strength.

---

## Project Structure

```text
ML_Final_Project/
│
├── concrete_strength_prediction.ipynb   # Complete 33-section executed Jupyter Notebook
├── Concrete_Compressive_Strength.csv   # Primary UCI dataset file
├── app.py                              # Interactive Streamlit prediction application
├── best_model_pipeline.joblib          # Trained & serialized Scikit-learn Pipeline
├── model_metadata.json                 # Companion model metrics & error statistics
├── requirements.txt                    # Exact pinned dependencies for reproducibility
├── README.md                           # Comprehensive project documentation
└── .gitignore                          # Git ignore specification
```

---

## How to Install

1. Clone or navigate into the project directory:
   ```bash
   cd /Users/vruttipatil/Desktop/ML_Final_Project
   ```

2. Activate the Python virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install the pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run Notebook

Launch JupyterLab or Jupyter Notebook:

```bash
jupyter notebook concrete_strength_prediction.ipynb
```

The notebook is fully pre-executed from cell 1 to 72 and contains all rendered charts, statistical outputs, regression diagnostics, and answers to all required research questions.

---

## How to Run Streamlit

To launch the interactive concrete strength prediction web interface:

```bash
streamlit run app.py
```

---

## Limitations

1. **Domain Boundary Constraints:** The model is valid strictly within the envelope of the UCI training dataset ($102\text{--}540\text{ kg/m}^3$ cement, $w/c$ ratios $0.27\text{--}1.88$, curing age $1\text{--}365\text{ days}$). Predictions outside these bounds introduce extrapolation uncertainty.
2. **Omission of Environmental Variables:** The dataset reflects standard laboratory curing conditions ($20^\circ\text{C}$, $>95\%$ relative humidity). Ambient temperature fluctuations, wind drying, and site curing variations alter actual field hydration.
3. **Material Quality Variations:** Cement mineralogy (e.g. $C_3A$ content) and aggregate morphology (angularity, texture, grading) are not captured as independent features.
4. **Unseen Chemical Admixtures:** Special additives such as silica fume, metakaolin, accelerators, retarders, or air-entraining agents are absent from the dataset.

---

## Disclaimer

This machine learning model is developed strictly for **academic demonstration, research analysis, and preliminary virtual mix screening**. It must **NOT replace mandatory physical destructive compression testing** of standard cube/cylinder specimens required by civil engineering regulatory codes (such as **IS 456**, **ACI 318**, or **Eurocode 2**) for structural compliance and life safety certification.
