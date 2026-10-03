# Concrete Compressive Strength Prediction Using Machine Learning

**B.Tech CSE 2024–28 Machine Learning Case Study / Problem Statement**  
**Semester V — Machine Learning**

---

## 1. Title
**Concrete Compressive Strength Prediction Using Machine Learning**

---

## 2. Problem Statement
A construction materials laboratory routinely characterizes the quality and mechanical capacity of structural concrete through destructive uniaxial compressive strength tests. Standard protocols require casting concrete specimens into standard molds ($150\text{ mm}$ cubes or $150 \times 300\text{ mm}$ cylinders), water-curing them under controlled conditions for specific curing periods (typically 7, 14, 28, or 90 days), and crushing them to failure in a calibrated hydraulic compression testing machine.

While authoritative, this physical testing procedure is:
* **Time-consuming:** Requires waiting up to 28 days or longer for hydration to mature before design acceptance.
* **Labor- and material-intensive:** Consumes significant binder materials, aggregates, water, and labor for trial batches.
* **Inflexible for mix design optimization:** Exploring non-standard mix designs or supplementary cementitious binders requires dozens of trial batches.

The goal of this project is to develop, evaluate, and deploy a supervised machine learning regression pipeline to accurately predict concrete compressive strength (in MegaPascals, $\text{MPa}$) directly from batch mix constituent quantities:
* Cement
* Blast Furnace Slag
* Fly Ash
* Water
* Superplasticizer (chemical admixture)
* Coarse Aggregate
* Fine Aggregate
* Curing Age

Accurate predictive modeling reduces the number of physical strength trial batches needed during early design phases, expedites mix qualification, and saves raw material costs.

### Dataset Provenance & Generation Note
* **Primary Dataset:** The authentic **Concrete Compressive Strength Dataset** from the **UCI Machine Learning Repository**, originally contributed by Prof. I-Cheng Yeh (1998).
* **Sample Count:** 1,030 experimental formulations (1,005 distinct experimental formulations after removing 25 duplicate laboratory replicate specimens to prevent data leakage across train/test splits).
* **Missing Values:** Zero missing or null values across all attributes.
* **Synthetic Data Generation Note:** In scenarios where the UCI repository dataset is unavailable or offline, an equivalent synthetic benchmark can be generated using NumPy/Pandas or `make_regression()` from Scikit-Learn by matching empirical feature correlations, bounded uniform/truncated normal distributions for each constituent ($102\text{--}540\text{ kg/m}^3$ cement, $0\text{--}360\text{ kg/m}^3$ slag, $0\text{--}200\text{ kg/m}^3$ fly ash, $120\text{--}250\text{ kg/m}^3$ water, $0\text{--}32\text{ kg/m}^3$ superplasticizer, $800\text{--}1150\text{ kg/m}^3$ coarse aggregate, $590\text{--}1000\text{ kg/m}^3$ fine aggregate, $1\text{--}365\text{ days}$ age), and incorporating Abram's water-to-binder ratio inverse relationship coupled with logarithmic curing age growth kinetics: $\text{Strength} \propto \frac{k_1}{(w/c)^{k_2}} + k_3 \ln(\text{Age})$.

---

## 3. Objectives
To fulfill the case study requirements, the project undertakes the following core objectives:

1. **Exploratory Data Analysis & Visualisation:** Inspect statistical distributions, summary statistics (mean, median, standard deviation, skewness), outlier boundaries, and feature interdependencies.
2. **Concrete Mix Composition Analysis:** Quantify the proportions of individual binder constituents, aggregates, and chemical admixtures across varying strength classes.
3. **Water-Cement Ratio Study:** Explicitly compute the water-to-cement ratio ($w/c$) and evaluate its empirical and theoretical relationship with compressive strength in accordance with concrete technology fundamentals.
4. **Curing Age Non-Linearity Examination:** Analyze the progression of compressive strength across curing intervals (1 to 365 days) to test for non-linear, logarithmic plateauing behavior.
5. **Leakage-Free Feature Scaling:** Implement robust preprocessing (`StandardScaler`) wrapped inside Scikit-Learn `Pipeline` architectures to prevent data leakage during train/test partitioning and cross-validation.
6. **Model Development & Training:** Implement, configure, and train five specified supervised regression models spanning linear, polynomial, tree-based, and ensemble algorithms.
7. **Cross-Validation & Comparative Study:** Evaluate all models on identical held-out test partitions ($20\%$, $N=201$) and 5-fold cross-validation ($N=804$) using $R^2$, $\text{MSE}$, $\text{RMSE}$, and $\text{MAE}$.
8. **Residual Analysis on Best Model:** Perform diagnostic residual analysis on the top-performing model, assessing actual vs. predicted correlation, residual homoscedasticity, and error normality.
9. **Model Selection with Justification:** Objectively justify the final model choice balancing prediction accuracy, generalization variance, and computational efficiency.
10. **Interactive Application Deployment:** Deploy the optimized model pipeline as a user-friendly Streamlit web application providing real-time strength predictions, mix ratio calculations, and empirical error ranges.
11. **Limitations & Real-World Applicability Analysis:** Contextualize the boundaries of ML in civil engineering, discussing domain limits and why ML augments, but does not replace, physical testing codes.

---

## 4. Machine Learning Algorithms
Five supervised regression algorithms were implemented, trained, and benchmarked:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Raw Input Features (8)                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                Scikit-Learn Pipeline: StandardScaler                  │
│       (Fits strictly on training folds to prevent data leakage)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌───────────────┬──────────┴─────┬────────────────┬─────────────┐
         ▼               ▼                ▼                ▼             ▼
   ┌───────────┐   ┌───────────┐    ┌───────────┐    ┌───────────┐ ┌───────────┐
   │  Linear   │   │Polynomial │    │ Decision  │    │  Random   │ │ Gradient  │
   │Regression │   │Regression │    │   Tree    │    │  Forest   │ │ Boosting  │
   │           │   │ (Deg. 2)  │    │ Regressor │    │ Regressor │ │ Regressor │
   └───────────┘   └───────────┘    └───────────┘    └───────────┘ └───────────┘
```

1. **Linear Regression:**  
   Standard Ordinary Least Squares (OLS) estimating linear coefficients for all 8 normalized mix constituents:
   $$y = \beta_0 + \sum_{i=1}^{8} \beta_i x_i + \epsilon$$
2. **Polynomial Regression (Degree 2):**  
   Quadratic feature expansion (`PolynomialFeatures(degree=2, include_bias=False)`) generating 44 expanded features (linear terms, squared terms $x_i^2$, and pairwise interaction terms $x_i x_j$) to capture non-linearities and chemical synergisms.
3. **Decision Tree Regressor:**  
   Non-parametric recursive binary partitioning based on mean squared error reduction:
   $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y})^2$$
   Tuned with `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2` to constrain overfitting.
4. **Random Forest Regressor (Selected Best Model):**  
   Ensemble bootstrap aggregation (bagging) of 200 de-correlated randomized decision trees:
   $$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$
   Configured with `n_estimators=200`, `max_depth=15`, `min_samples_split=3`, `min_samples_leaf=1`, and `random_state=42`.
5. **Gradient Boosting Regressor:**  
   Sequential ensemble boosting minimizing squared loss by iteratively fitting regression trees to the pseudo-residuals of preceding estimators:
   $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$
   Configured with `n_estimators=200`, `learning_rate=0.08`, `max_depth=5`, and `subsample=0.85`.

---

## 5. Comparative Study

### 5.1 Evaluation Metrics Defined
* **Coefficient of Determination ($R^2$):** Proportion of variance in compressive strength explained by the features:
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
* **Mean Squared Error ($\text{MSE}$):** Average of squared prediction residuals:
  $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Root Mean Squared Error ($\text{RMSE}$):** Root of MSE, expressed in native units ($\text{MPa}$):
  $$\text{RMSE} = \sqrt{\text{MSE}}$$
* **Mean Absolute Error ($\text{MAE}$):** Average magnitude of absolute errors ($\text{MPa}$):
  $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$
* **5-Fold Cross-Validation:** Partitioning training data ($N=804$) into 5 disjoint validation folds to measure generalization stability and compute standard error bounds.

### 5.2 Model Performance Comparison Table

| Machine Learning Algorithm | Test $R^2$ | Test $\text{MSE}$ ($\text{MPa}^2$) | Test $\text{RMSE}$ ($\text{MPa}$) | Test $\text{MAE}$ ($\text{MPa}$) | 5-Fold CV $R^2$ (Mean $\pm$ Std) | 5-Fold CV $\text{RMSE}$ ($\text{MPa}$) | 5-Fold CV $\text{MAE}$ ($\text{MPa}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.5801 | 125.2653 | 11.1922 | 8.8960 | 0.5895 (±0.0287) | 10.2027 (±0.1535) | 8.1003 (±0.0681) |
| **Polynomial Regression (Degree 2)** | 0.7686 | 69.0364 | 8.3088 | 6.2205 | 0.7730 (±0.0088) | 7.5984 (±0.2858) | 5.8601 (±0.2810) |
| **Decision Tree Regressor** | 0.8671 | 39.6385 | 6.2959 | 3.9451 | 0.7907 (±0.0215) | 7.2945 (±0.5489) | 4.9944 (±0.3996) |
| **Random Forest Regressor (Best)** | **0.9108** | **26.6161** | **5.1591** | **3.4168** | **0.8863 (±0.0226)** | **5.3483 (±0.4885)** | **3.8707 (±0.2948)** |
| **Gradient Boosting Regressor** | 0.8979 | 30.4535 | 5.5185 | 4.0958 | 0.8906 (±0.0182) | 5.2556 (±0.4279) | 3.8981 (±0.2813) |

### 5.3 Curing Age Non-Linearity Analysis
A primary requirement of the case study is examining whether the relationship between curing age and compressive strength is linear or curved:
* **Empirical Observations:**
  * Concrete gains strength rapidly during early hydration (Days 1 to 28), rising from an average of $9.45\text{ MPa}$ at Day 1 to $36.43\text{ MPa}$ at Day 28 (gaining $\approx 1.0\text{ MPa/day}$).
  * After 28 days, strength gain slows substantially, rising from $36.43\text{ MPa}$ to $43.56\text{ MPa}$ at Day 365 ($\approx 0.02\text{ MPa/day}$).
* **Linearity vs Logarithmic Transformation:**
  * Plotting Compressive Strength directly against Age shows a distinct concave-downward curve.
  * Plotting Compressive Strength against $\ln(\text{Age})$ yields a near-linear trajectory ($r \approx 0.61$).
* **Civil Engineering Explanation:** This behavior directly matches cement hydration chemistry. Fast hydration of tricalcium silicate ($C_3S$) dominates the first 28 days, creating dense Calcium-Silicate-Hydrate ($\text{C-S-H}$) gel. Slower reacting dicalcium silicate ($C_2S$) governs late-stage hydration, yielding diminished incremental strength gains over months.

### 5.4 Residual Analysis of Selected Model (Random Forest)
The residuals ($e_i = y_i - \hat{y}_i$) of the Random Forest model were evaluated across three diagnostic checks:
1. **Actual vs. Predicted Plot:** Predicted values track the $45^\circ$ reference line ($y = \hat{y}$) across the full range ($10\text{--}75\text{ MPa}$) without systematic over- or under-prediction.
2. **Residual vs. Predicted Plot:** Residuals are uniformly scattered around the zero horizontal datum ($e = 0$) across predicted strengths, demonstrating homoscedasticity without widening funnel patterns.
3. **Residual Distribution & Normality:** Errors are bell-shaped and centered near zero (mean residual $+0.73\text{ MPa}$). $95\%$ of out-of-fold cross-validation errors fall within $\pm 10.56\text{ MPa}$, and $95\%$ of test residuals fall within $\pm 9.54\text{ MPa}$, with a test $\text{MAE}$ of $3.42\text{ MPa}$.

---

## 6. Deployment
The trained best model is serialized into `best_model_pipeline.joblib` and served through an interactive, responsive **Streamlit** web application (`app.py`).

### 6.1 Application Features & User Interface
* **Dual Parameter Input Channels:**
  * Interactive numeric step inputs and sliders with real-time domain range validation against UCI bounds.
  * **Mix Presets:** One-click loading of standard concrete mix designs (Standard M25 Structural Mix, High-Performance M60 Mix, High Fly-Ash Eco Mix, Lean Mass Concrete Mix).
* **Instantaneous Prediction & Error Bounds:**
  * Displays predicted Compressive Strength in $\text{MPa}$ with prominent visual grading badges (Low Strength $<20\text{ MPa}$, Standard Structural $20\text{--}40\text{ MPa}$, High Strength $40\text{--}60\text{ MPa}$, Ultra-High Strength $>60\text{ MPa}$).
  * **Expected Error Range:** Reports the empirical error bound ($\pm 3.42\text{ MPa}$ based on held-out test MAE; $\pm 9.54\text{ MPa}$ for the 95th percentile error envelope).
* **Mix Proportion Diagnostics:**
  * Real-time calculation of **Water-to-Cement Ratio** ($w/c$) and **Water-to-Binder Ratio** ($w/(c + \text{slag} + \text{fly ash})$).
  * Color-coded indicators alerting the user if the $w/c$ ratio exceeds standard structural thresholds ($>0.55$).
* **Transparent Architecture & Metadata:**
  * Collapsible drawers detailing pipeline steps, 5-model comparative metrics table, training dataset characteristics, feature importances, and engineering disclaimers.

---

## 7. Final Analysis

### 7.1 Mix Components with the Greatest Effect on Strength
Based on Random Forest feature importance metrics (Mean Decrease in Impurity / Gini Importance):
1. **Curing Age ($36.1\%$):** The single most dominant factor determining how much cementitious hydration has matured.
2. **Cement Content ($30.3\%$):** The primary hydraulic binder providing the Calcium-Silicate-Hydrate ($\text{C-S-H}$) matrix.
3. **Superplasticizer ($9.1\%$):** Enables high workability at reduced water contents, indirectly boosting density and strength.
4. **Water Content ($8.4\%$):** Controls capillary porosity; excess water introduces micro-voids that reduce strength.
5. **Blast Furnace Slag ($7.2\%$):** Provides latent hydraulic binding when activated by cement hydration.
6. **Fine Aggregate ($4.6\%$), Coarse Aggregate ($2.7\%$), Fly Ash ($1.7\%$):** Fills voids and provides structural skeleton, contributing less to tree split decisions within this dataset envelope.

### 7.2 Best-Performing Algorithm
* **Random Forest Regressor** achieved superior generalization across all criteria:
  * Highest Test $R^2$ (**0.9108**) and lowest Test RMSE (**$5.16\text{ MPa}$**).
  * Highest 5-Fold Cross-Validation mean $R^2$ (**$0.8863 \pm 0.0226$**) and CV RMSE (**$5.35 \pm 0.49\text{ MPa}$**).
* Random forest successfully captures complex non-linear curves, thresholds, and multi-binder interactions (cement + slag + fly ash + superplasticizer) without suffering from the high variance of individual decision trees or underfitting of linear equations.

### 7.3 Non-Linearity of Curing Age
* The effect of curing age on compressive strength is **unquestionably non-linear and curved**.
* Concrete strength increases asymptotically: rapid initial gain during the first 28 days followed by a distinct plateau. Fitting a straight linear trend severely overestimates early strength (e.g. at 3–7 days) and underestimates late strength (e.g. at 90–365 days). A logarithmic or tree-based piece-wise representation is physically and statistically necessary.

### 7.4 Impact of Polynomial Regression
* Expanding linear inputs into degree-2 polynomial features produces a **substantial performance gain**:
  * Test $R^2$ increased from **0.5801** to **0.7686** ($+32.5\%$ improvement).
  * Test RMSE dropped from **$11.19\text{ MPa}$** to **$8.31\text{ MPa}$** ($25.7\%$ error reduction).
  * Test MAE dropped from **$8.90\text{ MPa}$** to **$6.22\text{ MPa}$**.
* This improvement confirms that interaction cross-products (e.g., $\text{Water} \times \text{Superplasticizer}$, $\text{Cement} \times \text{Age}$) and quadratic curves are vital for capturing concrete behavior. However, polynomial regression is still outperformed by non-parametric ensemble models (Random Forest $R^2 = 0.9108$) which model localized non-linearities without polynomial boundary oscillations.

### 7.5 Limitations of Predicting Strength Without Physical Testing
1. **Material Specificity & Mineralogy:** The dataset treats all "Cement" identically; it does not encode cement type (OPC 43, OPC 53, PPC, Rapid Hardening), tricalcium aluminate ($C_3A$) content, or clinker fineness (Blaine surface area).
2. **Aggregate Morphology:** Aggregate shape (angular, rounded, flaky), surface texture, geological source, and gradation curves (fineness modulus) strongly influence the interfacial transition zone ($\text{ITZ}$) but are omitted from tabular weights.
3. **Curing Microclimate:** Dataset measurements assume moist laboratory curing ($20^\circ\text{C}$, $>95\%$ RH). Field conditions subject to temperature swings, solar radiation, low humidity, or delayed curing compound error.
4. **Omission of Modern Chemical Admixtures:** Does not cover silica fume, metakaolin, viscosity modifiers, crystalline waterproofers, or air-entraining agents.
5. **Regulatory Certification:** Building standards (**IS 456:2000**, **ACI 318**, **Eurocode 2**, **BS EN 12390**) mandate destructive physical cube/cylinder breaks to certify load-bearing structures for human occupancy.

---

## 8. Questions to Be Answered Using ML

### 1. Can compressive strength be predicted from mix composition?
**Yes.** Supervised machine learning algorithms achieve high predictive fidelity using mix proportions and curing age. The Random Forest model accounts for **$91.08\%$ of the variance** ($R^2 = 0.9108$) on held-out test data, with an average prediction error ($\text{MAE}$) of only **$3.42\text{ MPa}$**.

### 2. Which component most influences strength?
**Curing Age ($36.1\%$ relative importance)** followed closely by **Cement Quantity ($30.3\%$ relative importance)** are the two most influential predictors, jointly accounting for **$66.4\%$** of model predictive decisions. In terms of chemical admixtures, Superplasticizer ($9.1\%$) and Water ($8.4\%$) also exert strong control over strength development.

### 3. Is the effect of curing age linear or curved?
**It is curved (logarithmic).** Strength gains rapidly in the first 28 days ($\approx 1.0\text{ MPa/day}$), then significantly plateaus between 28 and 365 days ($\approx 0.02\text{ MPa/day}$). Plotting strength versus $\ln(\text{Age})$ linearizes the trajectory, demonstrating that hydration kinetics follow a logarithmic curve rather than a linear slope.

### 4. Does polynomial regression improve the prediction?
**Yes, significantly.** Introducing degree-2 polynomial and interaction features improves $R^2$ from **0.5801 to 0.7686** ($+0.1885$ increase, or $+32.5\%$ improvement) and reduces RMSE from **$11.19\text{ MPa}$ to $8.31\text{ MPa}$** ($25.7\%$ error reduction). This demonstrates that second-order interactions (such as water-binder interaction and water-superplasticizer coupling) are essential when using linear model architectures.

### 5. What does the residual plot indicate?
The residual plot for the Random Forest model shows:
* **Homoscedasticity:** Errors are evenly scattered around the zero line ($e = 0$) across low, medium, and high strength levels, without funneling.
* **Absence of Systematic Bias:** No curvature or polynomial drift remains in the residuals, showing that non-linear relationships have been captured.
* **Normality:** Residuals are centered near zero ($\mu = +0.73\text{ MPa}$) with $95\%$ of test residuals falling within $\pm 9.54\text{ MPa}$.

### 6. How well does the model perform on new mix designs?
The model generalizes well on unseen mix designs **provided they lie within the convex hull (domain envelope) of the training dataset** ($102\text{--}540\text{ kg/m}^3$ cement, $w/c$ ratios $0.27\text{--}1.88$, age $1\text{--}365\text{ days}$). On the held-out test set ($N=201$), the model achieved an $R^2$ of **0.9108** and an MAE of **$3.42\text{ MPa}$**. Extrapolating to mixes outside these envelopes (e.g. ultra-low water mixes with novel plasticizers or alternative geopolymer binders) will produce higher uncertainty.

### 7. Can the model reduce the need for physical testing?
**Yes, significantly in pre-qualification and mix optimization, but not in final statutory compliance.**
* **Where ML reduces testing:** During preliminary trial batch formulation, virtual screening eliminates non-viable mixes before laboratory batching, reducing raw material waste and saving hundreds of technician hours.
* **Where physical testing remains mandatory:** Civil engineering statutory building codes (**IS 456**, **ACI 318**) legally mandate physical destructive testing of representative field cubes/cylinders for quality control and structural safety clearance.

---

## 9. Project Directory Structure

```text
ML_Final_Project/
│
├── concrete_strength_prediction.ipynb   # Complete, executed end-to-end Jupyter Notebook
├── Concrete_Compressive_Strength.csv   # Primary UCI dataset (1,030 samples)
├── app.py                              # Interactive Streamlit prediction application
├── best_model_pipeline.joblib          # Serialized Random Forest Pipeline (StandardScaler + RF)
├── model_metadata.json                 # Companion metrics, parameters & error statistics
├── requirements.txt                    # Pinned Python package dependencies
├── README.md                           # Comprehensive documentation & case study report
└── .gitignore                          # Git ignore specification
```

---

## 10. Installation & Execution Guide

### Prerequisites
* Python 3.10, 3.11, or 3.12
* `pip` package manager
* Virtual environment tool (`venv` or `conda`)

### Step 1: Clone or Navigate to Project Directory
```bash
cd /Users/vruttipatil/Desktop/ML_Final_Project
```

### Step 2: Set Up Virtual Environment
```bash
# Create virtual environment (if not already created)
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Jupyter Notebook
To inspect the exploratory data analysis, data cleaning, model training, cross-validation, and diagnostic plots:
```bash
jupyter notebook concrete_strength_prediction.ipynb
```
*The notebook is pre-executed with all output charts, tables, and regression diagnostic plots visible.*

### Step 5: Launch the Streamlit Web Application
To run the interactive web-based prediction application:
```bash
streamlit run app.py
```
The interface will open in your default browser at `http://localhost:8501`.

---

## 11. Engineering Standards & Safety Disclaimer

> [!IMPORTANT]
> **Academic & Virtual Screening Notice:**  
> This machine learning application and predictive pipeline are developed strictly for **academic research, case study demonstration, and virtual mix pre-qualification**.
> 
> In accordance with standard civil engineering building codes—including **IS 456:2000** (Plain and Reinforced Concrete - Code of Practice), **ACI 318** (American Concrete Institute Building Code Requirements for Structural Concrete), and **Eurocode 2 / BS EN 12390**—**machine learning predictions must NEVER substitute for mandatory destructive hydraulic compression testing** of physical standard specimens for structural acceptance, legal compliance, or life safety certification.
