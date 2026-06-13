# Laptop Price Predictor 💻📊

An end-to-end Machine Learning project that predicts laptop prices based on their hardware specifications, configurations, and brands. The project follows a complete data science workflow from data clearing and robust feature engineering to evaluating advanced regression pipelines.

## 🚀 Features & Techniques Used

### 1. Data Cleaning & Extraction
* **Unit Stripping:** Extracted numerical values from raw text columns such as converting `Ram` (e.g., "8GB" ➔ `8` int) and `Weight` (e.g., "1.37kg" ➔ `1.37` float) to proper numerical datatypes[cite: 1].
* **Anomalies Removal:** Handled unneeded fields, dropping arbitrary identification columns (`Unnamed: 0`)[cite: 1].
* **Target Transformation:** Addressed the right-skewed target distribution of `Price` by applying a natural log transformation (`np.log(df['Price'])`), allowing regression algorithms to converge and perform better[cite: 1].

### 2. Feature Engineering (The Core Enhancement)
* **Screen Specifications:**
    * Extracted the presence of **Touchscreen** (`TouchScreen`) capability[cite: 1].
    * Extracted the presence of high-end **IPS display panels** (`IPS`)[cite: 1].
    * Parsed the string configurations to isolate horizontal (`X_Resolution`) and vertical (`Y_Resolution`) pixel bounds, which were then used to calculate **PPI (Pixels Per Inch)**[cite: 1].
* **Processor (CPU) Parsing:** Processed full raw strings to map diverse microarchitectures into grouped macro-brands[cite: 1].
* **Storage (Memory) Structural Extraction:** Used text-matching logic to map complicated compound values into tracks representing exact capacities[cite: 1].
* **Operating System & GPU Optimization:** Categorized operating systems into standard ecosystems and categorized graphic processing units to major market spaces[cite: 1].

### 3. Machine Learning Pipelines
The project standardizes feature pipeline definitions using Scikit-Learn's workflow architectures[cite: 1]. This guarantees that data formatting steps seamlessly connect to downstream models without experiencing any data leakage[cite: 1].

---

## 📈 Model Performance & Evaluation

Multiple regression architectures were tested on the train-test splits. Since the model was fitted using log-transformed prices, accuracy metrics track performance symmetrically over exponential scale steps.

| Model Pipeline | R² Score | Mean Absolute Error (MAE) |
| :--- | :---: | :---: |
| **Linear Regression** | `0.8018` | `0.1935` |
| **Decision Tree Regressor** | `0.8392` | `0.1826` |

*Note: The Ensemble tree-based variants successfully capture non-linear configurations (e.g., balancing high RAM with premium GPU components) better than traditional standard linear baselines.*

---

## 🛠️ Project Structure & Execution Roadmap

1. **Exploratory Data Analysis (EDA):** Univariate distribution visualizer (`sns.distplot`)[cite: 1], brand frequency charts, and continuous multi-feature correlation matrices (`df.corr(numeric_only=True)`).
2. **Feature Engineering Module:** Regex maps and lambda filters to format raw data.
3. **Pipeline Fitting:** Structural data partitioning, categorical mapping, and model fitting.

## 💻 Tech Stack & Libraries
* **Core:** Python 3
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** scikit-learn
* **Deployment:** Streamlit

