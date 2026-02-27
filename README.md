# 🌸 Female Population Data Analysis with Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A comprehensive **Machine Learning project** that analyzes global female population development indicators using 4 ML classification models, rich EDA visualizations, and performance benchmarking. The project classifies countries/regions into **Low**, **Medium**, and **High** development levels based on key female-centric socioeconomic metrics.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [ML Models Used](#-ml-models-used)
- [Visualizations](#-visualizations)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This project investigates global disparities in female development using real-world socioeconomic indicators. The goal is to:

- Perform **Exploratory Data Analysis (EDA)** on female population statistics
- Build and compare **4 Machine Learning models** for development classification
- Visualize model performance using **confusion matrices, ROC curves, feature importance** and more
- Identify the **key drivers** of female development across regions

**Target Variable:** Development Level → `Low` | `Medium` | `High`

---

## 📊 Dataset

**Primary Kaggle Source:**
> 🔗 [World Bank Gender Statistics](https://www.kaggle.com/datasets/theworldbank/world-bank-intl-education)
> 🔗 [World Population by Gender – andrewmvd](https://www.kaggle.com/datasets/andrewmvd/world-population-by-gender)

**Kaggle Search Terms:**
- `gender statistics world bank`
- `world population female`
- `gender development index`

> ✅ The script also includes **built-in synthetic demo data** — no download required to run immediately.

### Features Used

| Feature | Description |
|---|---|
| `Female_Life_Expectancy` | Average female life expectancy (years) |
| `Female_Literacy_Rate` | % of adult females who are literate |
| `Female_Labor_Force_Pct` | % of females in the labor force |
| `Maternal_Mortality_Rate` | Deaths per 100,000 live births |
| `Female_School_Enrollment` | % of females enrolled in school |
| `Female_Internet_Usage_Pct` | % of females using the internet |
| `GDP_Per_Capita_USD` | GDP per capita in USD |
| `Urban_Population_Pct` | % living in urban areas |
| `Female_Political_Rep_Pct` | % of female political representatives |
| `Region` | Geographical region (encoded) |

---

## 🤖 ML Models Used

| # | Model | Library | Type |
|---|---|---|---|
| 1 | **Random Forest** | scikit-learn | Ensemble (Bagging) |
| 2 | **Logistic Regression** | scikit-learn | Linear Classifier |
| 3 | **XGBoost** | xgboost | Ensemble (Boosting) |
| 4 | **K-Nearest Neighbors** | scikit-learn | Instance-based |

All models are evaluated using:
- ✅ Test Accuracy
- ✅ 5-Fold & 10-Fold Cross Validation
- ✅ Precision, Recall, F1-Score
- ✅ ROC-AUC (One-vs-Rest)

---

## 📈 Visualizations

### Panel 1 — Exploratory Data Analysis
- Development Level class distribution (bar chart)
- Female life expectancy by region (horizontal bar)
- Feature correlation heatmap
- Literacy rate histogram by development level
- Scatter: Literacy Rate vs Life Expectancy
- Box plot: Maternal Mortality by level
- Bar: Labor force participation by region

### Panel 2 — ML Model Results
- Side-by-side test vs CV accuracy comparison
- Confusion matrix (best model)
- Feature importance (Random Forest)
- ROC curves for all 4 models

### Panel 3 — Advanced Analysis
- Violin plot: Literacy vs School Enrollment
- Precision / Recall / F1-Score per class
- 10-Fold CV distribution box plot (all models)

---

## 📁 Project Structure

```
female-population-ml/
│
├── female_population_ml_analysis.py   # Main ML script
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── .gitignore                         # Files to exclude from Git
│
├── data/                              # (Optional) Place Kaggle CSV here
│   └── gender_statistics.csv
│
└── outputs/                           # Auto-generated charts
    ├── 01_EDA_female_population.png
    ├── 02_ML_model_results.png
    └── 03_advanced_analysis.png
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/female-population-ml.git
cd female-population-ml
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run with Built-in Demo Data (No Download Needed)
```bash
python female_population_ml_analysis.py
```

### Run with Kaggle Dataset
1. Download the dataset from Kaggle (link above)
2. Place the CSV in the `data/` folder
3. Uncomment the line in the script:
```python
# df = pd.read_csv("data/gender_statistics.csv")
```
4. Run the script

---

## 📊 Results

> Results will vary slightly due to random seed and data generation.

| Model | Test Accuracy | CV Mean | CV Std |
|---|---|---|---|
| Random Forest | ~0.97 | ~0.95 | ±0.02 |
| XGBoost | ~0.96 | ~0.94 | ±0.02 |
| Logistic Regression | ~0.88 | ~0.87 | ±0.03 |
| KNN | ~0.91 | ~0.90 | ±0.03 |

🏆 **Best Model:** Random Forest / XGBoost (typically tied)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

Made with ❤️ for gender equality data science.
Feel free to ⭐ the repo if you found it useful!
