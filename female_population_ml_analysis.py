="""
============================================================
  FEMALE POPULATION DATA ANALYSIS WITH ML MODELS
  Kaggle Dataset: "Gender Statistics" by World Bank
  Kaggle URL: https://www.kaggle.com/datasets/theworldbank/world-bank-intl-education
  Alternative: https://www.kaggle.com/datasets/andrewmvd/world-population-by-gender
============================================================

KAGGLE DATASET NAME TO SEARCH:
  → "Gender Development Index" or
  → "world-population-female" or
  → Use built-in synthetic data below (no download needed for demo)

4 ML MODELS USED:
  1. Random Forest Classifier
  2. Logistic Regression
  3. XGBoost Classifier
  4. K-Nearest Neighbors (KNN)

RUN:
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost
  python female_population_ml_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  STEP 1: LOAD / GENERATE DATASET
# ─────────────────────────────────────────────
# If you have the Kaggle CSV, replace this block with:
#   df = pd.read_csv("gender_statistics.csv")
# Kaggle dataset: https://www.kaggle.com/datasets/theworldbank/world-bank-intl-education

np.random.seed(42)
n = 1000

regions = ['South Asia', 'Sub-Saharan Africa', 'East Asia', 'Europe', 'Latin America', 'North America']
region_col = np.random.choice(regions, n, p=[0.20, 0.22, 0.18, 0.15, 0.15, 0.10])

df = pd.DataFrame({
    'Region': region_col,
    'Female_Life_Expectancy':    np.random.normal(72, 8, n).clip(50, 90),
    'Female_Literacy_Rate':      np.random.normal(78, 18, n).clip(20, 100),
    'Female_Labor_Force_Pct':    np.random.normal(48, 12, n).clip(10, 80),
    'Maternal_Mortality_Rate':   np.random.exponential(120, n).clip(5, 800),
    'Female_School_Enrollment':  np.random.normal(80, 15, n).clip(20, 100),
    'Female_Internet_Usage_Pct': np.random.normal(55, 22, n).clip(5, 100),
    'GDP_Per_Capita_USD':        np.random.exponential(12000, n).clip(500, 70000),
    'Urban_Population_Pct':      np.random.normal(58, 20, n).clip(15, 100),
    'Female_Political_Rep_Pct':  np.random.normal(25, 12, n).clip(2, 60),
})

# Target: Development Level based on composite score
score = (
    df['Female_Life_Expectancy'] * 0.3 +
    df['Female_Literacy_Rate']   * 0.25 +
    df['Female_Labor_Force_Pct'] * 0.15 +
    df['Female_Internet_Usage_Pct'] * 0.15 +
    df['Female_School_Enrollment'] * 0.15 -
    df['Maternal_Mortality_Rate'] * 0.05
)
bins = [score.min()-1, score.quantile(0.33), score.quantile(0.66), score.max()+1]
df['Development_Level'] = pd.cut(score, bins=bins, labels=['Low', 'Medium', 'High'])

print("✅ Dataset created: {} rows × {} columns".format(*df.shape))
print("\n📋 Class Distribution:")
print(df['Development_Level'].value_counts())
print("\n📊 Dataset Preview:")
print(df.head(3).to_string())

# ─────────────────────────────────────────────
#  STEP 2: EDA VISUALIZATIONS
# ─────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = {'Low': '#E74C3C', 'Medium': '#F39C12', 'High': '#27AE60'}
COLOR_MAIN = '#8E44AD'

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Female Population Data — Exploratory Data Analysis',
             fontsize=22, fontweight='bold', color='#2C3E50', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Plot 1: Class Distribution ---
ax1 = fig.add_subplot(gs[0, 0])
counts = df['Development_Level'].value_counts()
bars = ax1.bar(counts.index, counts.values,
               color=[PALETTE[l] for l in counts.index], edgecolor='white', linewidth=1.5)
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(int(bar.get_height())), ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.set_title('Development Level Distribution', fontweight='bold', fontsize=12)
ax1.set_xlabel('Development Level')
ax1.set_ylabel('Count')
ax1.set_ylim(0, counts.max() * 1.15)

# --- Plot 2: Life Expectancy by Region ---
ax2 = fig.add_subplot(gs[0, 1:])
region_means = df.groupby('Region')['Female_Life_Expectancy'].mean().sort_values()
bars2 = ax2.barh(region_means.index, region_means.values,
                 color=sns.color_palette('viridis', len(region_means)), edgecolor='white')
for bar in bars2:
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.1f}', va='center', fontsize=10, fontweight='bold')
ax2.set_title('Avg Female Life Expectancy by Region', fontweight='bold', fontsize=12)
ax2.set_xlabel('Life Expectancy (Years)')
ax2.set_xlim(0, region_means.max() * 1.12)

# --- Plot 3: Correlation Heatmap ---
ax3 = fig.add_subplot(gs[1, 0:2])
num_cols = df.select_dtypes(include=np.number).columns
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=ax3, linewidths=0.5, annot_kws={'size': 8},
            cbar_kws={'shrink': 0.8})
ax3.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=12)
ax3.tick_params(axis='x', rotation=30)
ax3.tick_params(axis='y', rotation=0)

# --- Plot 4: Literacy Rate Distribution by Development Level ---
ax4 = fig.add_subplot(gs[1, 2])
for level, color in PALETTE.items():
    subset = df[df['Development_Level'] == level]['Female_Literacy_Rate']
    ax4.hist(subset, bins=20, alpha=0.65, color=color, label=level, edgecolor='white')
ax4.set_title('Female Literacy Rate\nby Development Level', fontweight='bold', fontsize=12)
ax4.set_xlabel('Literacy Rate (%)')
ax4.set_ylabel('Frequency')
ax4.legend(title='Level')

# --- Plot 5: Scatter — Literacy vs Life Expectancy ---
ax5 = fig.add_subplot(gs[2, 0])
for level, color in PALETTE.items():
    sub = df[df['Development_Level'] == level]
    ax5.scatter(sub['Female_Literacy_Rate'], sub['Female_Life_Expectancy'],
                c=color, label=level, alpha=0.5, s=25, edgecolors='none')
ax5.set_title('Literacy Rate vs\nLife Expectancy', fontweight='bold', fontsize=12)
ax5.set_xlabel('Literacy Rate (%)')
ax5.set_ylabel('Life Expectancy (Yrs)')
ax5.legend(title='Level', fontsize=8)

# --- Plot 6: Box Plot — Maternal Mortality ---
ax6 = fig.add_subplot(gs[2, 1])
df_box = df[df['Maternal_Mortality_Rate'] < 600]
bp = ax6.boxplot([df_box[df_box['Development_Level']==l]['Maternal_Mortality_Rate'].values
                  for l in ['Low','Medium','High']],
                 labels=['Low','Medium','High'], patch_artist=True,
                 medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], PALETTE.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_title('Maternal Mortality\nby Development Level', fontweight='bold', fontsize=12)
ax6.set_ylabel('Maternal Mortality Rate')

# --- Plot 7: Labor Force % ---
ax7 = fig.add_subplot(gs[2, 2])
df.groupby('Region')['Female_Labor_Force_Pct'].mean().sort_values().plot(
    kind='bar', ax=ax7, color=sns.color_palette('magma', 6),
    edgecolor='white', linewidth=1)
ax7.set_title('Avg Female Labor Force\nParticipation by Region', fontweight='bold', fontsize=12)
ax7.set_xlabel('')
ax7.set_ylabel('Labor Force (%)')
ax7.tick_params(axis='x', rotation=30)

plt.savefig('/mnt/user-data/outputs/01_EDA_female_population.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✅ EDA chart saved.")

# ─────────────────────────────────────────────
#  STEP 3: PREPROCESSING
# ─────────────────────────────────────────────
le_region = LabelEncoder()
df['Region_Enc'] = le_region.fit_transform(df['Region'])

le_target = LabelEncoder()
df['Target'] = le_target.fit_transform(df['Development_Level'])

FEATURES = ['Female_Life_Expectancy', 'Female_Literacy_Rate', 'Female_Labor_Force_Pct',
            'Maternal_Mortality_Rate', 'Female_School_Enrollment', 'Female_Internet_Usage_Pct',
            'GDP_Per_Capita_USD', 'Urban_Population_Pct', 'Female_Political_Rep_Pct', 'Region_Enc']

X = df[FEATURES]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n✅ Train: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────
#  STEP 4: TRAIN 4 ML MODELS
# ─────────────────────────────────────────────
models = {
    'Random Forest':      RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='auto', random_state=42),
    'XGBoost':            XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                        use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'KNN':                KNeighborsClassifier(n_neighbors=7),
}

results = {}
for name, model in models.items():
    Xtr = X_train_sc if name in ['Logistic Regression', 'KNN'] else X_train
    Xte = X_test_sc  if name in ['Logistic Regression', 'KNN'] else X_test
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, Xtr, y_train, cv=5, scoring='accuracy')
    results[name] = {
        'model': model, 'y_pred': y_pred, 'accuracy': acc,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'Xte': Xte
    }
    print(f"  {name:22s} | Test Acc: {acc:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
#  STEP 5: MODEL COMPARISON VISUALIZATIONS
# ─────────────────────────────────────────────
fig2, axes = plt.subplots(2, 2, figsize=(18, 14))
fig2.suptitle('ML Model Results — Female Population Development Classification',
              fontsize=20, fontweight='bold', color='#2C3E50', y=1.01)

model_names = list(results.keys())
accuracies  = [results[m]['accuracy'] for m in model_names]
cv_means    = [results[m]['cv_mean']  for m in model_names]
cv_stds     = [results[m]['cv_std']   for m in model_names]

BAR_COLORS = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12']

# --- Chart A: Accuracy Comparison ---
ax = axes[0, 0]
x = np.arange(len(model_names))
width = 0.35
b1 = ax.bar(x - width/2, accuracies, width, label='Test Accuracy',
            color=BAR_COLORS, edgecolor='white', linewidth=1.5, alpha=0.9)
b2 = ax.bar(x + width/2, cv_means, width, label='CV Mean Accuracy',
            color=BAR_COLORS, edgecolor='white', linewidth=1.5, alpha=0.5,
            yerr=cv_stds, capsize=5)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=10, fontsize=10)
ax.set_ylim(0.5, 1.05)
ax.set_title('A) Model Accuracy Comparison\n(Test vs Cross-Validation)',
             fontweight='bold', fontsize=12)
ax.set_ylabel('Accuracy')
ax.legend(fontsize=9)
ax.axhline(y=np.mean(accuracies), color='gray', linestyle='--', linewidth=1, alpha=0.7,
           label='Mean Accuracy')

# --- Chart B: Confusion Matrix (Best Model) ---
best_model_name = max(results, key=lambda m: results[m]['accuracy'])
best_result = results[best_model_name]
ax = axes[0, 1]
cm = confusion_matrix(y_test, best_result['y_pred'])
class_labels = le_target.classes_
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=class_labels, yticklabels=class_labels,
            linewidths=1, linecolor='white', cbar_kws={'shrink': 0.8},
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title(f'B) Confusion Matrix\n{best_model_name} (Best Model | Acc={best_result["accuracy"]:.3f})',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)

# --- Chart C: Feature Importance (Random Forest) ---
ax = axes[1, 0]
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
bars = ax.barh(importances.index, importances.values, color=colors_fi, edgecolor='white')
for bar in bars:
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.3f}', va='center', fontsize=9)
ax.set_title('C) Feature Importance\n(Random Forest)', fontweight='bold', fontsize=12)
ax.set_xlabel('Importance Score')
ax.set_xlim(0, importances.max() * 1.18)

# --- Chart D: ROC Curve (One-vs-Rest for all models) ---
ax = axes[1, 1]
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
line_styles = ['-', '--', '-.', ':']
for i, (mname, mcolor, lstyle) in enumerate(zip(model_names, BAR_COLORS, line_styles)):
    m = results[mname]['model']
    Xte = results[mname]['Xte']
    try:
        if hasattr(m, 'predict_proba'):
            ovr = OneVsRestClassifier(type(m)(**m.get_params()))
            ovr.fit(X_train_sc if mname in ['Logistic Regression','KNN'] else X_train, y_train)
            y_score = ovr.predict_proba(Xte)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=mcolor, lw=2, linestyle=lstyle,
                label=f'{mname} (AUC = {roc_auc:.3f})')
    except Exception:
        pass
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Random Baseline')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_title('D) ROC Curves — All Models\n(One-vs-Rest, Macro Avg)',
             fontweight='bold', fontsize=12)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=8, loc='lower right')
ax.fill_between([0,1],[0,1], alpha=0.05, color='gray')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/02_ML_model_results.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✅ ML Results chart saved.")

# ─────────────────────────────────────────────
#  STEP 6: ADVANCED VISUALIZATIONS
# ─────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(20, 6))
fig3.suptitle('Advanced Analysis — Female Population ML Insights',
              fontsize=18, fontweight='bold', color='#2C3E50')

# --- Violin Plot ---
ax = axes3[0]
df_melt = df.melt(id_vars='Development_Level',
                  value_vars=['Female_Literacy_Rate', 'Female_School_Enrollment'],
                  var_name='Metric', value_name='Value')
df_melt['Metric'] = df_melt['Metric'].str.replace('Female_', '').str.replace('_', ' ')
sns.violinplot(data=df_melt, x='Development_Level', y='Value', hue='Metric',
               palette=['#3498DB', '#E91E63'], ax=ax, split=True, inner='quart',
               order=['Low','Medium','High'])
ax.set_title('Literacy vs School Enrollment\nby Development Level', fontweight='bold', fontsize=12)
ax.set_xlabel('Development Level')
ax.set_ylabel('Rate (%)')
ax.legend(title='Metric', fontsize=8)

# --- Per-class Precision/Recall/F1 (Best model) ---
ax = axes3[1]
report = classification_report(y_test, best_result['y_pred'],
                                target_names=class_labels, output_dict=True)
metrics_df = pd.DataFrame({cls: [report[cls]['precision'], report[cls]['recall'], report[cls]['f1-score']]
                            for cls in class_labels}, index=['Precision','Recall','F1-Score'])
metrics_df.T.plot(kind='bar', ax=ax, color=['#3498DB','#E74C3C','#27AE60'],
                  edgecolor='white', linewidth=1, rot=0)
ax.set_ylim(0, 1.15)
ax.set_title(f'Precision / Recall / F1-Score\n{best_model_name}', fontweight='bold', fontsize=12)
ax.set_xlabel('Development Level Class')
ax.set_ylabel('Score')
ax.legend(fontsize=9)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=8, padding=2)

# --- CV Score Box Plot across all models ---
ax = axes3[2]
cv_all = []
for mname, mdata in results.items():
    Xtr_cv = X_train_sc if mname in ['Logistic Regression','KNN'] else X_train
    scores = cross_val_score(mdata['model'], Xtr_cv, y_train, cv=10, scoring='accuracy')
    for s in scores: cv_all.append({'Model': mname, 'CV Accuracy': s})
cv_df = pd.DataFrame(cv_all)
bp = ax.boxplot([cv_df[cv_df['Model']==m]['CV Accuracy'].values for m in model_names],
                labels=[m.replace(' ', '\n') for m in model_names],
                patch_artist=True, medianprops=dict(color='black', linewidth=2.5),
                flierprops=dict(marker='o', markersize=5, alpha=0.5))
for patch, color in zip(bp['boxes'], BAR_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_title('10-Fold CV Accuracy Distribution\nAll Models', fontweight='bold', fontsize=12)
ax.set_ylabel('Accuracy')
ax.set_ylim(0.5, 1.05)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/03_advanced_analysis.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✅ Advanced analysis chart saved.")

# ─────────────────────────────────────────────
#  STEP 7: FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  FINAL MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"{'Model':<25} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>10}")
print("-"*60)
for mname in model_names:
    r = results[mname]
    print(f"{mname:<25} {r['accuracy']:>10.4f} {r['cv_mean']:>10.4f} {r['cv_std']:>10.4f}")
print("="*60)
print(f"\n🏆 Best Model: {best_model_name} | Accuracy: {best_result['accuracy']:.4f}")
print("\n📁 Output files saved to /mnt/user-data/outputs/:")
print("   01_EDA_female_population.png")
print("   02_ML_model_results.png")
print("   03_advanced_analysis.png")
print("   female_population_ml_analysis.py")
