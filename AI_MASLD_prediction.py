# -*- coding: utf-8 -*-
"""AI-MASLD-prediction

## Libraries and Dependencies
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.base import clone
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt

"""## Data Preprocessing"""

# Load datasets separately for overall cohort, men, and women
data = pd.read_csv('AI MASLD.csv')


# Define features and target variable
X = data.drop(columns=['MASLD'])
y = data['MASLD']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=y)

# Define categorical and numerical features
categorical_features = ['sex', 'dm', 'htn', 'hlp',
                        'fmh_fld', 'fmh_CVD',
                        'fmh_dm', 'statin', 'asa',
                        'hypothyroid', 'cigarette_cat', 'alcohol_cat',
                        'physical_act'
]
numerical_features = ['age', 'ef', 'wc',
                      'hc', 'sbp', 'dbp',
                      'bmi', 'no_cmrf', 'sitting_time_hour',
                      'gfr', 'bun']

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', 'passthrough', categorical_features)
    ])

"""## Hyperparameter Optimization"""

# Define `calculate_metrics` before using it
def calculate_metrics(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    train_f1 = classification_report(y_train, y_train_pred, output_dict=True)['1']['f1-score']
    test_f1 = classification_report(y_test, y_test_pred, output_dict=True)['1']['f1-score']

    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
    sensitivity_train = tp / (tp + fn)
    specificity_train = tn / (tn + fp)
    npv_train = tn / (tn + fn)
    ppv_train = tp / (tp + fp)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    sensitivity_test = tp / (tp + fn)
    specificity_test = tn / (tn + fp)
    npv_test = tn / (tn + fn)
    ppv_test = tp / (tp + fp)

    return {
        'Train AUC': train_auc,
        'Test AUC': test_auc,
        'Train F1 Score': train_f1,
        'Test F1 Score': test_f1,
        'Train Sensitivity': sensitivity_train,
        'Test Sensitivity': sensitivity_test,
        'Train Specificity': specificity_train,
        'Test Specificity': specificity_test,
        'Train NPV': npv_train,
        'Test NPV': npv_test,
        'Train PPV': ppv_train,
        'Test PPV': ppv_test
    }

  # Define classifiers and their expanded parameter grids
model_params = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [100, 200, 500],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        }
    },
    'SGD Classifier': {
        'model': SGDClassifier(random_state=42),
        'params': {
            'classifier__loss': ['hinge', 'log_loss', 'modified_huber'],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__penalty': ['l2', 'l1', 'elasticnet'],
            'classifier__learning_rate': ['constant', 'optimal', 'adaptive'],
            'classifier__eta0': [0.01, 0.1]
        }
    },
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs', 'liblinear']
        }
    },
    'KNeighbors Classifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'classifier__n_neighbors': [3, 5, 10, 50],
            'classifier__leaf_size': [20, 30, 50],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__p': [1, 2]  # 1=Manhattan, 2=Euclidean
        }
    },
    'Gradient Boosting Classifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 4, 5],
            'classifier__subsample': [0.8, 1.0],
            'classifier__max_features': ['sqrt', 'log2']
        }
    },
    'GaussianNB': {
        'model': GaussianNB(),
        'params': {
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]
        }
    },
    'MLP Classifier': {
        'model': MLPClassifier(max_iter=2000, early_stopping=True, random_state=42),
        'params': {
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100), (50, 50, 50)],
            'classifier__activation': ['relu', 'tanh'],
            'classifier__solver': ['adam', 'lbfgs'],
            'classifier__alpha': [0.0001, 0.001, 0.01]
        }
    },
    'support vector machine': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf', 'poly'],
            'classifier__gamma': ['scale', 'auto']
        }
    }
}


# Dictionary to store results
results = {}

# Loop through models
for name, mp in model_params.items():
    print(f"Tuning and evaluating: {name}")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', mp['model'])
    ])

    grid = GridSearchCV(pipeline, mp['params'], cv=StratifiedKFold(n_splits=5),
                        scoring='roc_auc')

    grid.fit(X_train, y_train)

    # Best estimator after grid search
    best_model = grid.best_estimator_

    # Cross-validation scores
    cv_scores_auc = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
    cv_scores_f1 = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')

    results[name] = {
        'Best Parameters': json.dumps(grid.best_params_),
        'Mean AUC (Train)': np.mean(cv_scores_auc),
        'Mean F1 Score (Train)': np.mean(cv_scores_f1)
    }

    # Calculate metrics on train and test sets
    metrics = calculate_metrics(best_model, X_train, y_train, X_test, y_test)
    results[name].update(metrics)

# Create and save the result DataFrame
results_df = pd.DataFrame(results).transpose()
results_df = results_df.sort_values(by='Test AUC', ascending=False)
results_df.to_csv('model_evaluation_results_with_gridsearch.csv')

# Display result
print(results_df)

"""## Model performance"""

# Add ML classifiers here (with the best performing parameter combinations)
classifiers = [
    ('Random Forest', RandomForestClassifier(
        max_depth=None,
        min_samples_split=2,
        n_estimators=200,
        random_state=42
    )),

    ('SGD Classifier', SGDClassifier(
        alpha=0.01,
        eta0=0.1,
        learning_rate='adaptive',
        loss='hinge',
        penalty='l1',
        random_state=42
    )),

    ('Logistic Regression', LogisticRegression(
        C=0.1,
        penalty='l2',
        solver='lbfgs',
        random_state=42
    )),

    ('KNeighbors Classifier', KNeighborsClassifier(
        leaf_size=20,
        n_neighbors=50,
        p=2,
        weights='distance'
    )),

    ('Gradient Boosting Classifier', GradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        max_features='sqrt',
        n_estimators=100,
        subsample=0.8,
        random_state=42
    )),

    ('GaussianNB', GaussianNB(
        var_smoothing=1e-09
    )),

    ('MLP Classifier', MLPClassifier(
        activation='tanh',
        alpha=0.0001,
        hidden_layer_sizes=(50, 50, 50),
        solver='adam',
        max_iter=1000,
        random_state=42
    )),

    ('Support Vector Machine', SVC(
        C=0.1,
        gamma='scale',
        kernel='linear',
        random_state=42
    ))
]


# Function to calculate metrics
def calculate_metrics(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    def safe_divide(a, b):
        return a / b if b != 0 else 0

    def compute_all_metrics(y_true, y_pred, prefix=""):
        auc = roc_auc_score(y_true, y_pred)
        f1 = classification_report(y_true, y_pred, output_dict=True)['1']['f1-score']
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            f'{prefix}AUC': auc,
            f'{prefix}F1 Score': f1,
            f'{prefix}Sensitivity': safe_divide(tp, tp + fn),
            f'{prefix}Specificity': safe_divide(tn, tn + fp),
            f'{prefix}NPV': safe_divide(tn, tn + fn),
            f'{prefix}PPV': safe_divide(tp, tp + fp)
        }

    metrics = compute_all_metrics(y_train, y_train_pred, prefix="Train ")
    metrics.update(compute_all_metrics(y_test, y_test_pred, prefix="Test "))
    return metrics, model.predict(X_test), y_test

# Bootstrapping function for CI
def bootstrap_ci(metric_func, y_true, y_pred, n_iterations=1000, alpha=0.95):
    scores = []
    for _ in range(n_iterations):
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_pred[indices])
        scores.append(score)
    lower = np.percentile(scores, ((1.0 - alpha) / 2.0) * 100)
    upper = np.percentile(scores, (alpha + ((1.0 - alpha) / 2.0)) * 100)
    return f"{np.mean(scores):.3f} ({lower:.3f}-{upper:.3f})"

# Metric functions for bootstrap
metric_functions = {
    'Test AUC': roc_auc_score,
    'Test F1 Score': lambda y, p: classification_report(y, p, output_dict=True)['1']['f1-score'],
    'Test Sensitivity': lambda y, p: confusion_matrix(y, p).ravel()[3] / (confusion_matrix(y, p).ravel()[2] + confusion_matrix(y, p).ravel()[3]),
    'Test Specificity': lambda y, p: confusion_matrix(y, p).ravel()[0] / (confusion_matrix(y, p).ravel()[0] + confusion_matrix(y, p).ravel()[1]),
    'Test NPV': lambda y, p: confusion_matrix(y, p).ravel()[0] / (confusion_matrix(y, p).ravel()[0] + confusion_matrix(y, p).ravel()[2]),
    'Test PPV': lambda y, p: confusion_matrix(y, p).ravel()[3] / (confusion_matrix(y, p).ravel()[3] + confusion_matrix(y, p).ravel()[1]),
}

# Main evaluation loop
results = {}
for name, clf in classifiers:
    print(f"Evaluating: {name}")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

    # Cross-validated scores
    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='f1')

    results[name] = {
        'Mean AUC (Train)': np.mean(cv_auc),
        'Mean F1 Score (Train)': np.mean(cv_f1)
    }

    # Fit and compute metrics
    metrics, y_test_pred, y_test_true = calculate_metrics(pipeline, X_train, y_train, X_test, y_test)
    results[name].update(metrics)

    # Bootstrap CI for each test metric
    for metric, func in metric_functions.items():
        ci = bootstrap_ci(func, np.array(y_test_true), np.array(y_test_pred))
        results[name][metric + " (95% CI)"] = ci

# Convert results to DataFrame and save
results_df = pd.DataFrame(results).transpose()
results_df.to_csv("model_evaluation.csv")
print(results_df)

"""## Sensitivity analysis for top-perfoming algorithms (Cross-validation strategy)"""
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, random_state=42)

# Storage for results
fold_results = []  # will contain one row per fold per classifier
summary_results = []  # aggregated summary per classifier

# Helper function: compute metrics on test set
def compute_metrics(y_true, y_pred, y_score=None, pos_label=1):
    # Basic metrics
    auc = np.nan
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            # fallback: if y_score is shape (n_samples, n_classes) pick positive column
            try:
                auc = roc_auc_score(y_true, y_score[:, 1])
            except Exception:
                auc = np.nan
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    sensitivity = recall_score(y_true, y_pred, pos_label=pos_label)  # recall = sensitivity
    acc = accuracy_score(y_true, y_pred)
    # confusion matrix for specificity and NPV
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    ppv = precision  # same as precision
    return {
        'AUC': auc,
        'F1': f1,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'Accuracy': acc
    }

# Main loop: stratified k-fold
for name, clf in classifiers:
    print(f"Processing classifier: {name}")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', clf)])
    # list to collect each fold's metrics
    metrics_list = []

    fold_idx = 0
    for train_idx, test_idx in cv.split(X, y):
        fold_idx += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # fit
        pipeline.fit(X_train, y_train)

        # predict labels
        y_pred = pipeline.predict(X_test)

        # obtain score for AUC: try predict_proba, then decision_function
        y_score = None
        try:
            # pipeline.named_steps['classifier'] may be wrapped in pipeline; use pipeline.predict_proba
            y_score = pipeline.predict_proba(X_test)
            # if binary, keep positive class column
            if isinstance(y_score, np.ndarray) and y_score.ndim == 2:
                y_score = y_score[:, 1]
        except Exception:
            # try decision_function
            try:
                y_score = pipeline.decision_function(X_test)
            except Exception:
                y_score = None

        # compute metrics
        metrics = compute_metrics(y_test.values, y_pred, y_score=y_score, pos_label=1)
        metrics.update({
            'Classifier': name,
            'Fold': fold_idx,
            'Train_Size': len(train_idx),
            'Test_Size': len(test_idx)
        })
        metrics_list.append(metrics)
        fold_results.append(metrics)

    # aggregate summary (mean ± sd) across folds for this classifier
    df_metrics = pd.DataFrame(metrics_list)
    summary = {
        'Classifier': name,
        'AUC_mean': df_metrics['AUC'].mean(),
        'AUC_std': df_metrics['AUC'].std(),
        'F1_mean': df_metrics['F1'].mean(),
        'F1_std': df_metrics['F1'].std(),
        'Sensitivity_mean': df_metrics['Sensitivity'].mean(),
        'Sensitivity_std': df_metrics['Sensitivity'].std(),
        'Specificity_mean': df_metrics['Specificity'].mean(),
        'Specificity_std': df_metrics['Specificity'].std(),
        'PPV_mean': df_metrics['PPV'].mean(),
        'PPV_std': df_metrics['PPV'].std(),
        'NPV_mean': df_metrics['NPV'].mean(),
        'NPV_std': df_metrics['NPV'].std(),
        'Accuracy_mean': df_metrics['Accuracy'].mean(),
        'Accuracy_std': df_metrics['Accuracy'].std()
    }
    summary_results.append(summary)

# Save results
folds_df = pd.DataFrame(fold_results)
summary_df = pd.DataFrame(summary_results)

# Format summary metrics as "mean (± sd)" for readability 
def mean_sd_format(mean, sd):
    if np.isnan(mean):
        return "nan"
    return f"{mean:.3f} (\u00B1 {sd:.3f})"

readable_summary = []
for _, row in summary_df.iterrows():
    readable_summary.append({
        'Classifier': row['Classifier'],
        'AUC (mean ± SD)': mean_sd_format(row['AUC_mean'], row['AUC_std']),
        'F1 (mean ± SD)': mean_sd_format(row['F1_mean'], row['F1_std']),
        'Sensitivity (mean ± SD)': mean_sd_format(row['Sensitivity_mean'], row['Sensitivity_std']),
        'Specificity (mean ± SD)': mean_sd_format(row['Specificity_mean'], row['Specificity_std']),
        'PPV (mean ± SD)': mean_sd_format(row['PPV_mean'], row['PPV_std']),
        'NPV (mean ± SD)': mean_sd_format(row['NPV_mean'], row['NPV_std']),
        'Accuracy (mean ± SD)': mean_sd_format(row['Accuracy_mean'], row['Accuracy_std'])
    })

readable_summary_df = pd.DataFrame(readable_summary)

# Save both detailed fold-level results and the summarized table
folds_df.to_csv('model_fold_metrics_MASLD_total.csv', index=False)
summary_df.to_csv('model_summary_raw_MASLD_total.csv', index=False)
readable_summary_df.to_csv('model_summary_metrics_MASLD_total.csv', index=False)

print("Saved:")
print("- Detailed fold metrics -> model_fold_metrics.csv")
print("- Raw numeric summary -> model_summary_raw.csv")
print("- Readable summary -> model_summary_metrics.csv")

# show summary
print(readable_summary_df.sort_values(by='AUC (mean ± SD)', ascending=False))
"""## ROC Curve Analysis"""

# Set seaborn style
sns.set(style="whitegrid", font_scale=1.2)

# Recompute AUCs based on current fitting for correct plotting order
model_results = []

for name, clf in classifiers:
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clone(clf))  # prevent shared state
    ])
    pipeline.fit(X_train, y_train)

    # Get prediction scores
    if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score = pipeline.decision_function(X_test)

    # Compute ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Append all needed info
    model_results.append({
        'name': name,
        'clf': clf,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'auc_ci_str': results_df.loc[name, 'Test AUC (95% CI)']  # Keep CI string from table
    })

# Sort by actual ROC AUC from prediction
model_results = sorted(model_results, key=lambda x: x['roc_auc'], reverse=True)

# Define consistent color map
palette = sns.color_palette("tab10", len(model_results))
color_map = {model['name']: palette[idx] for idx, model in enumerate(model_results)}

# Plot
plt.figure(figsize=(10, 8))

for model in model_results:
    plt.plot(
        model['fpr'], model['tpr'],
        lw=2,
        color=color_map[model['name']],
        label=f"{model['name']} (AUC = {model['auc_ci_str']})"
    )

# Diagonal line
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

# Labels and title
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.title('ROC Curves for Predicting the Presence of MASLD in All Participants', fontsize=15, weight='bold')

# Legend
plt.legend(loc='lower right', fontsize=10, frameon=True, title='Model AUCs (95% CI) on the Test Set')

# Grid and layout
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# Save as high-resolution PDF
plt.savefig("roc_curves_MASLD_overall.pdf", format='pdf', dpi=600, bbox_inches='tight')

# Show
plt.show()

"""## Threshold Selection and Diagnostic Accuracy"""

# === Define and fit the models (RF for overall; GBC for men; LR for women) ===
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        max_depth=None,
        min_samples_split=2,
        n_estimators=200,
        random_state=42
    ))
])
rf_model.fit(X_train, y_train)

# === Get decision scores ===
y_proba = rf_model.predict_proba(X_test)[:, 1]

# === Normalize decision scores to [0, 1] for fair thresholding ===
thresholds = np.linspace(0, 1, 101)
results = []

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = precision_score(y_test, y_pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Threshold": threshold,
        "Test Sensitivity": sensitivity,
        "Test Specificity": specificity,
        "Test PPV": ppv,
        "Test NPV": npv,
        "Test F1 Score": f1
    })


# === Save to CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv("threshold_metrics_MASLD_curve.csv", index=False)

# Optional: show top few rows
print(results_df.head())

# customizing threshold for defining rule in and rule out metrics
# === Fit model (RF for overall; GBC for men; LR for women) ===
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        max_depth=None,
        min_samples_split=2,
        n_estimators=200,
        random_state=42
    ))
])
rf_model.fit(X_train, y_train)

# === Get decision scores ===
y_proba = rf_model.predict_proba(X_test)[:, 1]


# === Set your threshold here ===
your_threshold = 0.25  # You can change this to your optimal/rule-in/rule-out threshold
y_pred = (y_proba >= your_threshold).astype(int)

# === Define metric calculation ===
def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    youden = sens + spec - 1

    return sens, spec, ppv, npv, f1, youden

# === Bootstrap CI function ===
def bootstrap_ci(y_true, y_score, threshold, n_iterations=1000, alpha=0.95):
    metrics_list = []

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    for _ in range(n_iterations):
        indices = np.random.randint(0, len(y_true), len(y_true))
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        y_pred_boot = (y_score_boot >= threshold).astype(int)

        # skip resample with only one class
        if len(np.unique(y_true_boot)) < 2:
            continue

        m = compute_metrics(y_true_boot, y_pred_boot)
        metrics_list.append(m)

    metrics_array = np.array(metrics_list)
    ci_results = {}
    metric_names = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1 Score', 'Youden Index']

    for i, name in enumerate(metric_names):
        mean = np.mean(metrics_array[:, i])
        lower = np.percentile(metrics_array[:, i], 2.5)
        upper = np.percentile(metrics_array[:, i], 97.5)
        ci_results[name] = f"{mean:.3f} ({lower:.3f}–{upper:.3f})"

    return ci_results

# === Run CI analysis and print results ===
ci_results = bootstrap_ci(y_test, y_proba, threshold=your_threshold)
print(f"Performance metrics at threshold = {your_threshold}:")
for metric, value in ci_results.items():
    print(f"{metric}: {value}")

"""## Feature Importance Analysis"""

# === Feature Importance Analysis: RF for Overall Dataset; GBC for Men; LR for women ===
# Define pipeline with optimal model
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),  # your existing preprocessing pipeline
    ('classifier', RandomForestClassifier(
        max_depth=None,
        min_samples_split=2,
        n_estimators=200,
        random_state=42
    ))
])

# Fit the model
rf_model.fit(X_train, y_train)

# Helper function to extract feature names from preprocessor
def get_feature_names(preprocessor):
    try:
        return preprocessor.get_feature_names_out()
    except:
        output_features = []
        for name, transformer, columns in preprocessor.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out(columns)
            elif hasattr(transformer, 'get_feature_names'):
                names = transformer.get_feature_names(columns)
            else:
                names = columns
            output_features.extend([f"{name}__{col}" for col in names])
        return output_features

# Get feature names
feature_names = get_feature_names(rf_model.named_steps['preprocessor'])

# Extract feature importances from Random Forest
importances = rf_model.named_steps['classifier'].feature_importances_

# Create a DataFrame for plotting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()
