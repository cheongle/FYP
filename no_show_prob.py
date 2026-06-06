import pandas as pd
import matplotlib.pyplot as plt

# Load engineered data
df = pd.read_csv("featured_noshow.csv")

# Drop target features
X = df.drop(columns=['no_show', 'scheduledday', 'appointmentday'])
y = df['no_show']

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost
from xgboost import XGBClassifier

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=1,
    eval_metric='auc'
)

model.fit(X_train, y_train)

# Check feature importance
from xgboost import plot_importance
plot_importance(model, max_num_features=10)
plt.show()

# Extract probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Calibrate probabilities
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
calibrated_model.fit(X_train, y_train)

y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

# Evaluate model
from sklearn.metrics import roc_auc_score

print("AUC:", roc_auc_score(y_test, y_prob_calibrated))

# Save probabilities
X_test = X_test.copy()
X_test['no_show_prob'] = y_prob_calibrated

X_test.to_csv("no_show_prob.csv", index=False)

# Extra metrics 
from sklearn.metrics import brier_score_loss, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
 
brier = brier_score_loss(y_test, y_prob_calibrated)
avg_precision = average_precision_score(y_test, y_prob_calibrated)
 
print(f"Brier Score       : {brier:.4f}  (calibration quality, lower = better, 0 = perfect)")
print(f"Average Precision : {avg_precision:.4f}  (area under PR curve)")
print(f"No-show rate      : {y_test.mean():.3f}  (baseline to compare against)")
print(f"Prob range        : [{y_prob_calibrated.min():.3f}, {y_prob_calibrated.max():.3f}]")
print(f"Mean predicted    : {y_prob_calibrated.mean():.3f}")
 
# Graphs
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Calibration curve
fraction_pos, mean_pred = calibration_curve(y_test, y_prob_calibrated, n_bins=10)
axes[0].plot(mean_pred, fraction_pos, 's-', label='XGB + Isotonic')
axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
axes[0].set_xlabel('Mean predicted probability')
axes[0].set_ylabel('Fraction of actual no-shows')
axes[0].set_title('Calibration Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
 
# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_calibrated)
axes[1].plot(recall, precision, lw=2)
axes[1].axhline(y=y_test.mean(), color='r', linestyle='--', label=f'Baseline ({y_test.mean():.2f})')
axes[1].set_xlabel('Recall  (of actual no-shows caught)')
axes[1].set_ylabel('Precision  (flagged patients who actually no-show)')
axes[1].set_title('Precision-Recall Curve\n(use to choose MILP threshold)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150)
plt.show()