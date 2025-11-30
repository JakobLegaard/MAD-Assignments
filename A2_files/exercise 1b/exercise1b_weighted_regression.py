import numpy as np
import matplotlib.pyplot as plt
from linweighreg import WeightedLinearRegression
from linreg import LinearRegression
import os

# Check for CSV files in current directory
print("Current directory:", os.getcwd())
print("\nFiles in current directory:")
for f in os.listdir('.'):
    if f.endswith('.csv'):
        print(f"  - {f}")

# Try to find the CSV files
train_file = None
test_file = None

possible_train = ['boston_train.csv', 'boston_training.csv', 'train.csv']
possible_test = ['boston_test.csv', 'boston_testing.csv', 'test.csv']

for f in possible_train:
    if os.path.exists(f):
        train_file = f
        break

for f in possible_test:
    if os.path.exists(f):
        test_file = f
        break

if train_file is None or test_file is None:
    print("\n❌ ERROR: Could not find Boston Housing CSV files!")
    print("\nPlease make sure you have:")
    print("  - boston_train.csv")
    print("  - boston_test.csv")
    print("\nIn the same directory as this script.")
    print("\nYou should download these from Absalon (boston.zip)")
    exit(1)

print(f"\n✓ Found training data: {train_file}")
print(f"✓ Found test data: {test_file}")

# Load training and test data
train_data = np.loadtxt(train_file, delimiter=",")
test_data = np.loadtxt(test_file, delimiter=",")

X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]

print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# RMSE function
def rmse(t, tp):
    return np.sqrt(np.mean((t - tp)**2))

# ========================================
# Exercise 1b: Weighted Linear Regression
# ========================================

# Set weights as alpha_n = t_n^2 (square of target values)
alpha = t_train ** 2

print("\n" + "="*60)
print("WEIGHTED LINEAR REGRESSION (Exercise 1b)")
print("="*60)
print("\nWeights (alpha_n = t_n^2) statistics:")
print("  Min weight: %.2f" % alpha.min())
print("  Max weight: %.2f" % alpha.max())
print("  Mean weight: %.2f" % alpha.mean())
print("  Std weight: %.2f" % alpha.std())

# Fit weighted linear regression model using all features
model_weighted = WeightedLinearRegression()
model_weighted.fit(X_train, t_train, alpha)

# For comparison, also fit standard linear regression
model_standard = LinearRegression()
t_train_reshaped = t_train.reshape((len(t_train), 1))
model_standard.fit(X_train, t_train_reshaped)

# Print weights
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

print("\n" + "-"*60)
print("COMPARISON OF WEIGHTS")
print("-"*60)
print(f"{'Feature':<10} {'Weighted':<15} {'Standard':<15} {'Difference':<15}")
print("-"*60)
print(f"{'Intercept':<10} {model_weighted.w[0]:< 15.4f} {model_standard.w[0, 0]:< 15.4f} {model_weighted.w[0] - model_standard.w[0, 0]:< 15.4f}")
for i, name in enumerate(feature_names):
    w_weighted = model_weighted.w[i+1]
    w_standard = model_standard.w[i+1, 0]
    diff = w_weighted - w_standard
    print(f"{name:<10} {w_weighted:< 15.4f} {w_standard:< 15.4f} {diff:< 15.4f}")

# Make predictions on test set
t_pred_weighted = model_weighted.predict(X_test)
t_pred_standard = model_standard.predict(X_test).flatten()

# Calculate RMSE
rmse_weighted = rmse(t_test, t_pred_weighted)
rmse_standard = rmse(t_test, t_pred_standard)

print("\n" + "-"*60)
print("RMSE on test set:")
print("-"*60)
print("  Weighted Linear Regression: %.4f" % rmse_weighted)
print("  Standard Linear Regression: %.4f" % rmse_standard)
print("  Difference: %.4f" % (rmse_weighted - rmse_standard))
print("-"*60)

# Create scatter plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Weighted Linear Regression
axes[0].scatter(t_test, t_pred_weighted, alpha=0.5, s=30)
axes[0].plot([t_test.min(), t_test.max()], [t_test.min(), t_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('True House Prices ($1000s)', fontsize=11)
axes[0].set_ylabel('Predicted House Prices ($1000s)', fontsize=11)
axes[0].set_title('Weighted Linear Regression (α_n = t_n²)\nRMSE = %.4f' % rmse_weighted, fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Standard Linear Regression (for comparison)
axes[1].scatter(t_test, t_pred_standard, alpha=0.5, s=30)
axes[1].plot([t_test.min(), t_test.max()], [t_test.min(), t_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('True House Prices ($1000s)', fontsize=11)
axes[1].set_ylabel('Predicted House Prices ($1000s)', fontsize=11)
axes[1].set_title('Standard Linear Regression\nRMSE = %.4f' % rmse_standard, fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise1b_weighted_regression.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'exercise1b_weighted_regression.png'")
plt.show()

# ========================================
# ANALYSIS AND INTERPRETATION
# ========================================
print("\n" + "="*60)
print("ANALYSIS AND INTERPRETATION")
print("="*60)

print("\nWhat we EXPECTED to happen:")
print("  By setting alpha_n = t_n^2, we give MORE weight to data points")
print("  with HIGHER target values (expensive houses). This means:")
print("  - The model should fit expensive houses better")
print("  - The model should care less about prediction errors for cheap houses")
print("  - Weights should shift to minimize errors on high-value properties")

print("\nWhat we OBSERVED:")
print("  - RMSE comparison: Weighted = %.4f vs Standard = %.4f" % (rmse_weighted, rmse_standard))
if rmse_weighted > rmse_standard:
    print("  - The weighted model has HIGHER test error")
    print("  - This is expected because:")
    print("    * We optimized for expensive houses (high weights)")
    print("    * But test RMSE treats all predictions equally")
    print("    * The model trades off accuracy on cheap houses for expensive ones")
elif rmse_weighted < rmse_standard:
    print("  - The weighted model has LOWER test error")
    print("  - This suggests the weighting scheme helped generalization")
else:
    print("  - The two models have similar performance")

print("\nDo the additional weights have an INFLUENCE?")
# Calculate how many weights changed significantly (>10%)
significant_changes = 0
for i in range(len(model_weighted.w)):
    if i == 0:
        w_w = model_weighted.w[0]
        w_s = model_standard.w[0, 0]
    else:
        w_w = model_weighted.w[i]
        w_s = model_standard.w[i, 0]
    
    if abs(w_s) > 1e-6:  # avoid division by zero
        pct_change = abs((w_w - w_s) / w_s) * 100
        if pct_change > 10:
            significant_changes += 1

print(f"  - YES! {significant_changes} out of {len(model_weighted.w)} weights")
print(f"    changed by more than 10%")
print("  - The weighting scheme (alpha_n = t_n^2) DOES affect the solution")
print("  - Higher-valued houses receive quadratically more importance")
print("  - This fundamentally changes the optimization objective")

print("\n" + "="*60)