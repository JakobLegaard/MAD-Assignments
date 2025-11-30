import numpy as np
import matplotlib.pyplot as plt
from reglr import RegularizedLinearRegression

print("="*70)
print("Exercise 2: Polynomial Fitting with Regularized Linear Regression")
print("="*70)

# Load the Olympic 100m data
raw = np.genfromtxt('men-olympics-100.txt', delimiter=' ')

# Extract years (first column) and winning times (second column)
X = raw[:, 0]  # Years
t = raw[:, 1]  # First place times

print(f"\nData loaded: {len(X)} Olympic games")
print(f"Year range: {X.min():.0f} - {X.max():.0f}")
print(f"Time range: {t.min():.2f}s - {t.max():.2f}s")

# Normalize X for numerical stability
X_mean = X.mean()
X_std = X.std()
X_normalized = (X - X_mean) / X_std

print(f"\nNormalized X: mean={X_mean:.1f}, std={X_std:.1f}")

# Generate lambda values to test
lambda_values = np.logspace(-8, 0, 100, base=10)
print(f"\nTesting {len(lambda_values)} λ values from {lambda_values.min():.2e} to {lambda_values.max():.2e}")

# Leave-One-Out Cross-Validation
def leave_one_out_cv(X, t, lambda_reg):
    """
    Perform leave-one-out cross-validation.
    
    Parameters
    ----------
    X : Array of shape [n_samples,]
    t : Array of shape [n_samples,]
    lambda_reg : float
        Regularization parameter
    
    Returns
    -------
    cv_error : float
        Mean squared error from leave-one-out CV
    """
    n = len(X)
    errors = []
    
    for i in range(n):
        # Create training set by leaving out sample i
        X_train = np.delete(X, i)
        t_train = np.delete(t, i)
        
        # Create test set with only sample i
        X_test = np.array([X[i]])
        t_test = np.array([t[i]])
        
        # Train model on training set
        model = RegularizedLinearRegression(lambda_reg=lambda_reg)
        model.fit(X_train.reshape(-1, 1), t_train)
        
        # Predict on test sample
        t_pred = model.predict(X_test.reshape(-1, 1))
        
        # Compute squared error
        error = (t_test[0] - t_pred[0]) ** 2
        errors.append(error)
    
    # Return mean squared error
    return np.mean(errors)

# Compute CV error for each lambda
print("\nPerforming leave-one-out cross-validation...")
cv_errors = []

for i, lam in enumerate(lambda_values):
    if i % 20 == 0:
        print(f"  Progress: {i}/{len(lambda_values)} λ values tested")
    
    cv_error = leave_one_out_cv(X_normalized, t, lam)
    cv_errors.append(cv_error)

cv_errors = np.array(cv_errors)

# Find best lambda
best_idx = np.argmin(cv_errors)
best_lambda = lambda_values[best_idx]
best_cv_error = cv_errors[best_idx]

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"\nBest λ: {best_lambda:.6e}")
print(f"Best CV Error (MSE): {best_cv_error:.6f}")
print(f"Best CV Error (RMSE): {np.sqrt(best_cv_error):.6f}")

# Train models with λ=0 (no regularization) and best λ
print(f"\n{'='*70}")
print("Model Comparison")
print(f"{'='*70}")

# Model 1: No regularization (λ=0)
model_no_reg = RegularizedLinearRegression(lambda_reg=0.0)
model_no_reg.fit(X_normalized.reshape(-1, 1), t)
w_no_reg = model_no_reg.get_weights()

print(f"\nModel with λ = 0 (No Regularization):")
print(f"  w₀ (intercept): {w_no_reg[0]:.6f}")
print(f"  w₁ (slope):     {w_no_reg[1]:.6f}")
print(f"  CV Error (MSE): {leave_one_out_cv(X_normalized, t, 0.0):.6f}")

# Model 2: Best λ
model_best = RegularizedLinearRegression(lambda_reg=best_lambda)
model_best.fit(X_normalized.reshape(-1, 1), t)
w_best = model_best.get_weights()

print(f"\nModel with λ = {best_lambda:.6e} (Best):")
print(f"  w₀ (intercept): {w_best[0]:.6f}")
print(f"  w₁ (slope):     {w_best[1]:.6f}")
print(f"  CV Error (MSE): {best_cv_error:.6f}")

# Interpretation
print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")
print(f"\nThe negative slope (w₁ ≈ {w_best[1]:.3f}) indicates that Olympic")
print(f"100m times are DECREASING over the years - athletes are getting faster!")
print(f"\nRegularization effect:")
print(f"  - Without regularization: |w₁| = {abs(w_no_reg[1]):.6f}")
print(f"  - With regularization:    |w₁| = {abs(w_best[1]):.6f}")
print(f"  - Change: {((abs(w_best[1]) - abs(w_no_reg[1])) / abs(w_no_reg[1]) * 100):.2f}%")
print(f"\nRegularization shrinks the weights slightly to prevent overfitting.")

# Plot CV error vs lambda
plt.figure(figsize=(12, 5))

# Plot 1: CV Error vs Lambda (log scale)
plt.subplot(1, 2, 1)
plt.semilogx(lambda_values, cv_errors, 'b-', linewidth=2)
plt.semilogx(best_lambda, best_cv_error, 'ro', markersize=10, label=f'Best λ = {best_lambda:.2e}')
plt.xlabel('λ (Regularization Parameter)', fontsize=12)
plt.ylabel('Leave-One-Out CV Error (MSE)', fontsize=12)
plt.title('Cross-Validation Error vs Regularization', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Plot 2: Fitted models with data
plt.subplot(1, 2, 2)

# Generate predictions for plotting
X_plot = np.linspace(X_normalized.min(), X_normalized.max(), 200)
t_pred_no_reg = model_no_reg.predict(X_plot.reshape(-1, 1))
t_pred_best = model_best.predict(X_plot.reshape(-1, 1))

# Convert back to original scale for plotting
X_plot_original = X_plot * X_std + X_mean

plt.scatter(X, t, color='black', s=50, alpha=0.6, label='Olympic Data', zorder=3)
plt.plot(X_plot_original, t_pred_no_reg, 'b--', linewidth=2, label=f'λ = 0', alpha=0.7)
plt.plot(X_plot_original, t_pred_best, 'r-', linewidth=2, label=f'λ = {best_lambda:.2e}', alpha=0.7)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Winning Time (seconds)', fontsize=12)
plt.title('Linear Fit: Men\'s Olympic 100m', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise2_cv_plot.png', dpi=300, bbox_inches='tight')
print(f"\n{'='*70}")
print("Plot saved as 'exercise2_cv_plot.png'")
print(f"{'='*70}")
plt.show()

# Summary table
print(f"\n{'='*70}")
print("SUMMARY TABLE")
print(f"{'='*70}")
print(f"\n{'Model':<25} {'λ':<15} {'w₀':<15} {'w₁':<15} {'CV Error':<15}")
print("-" * 85)
print(f"{'No Regularization':<25} {0.0:<15.2e} {w_no_reg[0]:<15.6f} {w_no_reg[1]:<15.6f} {leave_one_out_cv(X_normalized, t, 0.0):<15.6f}")
print(f"{'Best Regularization':<25} {best_lambda:<15.2e} {w_best[0]:<15.6f} {w_best[1]:<15.6f} {best_cv_error:<15.6f}")
print("-" * 85)