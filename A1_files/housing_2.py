import numpy
import linreg
import matplotlib.pyplot as plt

# load data
train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

#RMSE function
def rmse(t, tp):
    return numpy.sqrt(numpy.mean((t - tp)**2))

# (b) fit linear regression using only the first feature
model_single = linreg.LinearRegression()
model_single.fit(X_train[:,0:1], t_train)

#weights printout and interpretation
print("\nWeights for single-feature model (CRIM only):")
print("  w0 (intercept): %.4f" % model_single.w[0, 0])
print("  w1 (CRIM coefficient): %.4f" % model_single.w[1, 0])
print("\nInterpretation:")
print("  - w0 = %.4f: The baseline house price when CRIM = 0" % model_single.w[0, 0])
print("  - w1 = %.4f: For each unit increase in crime rate," % model_single.w[1, 0])
print("    the house price decreases by $%.2f (in $1000s)" % abs(model_single.w[1, 0]))

# (c) fit linear regression model using all features
model_all = linreg.LinearRegression()
model_all.fit(X_train, t_train)

print("\nWeights for all-features model:")
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
print("  w0 (intercept): %.4f" % model_all.w[0, 0])
for i, name in enumerate(feature_names):
    print("  w%d (%s): %.4f" % (i+1, name, model_all.w[i+1, 0]))

# (d) evaluation of results

t_pred_single = model_single.predict(X_test[:,0:1])
rmse_single = rmse(t_test, t_pred_single)
t_pred_all = model_all.predict(X_test)
rmse_all = rmse(t_test, t_pred_all)
print("\nRMSE on test set:")
print("  Single feature model (CRIM only): %.4f" % rmse_single)
print("  All features model: %.4f" % rmse_all)
print("\nImprovement: %.4f (%.1f%% reduction in error)" % 
      (rmse_single - rmse_all, 100 * (rmse_single - rmse_all) / rmse_single))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(t_test, t_pred_single, alpha=0.5)
axes[0].plot([t_test.min(), t_test.max()], [t_test.min(), t_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('True House Prices ($1000s)')
axes[0].set_ylabel('Predicted House Prices ($1000s)')
axes[0].set_title('Single Feature Model (CRIM)\nRMSE = %.4f' % rmse_single)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(t_test, t_pred_all, alpha=0.5)
axes[1].plot([t_test.min(), t_test.max()], [t_test.min(), t_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('True House Prices ($1000s)')
axes[1].set_ylabel('Predicted House Prices ($1000s)')
axes[1].set_title('All Features Model\nRMSE = %.4f' % rmse_all)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise4_scatter_plots.png', dpi=300)
plt.show()