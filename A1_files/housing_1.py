import numpy
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

# (a) compute mean of prices on training set
mean_price = numpy.mean(t_train)
print("\nMean of house prices on training set: $%.2f (in $1000's)" % mean_price)

# (b) RMSE function
def rmse(t, tp):
    """
    Compute root-mean-square error between true values t and predictions tp.
    
    Params
    t : Array of shape [n_samples, 1] - true values
    tp : Array of shape [n_samples, 1] - predicted values
    
    returning:
    rmse_value : float - the RMSE
    """
    return numpy.sqrt(numpy.mean((t - tp)**2))

t_pred_mean = numpy.full_like(t_test, mean_price)

rmse_value = rmse(t_test, t_pred_mean)
print("RMSE on test set using mean model: %.4f" % rmse_value)

# (c) visualization of results
plt.figure(figsize=(8, 6))
plt.scatter(t_test, t_pred_mean, alpha=0.5)
plt.xlabel('True House Prices ($1000s)')
plt.ylabel('Predicted House Prices ($1000s)')
plt.title('Mean Model: True vs Predicted House Prices')
plt.plot([t_test.min(), t_test.max()], [t_test.min(), t_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_plot.png', dpi=300)
plt.show()