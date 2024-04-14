"""Multilayer perceptron (MLP) for supervised learning.

Notes
-----
  This script is version v0. It provides the base for all subsequent
  iterations of the project.
  
Requirements
------------
  See "requirements.txt"
  
Comments
--------
  The network has an input layer, one hidden layer and an output layer.
  The logistic function is used as the activation function associated with hidden and output layers.
  The network is trained using backpropagation via gradient descent.
  The learning method is online, where synaptic weights are adjusted on a sample-by-sample basis.
  The epoch in the MLP involves the entire training sample of input-target response pairs.
  The training samples are randomly shuffled after each epoch.
  The data is split into training and testing.
  The training data is further split into a training subset and a validation subset.
  The training session is stopped periodically to test the model performance on the validation subset.
  The training, validation and testing data were scaled independently, with each feature having zero mean and unit variance.
  
"""

#%% import libraries and modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np  
import random
import os

#%% figure parameters
plt.rcParams['figure.figsize'] = (6,6)
plt.rcParams['lines.linewidth'] = 5
plt.rcParams['font.size']= 15

#%% build MLP class
class MLP:
    """MLP class."""
    def __init__(self, dim_input=2, dim_hidden=20, dim_output=1, random_state=42):
        self.dim_input = dim_input # input layer dimension (default: 2)
        self.dim_hidden = dim_hidden # hidden layer dimension (default: 5)
        self.dim_output = dim_output # output layer dimension (default: 1)
        self.random_state = random_state # random initialization state (default: 2)
    
    def make_inputs(self, n_samples, radius=1.0, width=0.1, distance=-0.35): # default: (radius=1.0, width=0.1, distance=-0.35)
        """Build non-linearly separable data."""
        class_size = (n_samples//2, 1) # specify class sample size
        np.random.seed(self.random_state)
        
        """Class A."""
        r1 = np.random.normal(loc=radius, scale=width, size=class_size) # generate data from normal distribution
        theta1 = np.random.uniform(size=class_size) * np.pi # generate angles from uniform distribution
        Ax1 = r1 * np.cos(theta1) # generate first feature of class A
        Ax2 = r1 * np.sin(theta1) + distance # generate second feature of class A
        
        """Class B."""
        r2 = np.random.normal(loc=radius, scale=width, size=class_size) # generate data from normal distribution
        theta2 = np.random.uniform(size=class_size) * np.pi + np.pi # generate angles from uniform distribution
        Bx1 = r2 * np.cos(theta2) + radius # generate first feature of class B
        Bx2 = r2 * np.sin(theta2) # generate second feature of class B
        
        classA = np.hstack((Ax1, Ax2)) # concatenate class A features
        classB = np.hstack((Bx1, Bx2)) # concatenate class B features
        
        X = np.vstack((classA, classB)) # concatenate classes
        return X
    
    def scale_inputs(self, X):
        """Scale input features to unit mean and zero variance."""
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0) # apply feature transformation
        return X_scaled
    
    def make_targets(self, n_samples):
        """Build target patterns."""
        class_a_targets = np.zeros([n_samples//2, self.dim_output]) + 0 # assign class A targets to 0
        class_b_targets = np.zeros([n_samples//2, self.dim_output]) + 1 # assign class B targets to 1
        y = np.vstack((class_a_targets, class_b_targets)) # concatenate targets
        return y
    
    def train_val_test_split(self, X, y):
        """Perform train, validation and test split."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) # split data into training and testing
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train) # split training data into training and validation subsets
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def initialize_network(self):
        """Initialize synaptic weights and biases."""
        np.random.seed(self.random_state) # set random seed
        
        w_hidden = np.random.uniform(low=0.001, high=1.0, size=(self.dim_hidden, self.dim_input)) # synaptic weights from input to hidden
        w_output = np.random.uniform(low=0.001, high=1.0, size=(self.dim_output, self.dim_hidden)) # synaptic weights from hidden to output
        bias_hidden = np.ones(self.dim_hidden) # hidden layer bias
        bias_output = np.ones(self.dim_output) # output layer bias
        return w_hidden, w_output, bias_hidden, bias_output
    
    def activation_function(self, y, a=1):
        """Apply activation function."""        
        f_y = 1 / (1 + np.exp(a * -y)) # logistic function
        return f_y
    
    def activation_function_derivative(self, f_y, a=1):
        """Compute the activation function derivative."""
        f_y_prime = a * f_y * (1 - f_y) # logistic function derivative
        return f_y_prime
    
    def step_function(self, activation, threshold=0.5):
        """Apply step function."""
        activation[activation >= threshold] = 1 # if activation is greater than or equal to threshold, set to 1
        activation[activation <  threshold] = 0 # if activation is less than threshold, set to 0
        return activation
    
    def fit(self, X_train, y_train, X_val, y_val, min_learning_rate=0.0001, max_learning_rate=0.1, num_training_epochs=50, validation_epoch_cycle=5):
        """Fit model to training data."""
        w_hidden, w_output, bias_hidden, bias_output = self.initialize_network() # initialize network
        n_samples = X_train.shape[0] # number of training samples
        learning_rate = np.linspace(max_learning_rate, min_learning_rate, num_training_epochs) # linear annealing of learning rate
        training_loss_epochs = [] # empty list 
        validation_loss_epochs = [] # empty list
        for epoch_index in range(num_training_epochs):
            
            if epoch_index % validation_epoch_cycle == 0: # for every 'validation_epoch_cycle' epoch cycles
                """Perform cross-validation."""
                validation_loss_epoch, _, _ = self.predict(X_val, y_val, w_hidden, w_output, bias_hidden, bias_output) # compute validation error of epoch
                validation_loss_epochs.append(validation_loss_epoch) # store validation error of epoch
                
            training_loss_samples = [] # empty list
            random_sample = random.sample(range(n_samples), n_samples) # generate random samples
            sample_index = 0 # initialize sample index
            while sample_index < n_samples:
                
                """Input and target selection."""
                x = X_train[random_sample[sample_index]] # random selection of input pattern
                target = y_train[random_sample[sample_index]] # random selection of corresponding target pattern
                
                """Forward computation."""
                hidden_in = bias_hidden + np.dot(w_hidden, x) # from input layer to hidden layer
                hidden_out = self.activation_function(hidden_in) # apply activation function to hidden layer
                output_in = bias_output + np.dot(w_output, hidden_out) # from hidden layer to output layer
                output_out = self.activation_function(output_in) # apply activation function to output layer
                error = target - output_out # compute error
                
                training_loss_sample = np.dot(error, error) # compute squared error of sample
                training_loss_samples.append(training_loss_sample) # store squared error of sample
                
                """Backward computation."""
                output_local_gradient = self.activation_function_derivative(output_out) * error # compute local gradient of output layer
                w_output = w_output + learning_rate[epoch_index] * np.outer(output_local_gradient, hidden_out) # update weights from hidden layer to output layer
                bias_output = bias_output + learning_rate[epoch_index] * output_local_gradient # update bias applied to output layer
                hidden_local_gradient = self.activation_function_derivative(hidden_out) * np.dot(output_local_gradient, w_output) # compute local gradient of hidden layer
                w_hidden = w_hidden + learning_rate[epoch_index] * np.outer(hidden_local_gradient, x) # update weights from input layer to hidden layer
                bias_hidden = bias_hidden + learning_rate[epoch_index] * hidden_local_gradient # update bias applied to hidden layer
                
                sample_index += 1 # increment sample index
                
            training_loss_epoch = np.mean(training_loss_samples) # compute mean squared error of epoch
            training_loss_epochs.append(training_loss_epoch) # store mean squared error of epoch
            print('Epoch {}/{} - train loss: {:.4f} - val loss: {:.4f}'.format(epoch_index+1, num_training_epochs, training_loss_epoch, validation_loss_epoch)) # print mean squared error
        return training_loss_epochs, validation_loss_epochs, w_hidden, w_output, bias_hidden, bias_output
    
    def predict(self, X_test, y_test, w_hidden, w_output, bias_hidden, bias_output):
        """Perform model predictions."""
        n_samples = X_test.shape[0] # number of samples
        
        y_preds = np.empty(shape=y_test.shape) # empty array
        testing_loss_samples = [] # empty list
        random_sample = random.sample(range(n_samples), n_samples) # generate random samples
        sample_index = 0 # initialize sample index
        while sample_index < n_samples:
            
            """Input and target selection."""
            x = X_test[random_sample[sample_index]] # random selection of input pattern
            target = y_test[random_sample[sample_index]] # random selection of corresponding target pattern
            
            """Forward computation."""
            hidden_in = bias_hidden + np.dot(w_hidden, x) # from input layer to hidden layer
            hidden_out = self.activation_function(hidden_in) # apply activation function to hidden layer
            output_in = bias_output + np.dot(w_output, hidden_out) # from hidden layer to output layer
            output_out = self.activation_function(output_in) # apply activation function to output layer
            y_preds[random_sample[sample_index], :] = output_out # store output predictions
            error = target - output_out # compute error
            
            testing_loss_sample = np.dot(error, error) # compute squared error of sample
            testing_loss_samples.append(testing_loss_sample) # store squared error of sample
            
            sample_index += 1 # increment sample index
            
        testing_loss_epoch = np.mean(testing_loss_samples) # compute mean squared error of epoch
        return testing_loss_epoch, testing_loss_samples, y_preds
    
#%% specify filepath
cwd = os.getcwd() # get current working directory
fileName = 'images' # specify filename

if os.path.exists(os.path.join(cwd, fileName)) == False: # if path does not exist
    os.makedirs(fileName) # create directory with specified filename
    os.chdir(os.path.join(cwd, fileName)) # change cwd to the given path
    cwd = os.getcwd() # get current working directory
else:
    os.chdir(os.path.join(cwd, fileName)) # change cwd to the given path
    cwd = os.getcwd() # get current working directory
    
#%% instantiate MLP class
model = MLP()

#%% generate inputs and corresponding target patterns for training, validation and testing
n_samples = 10_000 # specify total number of samples

X = model.make_inputs(n_samples=n_samples) # input data
y = model.make_targets(n_samples=n_samples) # target data

X_train, X_val, X_test, y_train, y_val, y_test = model.train_val_test_split(X, y) # split data into training, validation and testing

X_train = model.scale_inputs(X_train) # scale training data
X_val = model.scale_inputs(X_val) # scale validation data
X_test = model.scale_inputs(X_test) # scale test data

#%% plot data
fig, ax = plt.subplots()
Ax1, Ax2 = X[:n_samples//2, 0], X[:n_samples//2, 1] # class A
Bx1, Bx2 = X[n_samples//2:, 0], X[n_samples//2:, 1] # class B
plt.scatter(Ax1, Ax2, marker='.', c='r', label='class A', s=20)
plt.scatter(Bx1, Bx2, marker='.', c='b', label='class B', s=20)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Double moon dataset')
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'figure_1'))

#%% plot activation function
fig, ax = plt.subplots()
plt.plot(np.linspace(-10, 10, 1000), model.activation_function(np.linspace(-10, 10, 1000)), color='k')
plt.xlabel('y')
plt.ylabel('f(y)')
plt.title('Logistic function')
plt.axvline(x=0, linewidth=1.0, color='k')
plt.axhline(y=0.5, linewidth=1.0, color='k')
plt.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'figure_2'))

#%% train network
training_loss_epochs, validation_loss_epochs, w_hidden, w_output, bias_hidden, bias_output = model.fit(X_train, y_train, X_val, y_val)

fig, ax = plt.subplots()
plt.plot(np.arange(0, len(training_loss_epochs), 1), training_loss_epochs, color='k', label='training', zorder=0)
plt.scatter(np.arange(0, len(training_loss_epochs), model.fit.__defaults__[-1]), validation_loss_epochs, color='g', label='validation', s=100)
plt.xlabel('Number of epochs')
plt.ylabel('MSE')
plt.title('Learning curve')
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'figure_3'))

#%% test network
testing_loss_epoch, testing_loss_samples, y_preds = model.predict(X_test, y_test, w_hidden, w_output, bias_hidden, bias_output)
model.step_function(y_preds)

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_preds, ax=ax, colorbar=False, cmap=plt.cm.Blues)
disp.ax_.set_title('Confusion matrix')
fig.savefig(os.path.join(os.getcwd(), 'figure_4'))
