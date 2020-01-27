# Manjunath, Samarth
# 1001-522-809
# 2019-09-23
# Assignment-01-01

import numpy as np
import itertools

class Perceptron(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self._initialize_weights()

    
        
    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes

        """
        self.weights = []
        self.weights=np.random.randn(self.number_of_classes,self.input_dimensions+1)
    
    def one_hot(a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).T

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
        self.weights = []
        self.weights= np.zeros((self.number_of_classes,self.input_dimensions+1))

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]
        """
        x_shape=X.shape
        y_shape=self.weights.shape
        result=y_shape[1]-x_shape[0]
        size=(result,x_shape[1])
        one=np.ones(size,None,'C')
        c=np.append(one,X,axis=0)
        summation=np.dot(self.weights,c)
        final_result=np.array([])
        for i in np.nditer(summation):
            if i>0:
                value=1
            else:
                value=0
            final_result=np.append(final_result,value)
        predict=final_result.reshape(y_shape[0],x_shape[1])
        return predict

    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights)

    def train(self, X, Y, num_epochs=10, alpha=0.001):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        x_shape=X.shape
        y_shape=self.weights.shape
        result=y_shape[1]-x_shape[0]
        size=(result,x_shape[1])
        one=np.ones(size,None,'C')
        c=np.append(one,X,axis=0)
        for i in range(num_epochs):
            for j in range(size[1]):
                self.prediction=self.predict(X[:,[j]])
                self.weights+=alpha*np.dot((Y[:,[j]]-self.prediction),np.transpose(c[:,[j]]))

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the output is not hte same as the desired output, Y,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :return percent_error
        """
        false_count = 0
        y_shape=Y.shape
        samples = X.shape[1]
        for j in range(y_shape[1]):
            final_prediction=self.predict(X[:,[j]])
            desired_output=Y[:,[j]]
            if any(final_prediction != desired_output):
                false_count = false_count +1
        percent_error=false_count/samples
        return percent_error

if __name__ == "__main__":
    """
    This main program is a sample of how to run your program.
    You may modify this main program as you desire.
    """
    input_dimensions=12
    number_of_classes=4
    number_of_samples=240
    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes,seed=2)
    X_train=np.random.randn(input_dimensions,number_of_samples)
    y_index=np.random.randint(0,number_of_classes,number_of_samples)
    
    Y_train=one_hot(y_index, number_of_classes)
    model.initialize_all_weights_to_zeros()
    model.train(X_train,Y_train,num_epochs=40,alpha=0.1)
    np.testing.assert_allclose(model.weights,
                            [[-0.5,  0.29194245, -0.46718261, -0.53100857, -0.05492079,  0.46907704, -0.20595509,  0.11783862, -0.0311177,  -0.54649448,  0.16957215, -0.32380812, 0.17996748],
                                [-0.3, -0.1175635,   0.06158569,  0.33174567, -0.20930241,  0.14466174,  0.18245077, -0.00909707,  0.3258576,   0.3529057,   0.19412227,  0.13648932, 0.06296011],
                                [-0.9,  0.12709901,  0.29077714, -0.12591162, -0.11445311,  0.19285466,  0.08919241, -0.40098894, -0.16019999, -0.24824168,  0.16594415,  0.1781876, -0.31126413],
                                [-0.9,  0.02515515,  0.01417218,  0.06317648,  0.26527972, -0.11759975,  0.037144,    0.34220146, -0.25388846,  0.3085406,  -0.28865999,  0.08495056, 0.06156483]], rtol=1e-3, atol=1e-3)