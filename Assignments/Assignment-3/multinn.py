# Manjunath, Samarth
# 1001-522-809
# 2019-10-28
# Assignment-03-01

import tensorflow as tf
import numpy as np



class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each the input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None

    def add_layer(self, num_nodes, activation_function):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param activation_function: Activation function for the layer
         :return: None
         """
        if self.weights:
            weights_of_n = np.array(
                [[np.random.normal() for column in range(num_nodes)] for row in range(self.weights[-1].shape[1])])
        else:
            weights_of_n = np.array(
                [[np.random.normal() for column in range(num_nodes)] for row in range(self.input_dimension)])

        weights_sample_t = tf.Variable(weights_of_n, trainable=True)
        bias_sample_n = np.array([[np.random.normal() for column in range(num_nodes)]])
        bias_sample_t = tf.Variable(bias_sample_n, trainable=True)

        self.weights.append(weights_sample_t)
        self.biases.append(bias_sample_t)
        self.activations.append(activation_function)

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        weight_of_the_layer = self.weights[layer_number]
        return weight_of_the_layer

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases). Note that the biases shape should be [1][number_of_nodes]
         """
        bias_of_the_layer = self.biases[layer_number]
        return bias_of_the_layer

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number] = weights

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number] = biases

    def set_loss_function(self, loss_fn):
        """
        This function sets the loss function.
        :param loss_fn: Loss function
        :return: none
        """
        self.loss = loss_fn

    def sigmoid(self, x):

        return tf.nn.sigmoid(x)

    def linear(self, x):
        return x

    def relu(self, x):
        out = tf.nn.relu(x)
        return out

    def cross_entropy_loss(self, y, y_hat):
        """
        This function calculates the cross entropy loss
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual outputs values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        y_hat=tf.Variable(X)
        for i in range(len(self.weights)):
            Weight_of_X = tf.matmul(y_hat, self.get_weights_without_biases(i), name="Weight_of_X")
            bias_of_X = tf.add(Weight_of_X, self.get_biases(i), "bias_of_X")
            y_hat = self.activations[i](bias_of_X)
        return y_hat


    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8, regularization_coeff=1e-6):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :param regularization_coeff: regularization coefficient
         :return: None
         """
        for epoch in range(num_epochs):
            for i in range(0, X_train.shape[0], batch_size):
                row_of_the_last = i + batch_size
                if row_of_the_last > X_train.shape[0]:
                    row_of_the_last = X_train.shape[0]
                i1 = tf.Variable(X_train[i : row_of_the_last,:])
                i2 = tf.Variable(y_train[i : row_of_the_last])

                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(self.weights)
                    tape.watch(self.biases)
                    tResults = self.predict(i1)
                    loss = self.cross_entropy_loss(i2, tResults)
                for j in range(len(self.weights)):
                    dl_dw = tape.gradient(loss, self.get_weights_without_biases(j))
                    dl_db = tape.gradient(loss, self.get_biases(j))
                    scaled_dl_dw = tf.scalar_mul(alpha, dl_dw)
                    scaled_dl_db = tf.scalar_mul(alpha, dl_db)
                    n_of_weight = tf.subtract(self.get_weights_without_biases(j), scaled_dl_dw)
                    n_of_bias = tf.subtract(self.get_biases(j), scaled_dl_db)
                    self.set_weights_without_biases(n_of_weight, j)
                    self.set_biases(n_of_bias, j)

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        output = self.predict(X).numpy()
        index_m = output.argmax(axis=1)
        expected_one_hot = self.One_Hot_conversion(y).transpose()
        final_one_hot = (index_m[:, None] == np.arange(output.shape[1])).astype(float)
        err = 0
        for i in range(output.shape[0]):
            var_1 = expected_one_hot[i]
            var_2 = final_one_hot[i]
            if not np.allclose(var_1, var_2):
                err = err + 1
        percent_error = err / output.shape[0]
        return percent_error


    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """

        matrix_of_confusion = np.zeros((self.weights[-1].shape[1], self.weights[-1].shape[1]))
        output = self.predict(X).numpy()
        index_m = output.argmax(axis=1)
        one_hot_encoder = self.One_Hot_conversion(y).transpose()
        final_one_hot = (index_m[:, None] == np.arange(output.shape[1])).astype(float)
        for i in range(final_one_hot.shape[0]):
            indices = np.where(final_one_hot[i] == 1)
            if indices[0].size != 0:
                final_prediction = indices[0][0]
                if not np.array_equal(final_one_hot[i], one_hot_encoder[i]):
                    matrix_of_confusion[y[0], final_prediction] += 1
                else:
                    matrix_of_confusion[final_prediction, final_prediction] += 1
        return matrix_of_confusion


    def One_Hot_conversion(self, Y):
        h_one = np.zeros((self.weights[-1].shape[1], Y.shape[0]))
        h_one[Y, np.arange(Y.shape[0])] = 1
        return h_one

if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist

    np.random.seed(seed=1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Reshape and Normalize data
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_train = y_train.flatten().astype(np.int32)
    input_dimension = X_train.shape[1]
    indices = list(range(X_train.shape[0]))
    # np.random.shuffle(indices)
    number_of_samples_to_use = 500
    X_train = X_train[indices[:number_of_samples_to_use]]
    y_train = y_train[indices[:number_of_samples_to_use]]
    multi_nn = MultiNN(input_dimension)
    number_of_classes = 10
    activations_list = [multi_nn.sigmoid, multi_nn.sigmoid, multi_nn.linear]
    number_of_neurons_list = [50, 20, number_of_classes]
    for layer_number in range(len(activations_list)):
        multi_nn.add_layer(number_of_neurons_list[layer_number], activation_function=activations_list[layer_number])
    for layer_number in range(len(multi_nn.weights)):
        W = multi_nn.get_weights_without_biases(layer_number)
        W = tf.Variable((np.random.randn(*W.shape)) * 0.1, trainable=True)
        multi_nn.set_weights_without_biases(W, layer_number)
        b = multi_nn.get_biases(layer_number=layer_number)
        b = tf.Variable(np.zeros(b.shape) * 0, trainable=True)
        multi_nn.set_biases(b, layer_number)
    multi_nn.set_loss_function(multi_nn.cross_entropy_loss)
    percent_error = []
    for k in range(10):
        multi_nn.train(X_train, y_train, batch_size=100, num_epochs=20, alpha=0.8)
        percent_error.append(multi_nn.calculate_percent_error(X_train, y_train))
    confusion_matrix = multi_nn.calculate_confusion_matrix(X_train, y_train)
    print("Percent error: ", np.array2string(np.array(percent_error), separator=","))
    print("************* Confusion Matrix ***************\n", np.array2string(confusion_matrix, separator=","))