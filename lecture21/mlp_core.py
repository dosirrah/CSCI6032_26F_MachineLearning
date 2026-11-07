# mlp_core.py
import numpy as np


class Neuron:
    def __init__(self, learning_rate=0.01):
        self._learning_rate = learning_rate
        self._theta_history = []  # Parameter vector including weights and bias
        self._cache = None        # Cache for storing intermediate values during forward pass

    @property
    def last_activation(self):
        """Returns the last activation value computed during the forward pass."""
        if self._cache is None:
            raise ValueError(
                "No cache available. Ensure that forward() has been called "
                "before accessing last_activation."
            )
        return self._cache['a']

    @property
    def theta(self):
        return self._theta_history[-1]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, n_features):
        """Randomly initialize theta (weights + bias as the first element)."""
        theta = np.random.randn(n_features + 1)
        self._theta_history.append(theta)

    def set_weights(self, weights):
        """
        Sets the neuron's weights and bias.

        Parameters:
        - weights: A numpy array containing [bias, weight1, weight2, ..., weightN]
        """
        self._theta_history = [weights.copy()]
        self._cache = None  # Reset cache when weights are set

    def forward(self, x):
        """Forward pass for a single sample."""
        # Add a bias term (x_0 = 1) for each x sample
        x_with_bias = np.insert(x, 0, 1)
        z = np.dot(self.theta, x_with_bias)
        a = self.sigmoid(z)
        # Store the cache for backpropagation
        self._cache = {'x_with_bias': x_with_bias, 'z': z, 'a': a}
        return a

    def backward(self, da=None, dz=None):
        """
        Backward pass: updates theta and returns the gradient of the loss with
        respect to this neuron's inputs, i.e., the da for the previous layer.

        Pass either da = dL/da, or dz = dL/dz directly.  The output neuron
        with binary cross-entropy loss passes dz = a - y, because the
        sigmoid's derivative cancels against the loss's derivative.
        """
        cache = self._cache
        if cache is None:
            raise ValueError(
                "No cache available. Ensure that forward() has been called "
                "before backward()."
            )
        x_with_bias = cache['x_with_bias']
        a = cache['a']
        if dz is None:
            dz = da * a * (1 - a)  # Chain rule through the sigmoid
        gradient = dz * x_with_bias

        # Gradient flowing to the previous layer, computed with the weights
        # *before* this update (exclude the bias term)
        theta = self.theta.copy()
        da_prev = dz * theta[1:]

        # Update parameters (theta)
        theta -= self._learning_rate * gradient
        self._theta_history.append(theta)

        # Clear the cache after backward pass
        self._cache = None
        return da_prev

    def update_backward(self):
        """Revert the last update (for stepping backward)."""
        if len(self._theta_history) > 1:
            self._theta_history.pop()
        # No need to manage cache here since it's cleared after backward()
        return self._theta_history[-1]

    def activations(self, X):
        """Compute activations (outputs) for all input samples X."""
        # Add bias term to each sample
        X_with_bias = np.insert(X, 0, 1, axis=1)
        z = np.dot(X_with_bias, self.theta)
        a = self.sigmoid(z)
        return a


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, learning_rate=0.01):
        """
        Initializes a multi-layer neural network.

        Parameters:
        - input_size: Number of features in the input data.
        - hidden_sizes: List containing the number of neurons in each hidden layer.
        - learning_rate: Learning rate for weight updates.
        """
        self.learning_rate = learning_rate
        self.layers = []  # List to hold each layer's neurons
        self.step = 0     # To keep track of training steps

        # Initialize the hidden layers
        previous_size = input_size
        for layer_size in hidden_sizes:
            layer = [Neuron(learning_rate) for _ in range(layer_size)]
            for neuron in layer:
                neuron.initialize_parameters(previous_size)
            self.layers.append(layer)
            previous_size = layer_size

        # Initialize the output neuron
        self.output_neuron = Neuron(learning_rate)
        self.output_neuron.initialize_parameters(previous_size)

    def forward(self, x):
        """
        Performs a forward pass through the network for a single sample.

        Parameters:
        - x: Input feature vector (numpy array).
        """
        activations = x
        for layer in self.layers:
            next_activations = []
            for neuron in layer:
                a = neuron.forward(activations)
                next_activations.append(a)
            activations = np.array(next_activations)
        output = self.output_neuron.forward(activations)
        return output

    def backward(self, y_true):
        """
        Performs a backward pass through the network, updating weights.

        Parameters:
        - y_true: True label for the input sample.
        """
        # With a sigmoid output and binary cross-entropy loss,
        # dL/dz at the output neuron is simply a - y.
        a_output = self.output_neuron.last_activation
        da_current = self.output_neuron.backward(dz=a_output - y_true)

        # Backpropagate through the hidden layers, last to first.  da_current
        # holds dL/da for each neuron in the layer being processed.
        for layer in reversed(self.layers):
            da_prev = np.zeros(len(layer[0].theta) - 1)  # one entry per input to this layer
            for i, neuron in enumerate(layer):
                # Each neuron returns its contribution to dL/da of the previous
                # layer; the contributions from all neurons in this layer add up.
                da_prev = da_prev + neuron.backward(da=da_current[i])
            da_current = da_prev

    def step_forward(self, x, y_true):
        """
        Advances the training by one SGD update (forward and backward pass).

        Parameters:
        - x: Input feature vector (numpy array).
        - y_true: True label for the input sample.
        """
        self.forward(x)
        self.backward(y_true)
        self.step += 1

    def update_backward(self):
        """
        Reverts the network to the previous state by undoing the last update.
        """
        # Revert output neuron
        self.output_neuron.update_backward()
        # Revert hidden layers
        for layer in self.layers:
            for neuron in layer:
                neuron.update_backward()
        if self.step > 0:
            self.step -= 1

    def activations(self, X):
        """
        Computes activations for all input samples at each layer.

        Parameters:
        - X: Input samples (numpy array of shape (n_samples, n_features)).

        Returns:
        - activations_per_layer: List of activations for each layer.
        """
        activations = X
        activations_per_layer = [activations]

        for layer in self.layers:
            next_activations = []
            for neuron in layer:
                a = neuron.activations(activations)
                next_activations.append(a)
            activations = np.column_stack(next_activations)
            activations_per_layer.append(activations)

        # Output neuron activations
        output_activations = self.output_neuron.activations(activations)
        activations_per_layer.append(output_activations)
        return activations_per_layer
