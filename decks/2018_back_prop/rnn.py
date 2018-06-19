"""RNN class, you need to fill the BPTT method"""
import operator
import numpy as np

from utils import softmax, build_training_set


class RNN():
    def __init__(self, input_dim=10, hidden_dim=100, bptt_truncate=5,
                 random_state=42):
        # Assign instance variables
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.random_state = random_state

        # Randomly initialize the network parameters
        scale_input = np.sqrt(1. / input_dim)
        scale_hidden = np.sqrt(1. / hidden_dim)
        rng = np.random.RandomState(random_state)
        self.U = rng.uniform(-1, 1, (hidden_dim, input_dim)) * scale_input
        self.V = rng.uniform(-1, 1, (input_dim, hidden_dim)) * scale_hidden
        self.W = rng.uniform(-1, 1, (hidden_dim, hidden_dim)) * scale_hidden

    def forward_propagation(self, x):
        # The total number of time steps
        n_times = len(x)

        # We add one additional element for the initial hidden states, set to 0
        states = np.zeros((n_times + 1, self.hidden_dim))
        states[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step
        outputs = np.zeros((n_times, self.input_dim))

        for t in np.arange(n_times):
            # Note that we are indexing U by x[t].
            # This is the same as multiplying U with a one-hot vector.
            states[t] = np.tanh(self.U[:, x[t]] + self.W.dot(states[t - 1]))
            outputs[t] = softmax(self.V.dot(states[t]))

        # We not only return the calculated outputs, but also the hidden states
        # as we will use them later to calculate the gradients.
        return outputs, states

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        outputs, _ = self.forward_propagation(x)
        return np.argmax(outputs, axis=1)

    def calculate_total_loss(self, x, y):
        loss = 0
        for i in np.arange(len(y)):
            outputs, _ = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" digit
            correct_word_predictions = outputs[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            loss += -1 * np.sum(np.log(correct_word_predictions))

        return loss

    def calculate_loss(self, x, y):
        loss = self.calculate_total_loss(x, y)
        # Divide the total loss by the number of training examples
        loss /= np.sum((len(y_i) for y_i in y))

        return loss

    def bptt(self, x, y):
        n_times = len(y)
        # Perform forward propagation
        outputs, states = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_outputs = outputs
        delta_outputs[np.arange(len(y)), y] -= 1.

        # For each output backwards...
        for t in np.arange(n_times)[::-1]:
            # update dLdV
            # ...

            # Initial delta calculation
            # ...

            # We backpropagate for at most self.bptt_truncate steps
            t0 = max(0, t - self.bptt_truncate)
            # Backpropagation through time
            for bptt_step in np.arange(t0, t + 1)[::-1]:
                # update dLdW
                # ...

                # update dLdU
                # ...

                # Update delta for next step
                # ...

                pass

        return dLdU, dLdV, dLdW

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation.
        bptt_gradients = self.bptt(x, y)

        # Gradient check for each parameter
        for pidx, pname in enumerate(['U', 'V', 'W']):
            # Get the actual parameter value from the mode, e.g. self.W
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter %s with size %d." %
                  (pname, np.prod(parameter.shape)))
            # Iterate over each element of the parameter matrix
            it = np.nditer(parameter, flags=['multi_index'],
                           op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # Gradient for this parameter calculated with backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate the relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(
                    backprop_gradient - estimated_gradient) / (
                        np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print("Gradient Check ERROR: parameter=%s ix=%s" % (pname,
                                                                        ix))
                    print("+h Loss: %f" % gradplus)
                    print("-h Loss: %f" % gradminus)
                    print("Estimated_gradient: %f" % estimated_gradient)
                    print("Backpropagation gradient: %f" % backprop_gradient)
                    print("Relative Error: %f" % relative_error)
                    return
                it.iternext()
            print("Gradient check for parameter %s passed." % (pname))

    def bptt_viz(self, x, y, only_paths_of_size=0):
        n_times = len(y)
        # Perform forward propagation
        outputs, states = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_outputs = outputs
        delta_outputs[np.arange(len(y)), y] -= 1.

        # For each output backwards...
        for t in np.arange(n_times)[::-1]:
            # update dLdV
            # ...

            # Initial delta calculation
            # ...

            # We backpropagate for at most self.bptt_truncate steps
            t0 = max(0, t - self.bptt_truncate)
            # Backpropagation through time
            for bptt_step in np.arange(t0, t + 1)[::-1]:

                if t - bptt_step == only_paths_of_size:
                    # update dLdW
                    # ...

                    pass

                # update dLdU
                # ...

                # Update delta for next step
                # ...

                pass

        return dLdU, dLdV, dLdW


def run_gradient_check():
    X_train, y_train = build_training_set(n_samples=10)
    model = RNN(input_dim=10, hidden_dim=100, bptt_truncate=1000)
    model.gradient_check(X_train[0], y_train[0])
