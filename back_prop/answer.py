"""Implementation of the BPTT and of the viz of exercise 3."""
import matplotlib.pyplot as plt


class RNN():
    def bptt(self, x, y):
        # example of solution
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
            dLdV += np.outer(delta_outputs[t], states[t].T)

            # Initial delta calculation
            delta_t = self.V.T.dot(delta_outputs[t]) * (1 - (states[t] ** 2))

            # We backpropagate for at most self.bptt_truncate steps
            t0 = max(0, t - self.bptt_truncate)
            # Backpropagation through time
            for bptt_step in np.arange(t0, t + 1)[::-1]:
                # update dLdW
                dLdW += np.outer(delta_t, states[bptt_step - 1])

                # update dLdU
                dLdU[:, x[bptt_step]] += delta_t

                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (
                    1 - states[bptt_step - 1] ** 2)

        return [dLdU, dLdV, dLdW]


def plot_gradient_propagation():
    # example of solution
    X_train, y_train = build_training_set(n_samples=10)
    model = RNN(input_dim=10, hidden_dim=100, bptt_truncate=1000)

    n_step_array = np.arange(10)
    mean_abs_dLdW = np.zeros(n_step_array.size)
    for ii, n_step in enumerate(n_step_array):
        dLdU, dLdV, dLdW = model.bptt_viz(X_train[0], y_train[0],
                                          only_paths_of_size=n_step)
        mean_abs_dLdW[ii] = np.mean(np.abs(dLdW.ravel()))

    plt.plot(n_step_array, mean_abs_dLdW, '-o')
    plt.title('Contributions of different path lengths to the gradient')
    plt.show()
