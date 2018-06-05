"""Not used for the moment, but could be used to extend the homework"""
import sys
import datetime

from rnn import RNN
from utils import build_training_set


def sdg_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100,
                   evaluate_loss_after=5):
    """Outer SGD Loop

    Parameters
    ----------
    - model: The RNN model instance
    - X_train: The training data set
    - y_train: The training data labels
    - learning_rate: Initial learning rate for SGD
    - nepoch: Number of times to iterate through the complete dataset
    - evaluate_loss_after: Evaluate the loss after this many epochs
    """
    model.sgd_step = sdg_step

    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" %
                  (time, num_examples_seen, epoch, loss))

            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()

        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

    return losses


def test_sgd():
    X_train, y_train = build_training_set(n_samples=100)
    # Train on a small subset of the data to see what happens
    model = RNN()
    train_with_sgd(model, X_train, y_train, nepoch=5, evaluate_loss_after=1)
