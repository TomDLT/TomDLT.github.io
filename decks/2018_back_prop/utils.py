"""Utils used in the RNN class, and to build a dataset"""
import numpy as np

N_DIGITS = 8


def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)


def array_to_decimal(array):
    return int(''.join([str(i) for i in array]), 10)


def decimal_to_array(num, n_digits=N_DIGITS):
    return np.int_(list(("%%0%dd" % n_digits) % num)[-n_digits:])


def test_array_to_decimal():
    for a in range(1000):
        assert a == array_to_decimal(decimal_to_array(a))


def build_training_set(n_samples=1000, n_digits=N_DIGITS, random_state=42):
    rng = np.random.RandomState(random_state)
    X_train = rng.randint(10, size=(n_samples, 2 * n_digits))
    y_train = np.zeros((n_samples, 2 * n_digits), dtype=int)

    for i in range(n_samples):
        a = array_to_decimal(X_train[i, :n_digits])
        b = array_to_decimal(X_train[i, n_digits:])
        y_train[i] = decimal_to_array(a + b, 2 * n_digits)

    return X_train, y_train


def print_X_y(X, y):
    n_digits = X.shape[1] // 2
    print('             ', X[0, :n_digits])
    print('            +', X[0, n_digits:])
    print('-----------------------------')
    print(y[0])
