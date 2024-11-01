import numpy as np
import random


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initialization for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: the value of the MSE corresponding to the final value of w
        w: the final model returned by GD
    """
    n, _ = tx.shape
    w = initial_w

    for n_iter in range(max_iters):
        e = y - tx.dot(w)  # the error
        grad = (tx.T.dot(e)) * (-1 / n)
        w = w - grad * gamma
    e = y - tx.dot(
        w
    )  # we only need to compute the loss for the final value of w, but to do so we must refresh the error with the final value of w
    loss = e.T.dot(e) / (2 * n)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent (SGD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initialization for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: the value of the MSE corresponding to the final value of w
        w: the final model returned by SGD
    """
    n, _ = tx.shape
    w = initial_w

    for n_iter in range(max_iters):
        i_rand = random.randint(0, n)
        xi = tx[i_rand, :]  # we compute the gradient with respect to this random point
        e = y[i_rand] - xi.dot(w)  # the error, just a scalar since minibatch size is 1
        grad = (xi.T) * (-e / n)
        w = w - grad * gamma
    e = y - tx.dot(
        w
    )  # we only need to compute the loss for the final value of w, but to do so we must refresh the error with the final value of w
    loss = e.T.dot(e) / (2 * n)

    return w, loss


def least_squares(y, tx):
    """implement least squares regression.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)

    Returns:
        w: optimal weights obtained via normal equations using pseudo-inverse matrix
        loss: the value of the MSE corresponding to the final value of w

    """
    n, _ = tx.shape

    pinv = np.linalg.pinv(
        tx
    )  # w = np.linalg.solve(tx.T.dot(tx),tx.T.dot(y)) would not work if X^T.X is not full rank, and lstsq was not authorized, so we used the pseudo-inverse based on SVD decomposition
    w = pinv.dot(y)
    e = y - tx.dot(w)
    loss = e.T.dot(e) / (2 * n)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        lambda_: scalar.

    Returns:
        w: weights obtained via normal equations
        loss: the value of the MSE corresponding to the final value of w

    """
    n, d = tx.shape
    new_lam = lambda_ * 2 * n
    I = np.eye(d)
    w = np.linalg.solve(tx.T.dot(tx) + I * new_lam, tx.T.dot(y))

    e = y - tx.dot(w)
    loss = e.T.dot(e) / (2 * n)
    return w, loss


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def calculate_nll(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, ) Unlike lab5, here all vectors are (n,) and not (n,1) ; values are 0 or 1
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a non-negative loss
    """
    n = y.shape[0]
    loss = 0
    for i in range(n):
        loss += y[i] * np.log(sigmoid(tx[i, :].T.dot(w))) + (1 - y[i]) * np.log(
            1 - sigmoid(tx[i, :].T.dot(w))
        )
    return -loss / n


def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, ) with values 0 or 1
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a vector of shape (D, )

    """
    n = y.shape[0]
    grad = tx.T.dot(sigmoid(tx.dot(w)) - y) / n
    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic Regression using Gradient Descent.

    Args:
        y: numpy array of shape=(N, ), with values 0 or 1
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initialization for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: the value of the negative log-likelihood corresponding to the final value of w
        w: the final model returned by GD
    """
    n, _ = tx.shape
    w = initial_w

    for n_iter in range(max_iters):
        grad = calculate_gradient(y, tx, w)
        w = w - grad * gamma
    loss = calculate_nll(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized Logistic Regression using Gradient Descent.

    Args:
        y: numpy array of shape=(N, ), with values 0 or 1
        tx: numpy array of shape=(N,D)
        lambda_: scalar denoting the strength of regularization
        initial_w: numpy array of shape=(D, ). The initialization for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: the value of the negative log-likelihood corresponding to the final value of w
        w: the final model returned by GD
    """
    n, _ = tx.shape
    w = initial_w

    for n_iter in range(max_iters):
        grad = (
            calculate_gradient(y, tx, w) + w * 2 * lambda_
        )  # the only thing that changes in the regularized method is the value of the gradient
        w = w - grad * gamma
    loss = calculate_nll(y, tx, w)

    return w, loss



def lasso_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized Logistic Regression using Gradient Descent.

    Args:
        y: numpy array of shape=(N, ), with values 0 or 1
        tx: numpy array of shape=(N,D)
        lambda_: scalar denoting the strength of regularization
        initial_w: numpy array of shape=(D, ). The initialization for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: the value of the negative log-likelihood corresponding to the final value of w
        w: the final model returned by GD
    """
    n, _ = tx.shape
    w = initial_w

    for n_iter in range(max_iters):
        grad = ( # Use the l1 regularization
            calculate_gradient(y, tx, w) + lambda_ * np.sign(w)
        )  # the only thing that changes in the regularized method is the value of the gradient
        w = w - grad * gamma
    loss = calculate_nll(y, tx, w)

    return w, loss

def training(y, tx, lambda_, initial_w, max_iters, gamma, X_val, y_val):
    """Regularized Logistic Regression using Gradient Descent.

    Args:
        y: numpy array of shape=(N, ), with values 0 or 1
        tx: numpy array of shape=(N,D)
        lambda_: scalar denoting the strength of regularization
        initial_w: numpy array of shape=(D, ). The initialization for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: the value of the negative log-likelihood corresponding to the final value of w
        w: the final model returned by GD
    """
    n, _ = tx.shape
    w = initial_w
    losses = []
    val_losses = []

    for n_iter in range(max_iters):
        grad = ( # Use the l1 regularization and the l2 regularization
            calculate_gradient(y, tx, w) + 2 * lambda_ * w
        )  # the only thing that changes in the regularized method is the value of the gradient
        w = w - grad * gamma
        if n_iter % (max_iters/5) == 0:
            losses.append(calculate_nll(y, tx, w))
            val_losses.append(calculate_nll(y_val, X_val, w))

    return w, losses, val_losses


def split_data(x, y, ratio, seed=1):
    np.random.seed(seed)
    n = len(y)
    indices = np.random.permutation(n)
    n_train = int(np.floor(n * ratio))
    indices_train = indices[:n_train]
    indices_test = indices[n_train:]
    return x[indices_train], x[indices_test], y[indices_train], y[indices_test]

def compute_accuracy(y, y_pred):
    return np.sum(y == y_pred) / len(y)

def compute_f1_score(y, y_pred):
    # True positives: predicted 1 and actual 1
    tp = np.sum((y == 1) & (y_pred == 1))
    # False positives: predicted 1 but actual -1
    fp = np.sum((y == -1) & (y_pred == 1))
    # False negatives: predicted -1 but actual 1
    fn = np.sum((y == 1) & (y_pred == -1))
    
    # Precision and recall, with safety check for division by zero
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    
    return 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0