import numpy as np


class SVM(object):
    """
    Linear SVM classifier.

    Parameters
    ----------

    n_iter : int
        Passes (epochs) over the training set.
    reg : float
        parameter of regularization.

    learning_rate : float
        that controls how much we are adjusting
        the weights of model with respect the loss gradient.

    batch_size : int
        parameter controls the number of training samples to work
        through before the model’s internal parameters are updated.

    Attributes
    ----------
    loss_history : list
        list of loss in each epoch, progress in fit process

    W : 2D array
        Weights after fitting.
    """

    def __init__(self, learning_rate=1e-3, n_iter=100, batch_size=15, reg=0):
        self.W = None
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.reg = reg
        self.loss_history = []

    def loss(self, X, target, reg, delta=1):
        """
        to each class return loss and gradient matrix
        """
        loss = 0
        # gradient of weights matrix
        dW = np.zeros(self.W.shape)
        num_train = X.shape[0]

        scores = X.dot(self.W)
        y = target.astype(int)

        correct_class = scores[range(num_train), y]

        correct_class_scores = np.reshape(correct_class, (num_train, 1))

        margin = np.maximum(0, scores - correct_class_scores + delta)
        loss = np.sum(margin) / num_train

        # 0.5 in calculation gradient this scalar going be 1
        loss += 0.5 * reg * np.sum(self.W * self.W)

        margin[margin > 0] = 1
        margin[range(num_train), y] -= np.sum(margin, axis=1)

        # calculate gradient
        dW = X.T.dot(margin) / num_train + reg * self.W
        return loss, dW

    def fit(self, X, y, verbose=False):

        num_train, dim = X.shape
        num_classes = 1
        if (len(y) != 0):
            num_classes = int(np.max(y) + 1)

        # Initialization of weights normal distribution
        if self.W is None:
            self.W = 0.001 * np.random.rand(dim, num_classes)

        for it in range(self.n_iter):
            idx = np.random.choice(num_train, self.batch_size, replace=False)
            X_batch = X[idx, :]
            y_batch = y[idx]

            loss, grad = self.loss(X_batch, y_batch, self.reg)
            self.loss_history.append(loss)

            self.W -= self.learning_rate * grad

            # if verbose and it % 100 == 0:
            #     print('iter {} / {} : loss {}'.format(it, self.n_iter, loss))

        return self.loss_history

    def predict(self, X):
        y_pred = np.argmax(X.dot(self.W), axis=1)
        return y_pred




