import numpy as np


class ModelNotFittedError(Exception):
    def __init__(self, text):
        ModelNotFittedError.text = text


class Perceptron(object):
    """Perceptron classifier.

      Parameters
      ----------
      eta : float
          Learning rate (between 0.0 and 1.0).
      n_iter : int
          Passes (epochs) over the training set.

      Attributes
      ----------
      w  : Id-array
          Weights after fitting.
      __is_fitted : boll
          Flag showing execution of train
    """
    def __init__(self, eta, n_iter, initialize_w = True):
        if (eta <= 0) or (eta >= 1):
            raise ValueError("Eta must be greater than 0 and less than 1")
        self.eta = eta
        self.n_iter = n_iter
        self.__is_fitted = False
        self.w = None
        self.initialize_w = initialize_w

    def fit(self, X, y):
        """
        Fit method for training data.

        Parameters
        ----------

        X : {array-like}, shape = [n_samples, n_features]
        Training vectors, where 'n_samples', is the number
        of samples and 'n_features' is the number of features.
        y : {array-like}, shape = [n_samples] Target value
        initialize_w : flag of initialization of weights
        Returns
        -------
        self : object

        """

        self.__is_fitted = True
        num_train, dim = X.shape
        num_classes = 1
        if(len(y) != 0):
            num_classes = int(np.max(y) + 1)
        y = y.astype(np.int8)

        self.w = np.zeros((num_classes + 1, dim))

        # different initialization of weights
        if self.initialize_w:
            self.w = 0.1 * np.random.rand(num_classes + 1, dim)

        for _ in range(self.n_iter):

            X, y = self.shuffle_data(X, y)
            for x, y_i in zip(X, y):
                x = x[np.newaxis,:]
                y_hat = self.predict(x)
                # update weights
                if y_i != y_hat:
                    self.w[y_i, :] = self.w[y_i, :] + self.eta * x
                    self.w[y_hat, :] = self.w[y_hat, :] - self.eta * x
                    self.w[0, :] = self.eta * x
        return self

    def gradient(self, X):
        """ Calculation of Gradient
        param X: unit observation or matrix of observations
        return: vector with shape (1, 3)
        """
        return np.dot(X, self.w.T)

    def predict(self, X, y=None):
        """Return class label after unit step function.
        """
        if not self.__is_fitted:
            raise ModelNotFittedError('Model is not fitted.')
        return np.argmax(self.gradient(X), axis=1)

    def shuffle_data(self, features, target):
        """
        Given a standard training set D of size n,
        bagging generates m new training sets D_{i} ,each of size n′,
        by sampling from D uniformly and with replacement.
        By sampling with replacement, some observations
        may be repeated in each D_{i}.
        If n′=n, then for large n the set D_{i}
        D_{i} is expected to have the fraction (1 - 1/e) (≈63.2%)
        of the unique examples of D, the rest being duplicates.
        This kind of sample is known as a bootstrap sample.

        link to documentation: https://en.wikipedia.org/wiki/Bootstrap_aggregating

        """

        zipped_data = list(zip(features, target))

        np.random.shuffle(zipped_data)
        return zip(*zipped_data)

