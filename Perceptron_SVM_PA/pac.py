import numpy as np

class ModelNotFittedError(Exception):
    def __init__(self, text):
        ModelNotFittedError.text = text


class PAC(object):
    """ Passive Aggressive Classifier.

      Parameters
      ----------

      n_iter : int
          Passes (epochs) over the training set.

      Attributes
      ----------
      errors_ : list
          Number of misclassification in every epoch.
      w  : Id-array
          Weights after fitting.
      __is_fitted : boll
          Flag showing execution of train
    """
    def __init__(self, n_iter):
        self.n_iter = n_iter
        self.__is_fitted = False
        self.errors_ = []
        self.w = None

    def fit(self, X, y):
        """
        Fit method for training data.

        Parameters
        ----------

        X : {array-like}, shape = [n_samples, n_features]
        Training vectors, where 'n_samples', is the number
        of samples and 'n_features' is the number of features.
        y : {array-like}, shape = [n_samples] Target value

        Returns
        -------
        self : object

        """

        self.__is_fitted = True
        num_train, dim = X.shape
        num_classes = 1
        if (len(y) != 0):
            num_classes = int(np.max(y) + 1)
        y = y.astype(np.int8)

        # different initialization of weights
        # if self.w is None:
        #     self.w = 0.001 * np.random.rand(num_classes + 1, dim)

        # Add initialization of bias
        self.w = np.zeros((num_classes + 1, dim))

        for _ in range(self.n_iter):
            errors = 0.0
            X, y = self.shuffle_data(X, y)
            for xi, y_i in zip(X, y):
                xi = xi[np.newaxis,:]
                y_hat = self.predict(xi)
                # update weights
                if y_i != y_hat:
                    tau = self.loss(xi, y_i, y_hat) / (2 * (np.linalg.norm(xi)) ** 2)

                    # temp must Change!!!!!!!!!!!!!
                    self.u = self.w[y_i, np.newaxis]
                    self.c = self.w[y_hat, np.newaxis]

                    self.u += tau * xi
                    self.c -= tau * xi
                    self.w[y_i, :] = self.u
                    self.w[y_hat, :] = self.c
                    errors += 1
            self.errors_.append(errors)
        return self

        # hinge loss

    def loss(self, example, true_label_idx, predicted_label_idx):
        predicted__label_proj = np.dot(self.w[predicted_label_idx], example.T)
        true_label_proj = np.dot(self.w[true_label_idx], example.T)
        summation = predicted__label_proj + true_label_proj
        return max(0, 1 - summation)

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

