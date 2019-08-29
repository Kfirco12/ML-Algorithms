import numpy as np
import sys


class NeuralNetwork(object):
    def __init__(self, hidden1=50, hidden2=10, learning_rate=1e-3, num_iter=100):

        # h1 for size of first hidden layer
        self.h1 = hidden1

        # Number of nodes in output (10 because we have 10 different digits)
        self.h2 = hidden2

        # Learning rate
        self.learning_rate = learning_rate

        # number iteration of calculate weights
        self.num_iter = num_iter

        # Loss indicate
        self.err = []

    def initialize_weights(self, l0, l1):
        # send in previous layer size L0 and next layer size L1 and returns small random weights
        w = np.random.randn(l0, l1) * 0.01
        b = np.zeros((1, l1))
        return w, b

    def forward_prop(self, X):

        W2 = self.parameters['W2']
        W1 = self.parameters['W1']
        b2 = self.parameters['b2']
        b1 = self.parameters['b1']

        # forward prop
        a0 = X
        z1 = np.dot(a0, W1) + b1

        # apply nonlinearity (ReLU)
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, W2) + b2

        # softmax on the last layer
        scores = z2
        exp_scores = np.exp(scores)

        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # cache values from forward pass to use for backward pass
        cache = {'a0': X, 'probs': probs, 'a1': a1}
        return cache, probs

    def compute_cost(self, y, probs):
        W2 = self.parameters['W2']
        W1 = self.parameters['W1']

        loss = -np.log(probs[np.arange(self.m), y])
        avg_loss = np.sum(loss) / self.m
        return avg_loss

    def backward_prop(self, cache, y):

        # Unpack from parameters
        W2 = self.parameters['W2']
        W1 = self.parameters['W1']
        b2 = self.parameters['b2']
        b1 = self.parameters['b1']

        # Unpack from forward prop
        a0 = cache['a0']
        a1 = cache['a1']
        probs = cache['probs']

        # Start backward propogation
        dz2 = probs
        dz2[np.arange(self.m), y] -= 1
        dz2 /= self.m

        # backprop through values dW2 and db2
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Back to the (only) hidden layer in this case
        dz1 = np.dot(dz2, W2.T)
        dz1 = dz1 * (a1 > 0)

        # backprop through values dw1, db1
        dW1 = np.dot(a0.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {'dW1': dW1, 'dW2': dW2, 'db1': db1, 'db2': db2}

        return grads

    def update_parameters(self, grads):

        W2 = self.parameters['W2']
        W1 = self.parameters['W1']
        b2 = self.parameters['b2']
        b1 = self.parameters['b1']

        dW2 = grads['dW2']
        dW1 = grads['dW1']
        db2 = grads['db2']
        db1 = grads['db1']

        W2 -= self.learning_rate * dW2
        W1 -= self.learning_rate * dW1

        b2 -= self.learning_rate * db2
        b1 -= self.learning_rate * db1

        self.parameters = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
        return None

    def train(self, X, y):

        # m for training examples (or how many images we have)
        self.m = X.shape[0]

        # n for number of features in our input data (32 x 32 = 784)
        self.n = X.shape[1]

        # initialize our weights
        W1, b1 = self.initialize_weights(self.n, self.h1)
        W2, b2 = self.initialize_weights(self.h1, self.h2)

        # pack into dictionary weights
        self.parameters = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

        for it in range(self.num_iter + 1):
            # forward propogation
            cache, probs = self.forward_prop(X)

            # calculate cost
            loss = self.compute_cost(y, probs)
            self.err.append(loss)

            # back prop
            grads = self.backward_prop(cache, y)

            # update param
            self.update_parameters(grads)
        return self

    def predict(self, X, y=None):
        _, probs = self.forward_prop(X)

        return np.argsort(probs, axis=1)[:, -1]


def cross_val(model, x, y, folds_num=5):
    ''' Cross validation method.
    
    Args:
        model - model object (for generic implementation).
        x, y - observations and labels.
        folds_num - number of folds in the cross validation.
    
    Retruns:
        an array of accuracies of each fold.
    '''

    validation_data = all_data = np.c_[(x, y)]

    accuracy_validation = []
    num_obv = validation_data.shape[0] // folds_num

    for k in range(folds_num):
        data = np.roll(validation_data, k * num_obv, axis=0)
        val_train, val_test = data[: 4 * num_obv, :-1], data[4 * num_obv:, :-1]
        val_target_train, val_target_test = data[:4 * num_obv, -1], data[4 * num_obv:, -1]

        model.train(val_train, val_target_train)
        y_val_pred = model.predict(val_test)

        acc = np.mean(val_target_test == y_val_pred)
        accuracy_validation.append(acc)

    return accuracy_validation

def write_to_csv(file_name, pred):

    ''' Prints a list to a file

    Args:
        file_name -- the file you want to write to.
        pred -- a list that its content will be written to the file.

    '''
    with open(file_name, 'w')as file:
        for i in range(pred.size - 1):
            file.write("{0}\n".format(pred[i]))
        file.write(str(pred[-1]))

if __name__ == '__main__':

    # Read paths.
    if len(sys.argv) == 4:
        path_train = sys.argv[1]
        path_target = sys.argv[2]
        path_test = sys.argv[3]
    else:
        path_train = r"train_x"
        path_target = r"train_y"
        path_test = r"test_x"

    # Load data
    train_data = np.loadtxt(path_train, delimiter=" ")
    target_data = np.loadtxt(path_target, delimiter=" ")
    test_data = np.loadtxt(path_test, delimiter=" ")

    X = train_data.astype(np.uint0)
    y = target_data.astype(np.uint0)

    # create NN object
    NN = NeuralNetwork(learning_rate=1e-3, num_iter=2000, hidden1=256)

    # Cross validation
    cross_val_acc = cross_val(NN, X, y)

    # Predict
    y_pred = NN.predict(test_data)

    # Print prediction
    write_to_csv('test_y', y_pred)