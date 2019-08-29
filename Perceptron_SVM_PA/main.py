import sys
import numpy as np
from perceptron import Perceptron
from svm import SVM
from pac import PAC

# This method gets the validation data and performs k-fold cross validation. it returns an accuracy list.
def cross_val(model, validation_data, folds_num=5):

    accuracy_validation = []
    num_obv = validation_data.shape[0] // folds_num

    for k in range(folds_num):
        data = np.roll(validation_data, k * num_obv, axis=0)
        val_train, val_test = data[: 4 * num_obv, :-1], data[4 * num_obv:, :-1]
        val_target_train, val_target_test = data[:4 * num_obv, -1], data[4 * num_obv:, -1]

        model.fit(val_train, val_target_train)
        y_val_pred = model.predict(val_test)

        acc = np.mean(val_target_test == y_val_pred)
        accuracy_validation.append(acc)

    return accuracy_validation

# Min max scaler
def min_max_scaler(arr):
    scaler_arr = (arr - np.min(arr, axis=0)) / (np.max(arr, axis=0) - np.min(arr, axis=0))
    return scaler_arr


# This method gets the features path and returns a dummies matrix of the 'Sex' feature.
def map_dummies(features_path):
    # Mapping dummies
    dummies = np.loadtxt(features_path, dtype=str, usecols=0, delimiter=',').reshape(-1, 1)
    num_dummies = np.unique(dummies)
    mapping_dummies = np.zeros((dummies.shape[0], num_dummies.size), dtype=float)

    for i in range(len(dummies)):
        if dummies[i, :] == 'M':
            mapping_dummies[i, 0] = 1
        elif dummies[i, :] == 'F':
            mapping_dummies[i, 1] = 1
        else:
            mapping_dummies[i, 2] = 1

    return mapping_dummies

# Gets results matrix and print the results in the required format.
def print_predict(res_matrix):
    for i in range(res_matrix.shape[0]):
        print("perceptron: {0}, svm: {1}, pa: {2}".format(res_matrix[i][0], res_matrix[i][1], res_matrix[i][2]))

# Main
def main():
    # Load data
    if len(sys.argv) == 4:
        features_path = sys.argv[1]
        path_target = sys.argv[2]
        test_path = sys.argv[3]
    else:
        features_path = r'train_mx.txt'
        path_target = r'train_my.txt'
        test_path = r'test_mx.txt'

    # Get examples and classes from the input paths.
    data = np.loadtxt(features_path, dtype=float, usecols=(1, 2, 3, 4, 5, 6, 7), delimiter=',')
    target = np.genfromtxt(path_target, dtype=np.int8)
    test = np.loadtxt(test_path, dtype=float, usecols=(1, 2, 3, 4, 5, 6, 7), delimiter=',')

    # Map the 'sex' feature as dummies.
    mapped_dummies_features = map_dummies(features_path)
    mapped_dummies_test = map_dummies(test_path)
    data = np.append(data, mapped_dummies_features, axis=1)
    test = np.append(test, mapped_dummies_test, axis=1)

    # Stack targets and features.
    target = target[:, np.newaxis]
    all_data = np.c_[(data, target)]

    # Data scale
    # all_data = min_max_scaler(all_data)

    result = []
    # Training model with cross-validation and prediction.
    pr = Perceptron(eta=0.1, n_iter=100, initialize_w=False)
    validation_data = all_data
    _ = cross_val(pr, validation_data)
    result.append(pr.predict(test))

    svm = SVM(learning_rate=0.1, n_iter=1800, batch_size=15, reg=0.0001)
    validation_data = all_data
    _ = cross_val(svm, validation_data)
    result.append(svm.predict(test))

    pac = PAC(n_iter=200)
    validation_data = all_data
    _ = cross_val(pac, validation_data)
    result.append(pac.predict(test))

    print_predict(np.transpose(result).astype(int))


if __name__ == '__main__':
        main()