import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.spatial import distance
import init_centroids as cent


# This method gets a list of centroids and a list of pixles.
# A new pixles' list is returned.
def k_mean(cents, pixles, compressed, avg_losses):
    tmp_loss = 0
    cents_list = [[] for i in range(len(cents))]
    for i in range(len(pixles)):
        dist, cent_idx = np.inf, -1  # Init indexes.
        for j in range(len(cents)):
            tmp_dist = (distance.euclidean(pixles[i], cents[j])) ** 2  # ||v-u||^2
            # Get the values with the minimal loss.
            if tmp_dist <= dist:
                dist = tmp_dist
                cent_idx = j
        compressed[i] = cents[cent_idx]  # Saving the relevant centroid value for getting the compressed image.
        cents_list[cent_idx].append(pixles[i])  # Udpate pixle list by centroid.
        tmp_loss += dist  # For calculating the overall loss.
    avg_losses.append(tmp_loss / len(pixles))  # Average the overall loss.
    return cents_list


# Get a vector of losses and plot them on a graph
def plot_graph(loss_vec, k):
    plt.plot([i for i in range(1, len(loss_vec) + 1)], loss_vec)
    plt.xlabel('Iterations')
    plt.ylabel('Average Loss')
    plt.title('Average loss over 10 iterations for K={0}'.format(k))
    plt.show()


# plot the image
def plot_image(pixles_arr):
    # reshape new pixels for plotting.
    reshaped = compressed_img.reshape(img_size[0], img_size[1], img_size[2])
    plt.imshow(reshaped)
    plt.grid(False)
    plt.show()


# given function
def print_cents(cents, iteration):
    print('iter {0}:'.format(iteration), end=' ')
    if type(cents) == list:
        cents = np.asarray(cents)
    if len(cents.shape) == 1:
        print(' '.join(str(np.floor(100 * cents) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',
                                                                                                               ']').replace(
            ' ', ', '))
    else:
        print(' '.join(str(np.floor(100 * cents) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',
                                                                                                               ']').replace(
            ' ', ', ')[1:-1])


# update the new centroids by calculating the mean of the pixels
# and their current centroids. The function returns new list.
def centroid_update(cents_list):
    new_cents = []
    for i in range(len(cents_list)):
        tmp_arr = np.array(cents_list[i])
        tmp_mean = tmp_arr.mean(axis=0)
        new_cents.append(tmp_mean)
    return np.asarray(new_cents)


if __name__ == '__main__':
    # define variables
    pic_path = 'dog.jpeg'
    k_arr = [2, 4, 8, 16]  # list of K values.
    iterations = 10  # num. of iterations for each K.
    # Get and normalize pixels
    A = imread(pic_path)
    A = A.astype(float) / 255.
    img_size = A.shape
    pixles = A.reshape(img_size[0] * img_size[1], img_size[2])
    # copy pixels matrix for plotting compressed image.
    compressed_img = np.array(pixles)

    # Perform k_means algorithm for X iterations and print the centroids.
    for i in range(len(k_arr)):
        centroids = cent.init_centroids(pixles, k_arr[i])
        avg_losses = []
        # printing O iteration (given centroids)
        print('k={0}:'.format(k_arr[i]))
        print_cents(centroids, 0)
        for j in range(1, iterations + 1):
            # update new centroids
            cents_list = k_mean(centroids, pixles, compressed_img, avg_losses)
            centroids = centroid_update(cents_list)
            print_cents(centroids, j)
