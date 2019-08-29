Here I implemented train a neural network, from scratch, on a data-set called  
“Fashion-MNIST” which contains 10 different categories of clothing.  

**Architecture:**
* 2 hidden layers:
	 - First layer contains 50 nodes.
	 - Second layer contains 10 nodes.

* Learning rate is 1e-3.
* The model is minimize the Negative Log Likelihood (NLL).
* Activation functions is ReLU.

**Input:**
The “Fashion-MNIST” is provided here as split data (train_x and train_y files) for training the model. and a test_x file for evaluation.
For running that model, extract the mnist_files to the folder witch contains the NN,py file, or give those files paths as parameters to NN.py file.

**Output:**
An output file, called test_y with the predictions of the model.