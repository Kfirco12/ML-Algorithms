Here I implemented and trained a neural network of voice recognition to classify words, Using Pytorch library. Unfortunately, the voice files are too big to upload, so only the code itself is available.

**Architecture:**
 - 2 convolution layers for feature extraction.
 - 2 fully connect linear layers for prediction
 - The activation function is ReLU.
 - The optimization of the model is ADAM Grad. 
 - A dropout technique with probability of 0.5 is being used between 
the feature extraction layers and the fully connection layers, to reduce over-fitting.

**Input:**
.wav files which orgenized in train, validation and test folders.
The voice files are being extracted with the provided files: 
 - data_loader_tester.py
 - gcommand_loader.py

You may need to install additional libraries in your python IDE.

**Output:**
An output file, called test_y with the predictions of the model.