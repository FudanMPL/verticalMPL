## Load and split mnist training data
You can download the mnist data from this URL: MNIST Dataset: http://yann.lecun.com/exdb/mnist/  and decompress the downloaded file and move them to this directory.

Then split the dataset vertically. Here we set the number of features for Party 0 to 400, the number of features for Party 1 to 384

There should be four files:
- mnist_train.csv
- mnist_train_party0.csv
- mnist_train_party1.csv
- mnist_test.csv