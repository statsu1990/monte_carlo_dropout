# monte_carlo_dropout
Uncertainty estimation in deep learning using monte carlo dropout with keras.<br>

In this sample, estimate uncertainty in CNN classification of dogs and cats using monte carlo dropout.<br>
The details are described in the blog below.<br>
https://st1990.hatenablog.com/entry/2019/07/31/010010<br>

#### dnn_uncertainty.py
estimate uncertainty in classification of dogs and cats.

#### montecarlo_dropout.py
Convert keras model to model which uses dropout in inference.

#### classifier_cnn.py
Create binary calassification CNN model with dropout and L2 regularization.

#### cifar10_data.py
Treat cifar10 data.
