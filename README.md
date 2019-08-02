# monte_carlo_dropout
Uncertainty estimation in deep learning using monte carlo dropout with keras.<br>
In this sample, estimate uncertainty in CNN classification of dogs and cats images using monte carlo dropout.<br>
I examined CNN using sigmoid and softmax. Some of the results are shown below. Note that P(dog) is the probability of being a dog.<br>

![mrc](https://github.com/statsu1990/monte_carlo_dropout/blob/master/result_sigmoid/std_vs_prob_dog.png)<br>
      fig. [CNN using sigmoid] std of predicted P(dog) vs predicted P(dog) when using a dog image for input.<br>
![mrc](https://github.com/statsu1990/monte_carlo_dropout/blob/master/result_sigmoid/std_vs_prob_cat.png)<br>
      fig. [CNN using sigmoid] std of predicted P(dog) vs predicted P(dog) when using a cat image for input.<br>

![mrc](https://github.com/statsu1990/monte_carlo_dropout/blob/master/result_softmax/std_vs_prob_dog.png)<br>
      fig. [CNN using softmax] std of predicted P(dog) vs predicted P(dog) when using a dog image for input.<br>
![mrc](https://github.com/statsu1990/monte_carlo_dropout/blob/master/result_softmax/std_vs_prob_cat.png)<br>
      fig. [CNN using softmax] std of predicted P(dog) vs predicted P(dog) when using a cat image for input.<br>

The details are described in the blog below.<br>
https://st1990.hatenablog.com/entry/2019/07/31/010010<br>

MC DropoutでのDLの不確かさ評価(keras)。<br>
このサンプルでは、犬猫画像のCNNでの分類の不確かさを評価する。<br>
詳細は以下のブログ参照。<br>
https://st1990.hatenablog.com/entry/2019/07/31/010010<br>

#### dnn_uncertainty.py
- estimate uncertainty in classification of dogs and cats.<br>
犬猫分類の不確かさを評価する。<br>

#### montecarlo_dropout.py
- Convert keras model to model which uses dropout in inference.<br>
kerasモデルを推論時にdropoutを使えるモデルに変換する。<br>

#### classifier_cnn.py
- Create binary calassification CNN model with dropout and L2 regularization.<br>
dropoutとL2正則化を使った二値分類CNNを作成する。<br>

#### cifar10_data.py
- Treat cifar10 data.<br>
cifar10のデータを扱う。<br>



# monte_carlo_dropout
MC DropoutでのDLの不確かさ評価(keras)。<br>
このサンプルでは、犬猫画像のCNNでの分類の不確かさを評価する。<br>
詳細は以下のブログ参照。<br>
https://st1990.hatenablog.com/entry/2019/07/31/010010<br>

#### dnn_uncertainty.py
犬猫分類の不確かさを評価する。<br>

#### montecarlo_dropout.py
kerasモデルを推論時にdropoutを使えるモデルに変換する。<br>

#### classifier_cnn.py
dropoutとL2正則化を使った二値分類CNNを作成する。<br>

#### cifar10_data.py
cifar10のデータを扱う。<br>
