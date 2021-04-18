# A project investigating resyllabification using deep learning with a modified CTC loss
- The model is a cnn-rnn network inspired by [DeepSpeech](https://arxiv.org/pdf/1512.02595.pdf), [AssemblyAI](https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO) and [ResNet](https://arxiv.org/pdf/1603.05027.pdf).
- The CTC loss function is modified with a regularisation term which penalises large or small probability and rewards large entropy, thereby encouraging exploration and avoids peaky distribution. The modified loss function (enCTC) is adopted from [Liu et al., 2018](https://papers.nips.cc/paper/2018/hash/e44fea3bec53bcea3b7513ccef5857ac-Abstract.html), and the code for Liu et al., 2018 is available on [github](https://github.com/liuhu-bigeye/enctc.crnn).