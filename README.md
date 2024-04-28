## Hopfield Network
This repository is mainly for an assignment from my university.
Hopfield network is the recurrent neural network which realize associated memory.
This network learns the self weight from training data only. And the rule of learning(Hebbian) is presented as below.
$$
W = \sum_q^Q x^{(q)}(x^{q})^T \frac{1}{Q}
$$
Not complecated method is here. Not need back propagation. This method is very easy for model to learn.

### Description
Mainly use Hopfield.py, and the image created from this file is collected into fig directory.
Module is already everyone knows, matplotlib and numpy.