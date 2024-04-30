## Hopfield Network
This repository is mainly for an assignment from my university.
Hopfield network is the recurrent neural network which realize associated memory.
This network learns the self weight from training data only. And the rule of learning(Hebbian) is presented as below.
$$
W = \sum_q^Q x^{(q)}(x^{q})^T \frac{1}{Q}
$$
Not complecated method is here. Not need back propagation. This method is very easy for model to learn.

### Description, run env
Mainly use Hopfield.py, and the image created from this file is collected into result directory.
Module is already everyone knows, matplotlib and numpy.
'''
Python 3.11.4
numpy 1.25.2
matplotlib 3.7.2
'''
to run
'''
python3 Hopfield.py
'''

### result memo
In data lines shown as below,
<div style='display: flex;' >
<img src='https://github.com/nemusheep/HopfieldNet/blob/main/data/lines_0.png'>
<img src='https://github.com/nemusheep/HopfieldNet/blob/main/data/lines_1.png'>
<img src='https://github.com/nemusheep/HopfieldNet/blob/main/data/lines_2.png'>
</div>
these train data often leads the unexpected result:
<img src='https://github.com/nemusheep/HopfieldNet/blob/main/fig/false_attractor.png'>
Finally this model converges to the state which has only value 1 on the most overlapping point.
I think this is false attractor.