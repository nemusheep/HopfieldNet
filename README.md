# Hopfield Network
Hopfield network is the recurrent neural network which realize associated memory.
This network learns the self weight W from training data only. And the rule of learning(Hebbian) is presented as below, except diagonal elements. x(q) is the memorizing pattern of index q, and Q is the number of memorizing patterns.
```math
W = \sum_q^Q x^{(q)}(x^{q})^T \frac{1}{Q}
```
And diagonal elements is constantly zero.
Not complecated method is here. Not need back propagation. This method is very easy for model to learn.

### Description, run env
Mainly use Hopfield.py, and the image created from this file is collected into result or fig directory. Data used as memorizing patterns is collected in data directory.
Modules I used in this are matplotlib and numpy. Version is shown below.
```
Python 3.11.4
numpy 1.25.2
matplotlib 3.7.2
```
To run
```
$ python3 Hopfield.py
```

### Result memo : false attractor
In data lines shown as below,
<div style='display: flex;' >
<img src='https://github.com/nemusheep/HopfieldNet/blob/main/data/lines_0.png' width='30%' height='auto'>
<img src='https://github.com/nemusheep/HopfieldNet/blob/main/data/lines_1.png' width='30%' height='auto'>
<img src='https://github.com/nemusheep/HopfieldNet/blob/main/data/lines_2.png' width='30%' height='auto'>
</div>
these train data often leads the unexpected result:
<img src='https://github.com/nemusheep/HopfieldNet/blob/main/fig/false_attractor.png'>
Finally this model converges to the state which has only value 1 on the most overlapping point.
This is false attractor.

### cf. Moore-Penrose pseudo inverse matrix model
Instead of using learning model of self-correlation(Hebbian), the method using Moore-Penrose Pseudo Inverse Matrix is the effective way for eliminating or reducing false attractors and inproving the accuracy of association.
This method is learning connecting(synaptic) weight as below. X is the (25, Q) size matrix generated from aranging col vectors of each memorizing pattern.
```math
W = X (X^T X)^{-1} X^T
```
W learned by this method has good characteristics for each memorizing pattern x_i that
```math
W x_i = x_i
```
This means the memorizing patterns are definitely attractor of the potencial space, so without adding noise, the accuracy of association of this model is just 100%.
The graphs shown below are the fluctuation of accuracy dependent on noise probability, left side using self-correlation, right side using Pseudo Inverse Matrix. They obviously describe latter is more good strategy for associating.
<div style='display: flex;' >
<img src='https://github.com/nemusheep/HopfieldNet/blob/main/fig/acc.png' width='40%' height='auto'>
<img src='https://github.com/nemusheep/HopfieldNet/blob/main/fig/accPIM.png' width='40%' height='auto'>
</div>

### ref
- https://qiita.com/grouse324/items/95e93f8c3e4679ecf39a
- https://rishida.hatenablog.com/entry/2014/03/03/174331
- https://www.cis.twcu.ac.jp/~asakawa/chiba2002/lect6-mutual/mutual.html
- https://www.jstage.jst.go.jp/article/jnns/3/4/3_4_141/_pdf