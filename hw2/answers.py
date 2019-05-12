r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0.1, 0.02, 0.1
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 0.02, 0.005, 0.0001, 0.001

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0.1, 0.01
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1. We expected that as we increase the dropout parameter, the likelihood of over-fitting will decrease. 
We assume to have this behavior because as we drop more neurons, we decrease the number of parameters that participate 
 in the model, and by restricting the model we are making it more general. That prevents it from getting to over-fitting situation.  
 That is shown in the graphs we got for dropout=0.4 comparing to dropout=0:  on the no-dropout graph, we can clearly see 
 that the model is over-fitting, but for dropout=0.4, this doesn't happen.
We can conclude from it that sometimes making the model "less sophisticated" (=less parameters to tune and learn)
can actually improve the performance.

2. The best dropout value was 0.4.  we can observe that as the train loss decreased, the test loss also decreased.
that's because this model ability of generalization was getting better, as we increased the dropout parameter from
0 to 0.4.
Later on, for the value of 0.8 the performance was getting worse. that's because we lost to much information when
we drop too many neurons.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""

Yes, it is possible. the accuracy is based on the number of correct classifications, while the loss calculates the exact 
"distance" from the correct classification.  For example, there can be a situation when we have one sample that we
succeeded on increasing the probability of the grund-truth label (so number of correct classification increased, and
therefore the accuracy increased), but on the other examples, we decreased the probability for the right labels, and
therefore the loss also increased.  This is a small example (only 2 samples), but this behavior can happen for larger 
data sets as well.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

1. Comparing depth=2 to depth=4, the accuracy increased as we assumed, both for K=32 and K=64.
As we fo deeper (for L=4) we allow the model to check a larger hypotheses set (comparing to L=2), because we have two more
layers (and their parameters) to fix.  and that's why we can come closer to our train-data and fit it better.
In fact, in the L=4, K=64 graph we can observe that the model is slightly over-fitting, when is gets to 80 percent
train accuracy, but only 70 percent on the test accuracy.
later on, when fitting the model on depth=8 and depth=16, we got very bad results. the model couldn't train in these depths.  
we can think of some reasons for that. (see question 1.2).
The best results were achieved in depth=4.  We believe that we might get better results in deeper depth, 
because the general pattern we can observe is that "deeper is better", but we can't examine it in this experiment, because for 
that we have to handle the issue which causes the model to be not trainable in deeper depths.

2. for L=8,16 the network wasn't trainable, for all values of K.
As the network becomes deeper, each layer uses its previous layer's output, and that causes the layers to use
inputs which are not-scaled: we can have some inputs from 0 to 1, and some from 1 to 1000. that can cause what is known
as covariate shift.  Covariate shift refers to the change in the input distribution to a learning system. 
as we go deeper in our network, the input to each layer is affected by parameters in all the input layers. So even small 
changes to the network get amplified down the network. This leads to change in the input distribution to internal layers 
of the deep network and is known as internal covariate shift.
To overcome this, we can use batch normalization.  batch normalization makes sure that there’s no activation that’s gone 
really high or really low. this reduces internal covariant shift (and also allow us to use higher learning rates).
Another solution is to use dropout layers.  when increasing the depth, we allow the model to fit the train-data more precisely.
We can observe over-fitting in depth=4, as we described earlier.  
To overcome this, we can use dropout layers. 


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


for L=2: the best number of filters per layer is 128, although we got very similar results for K=64.  
for K=64 we obviously got same results like experiment 1.1.
for L=4: the best number of filters per layer is 128.  we got better results from experiment 1.1. 
we can still observe an over-fitting behavior for this value of L, like we saw in experiment 1.1.
for L=8: the model wasn't trainable, like it was in experiment 1.1.  This time, we can see the trend of the loss 
function, because we zoom-in. But the accuracy graph is still looks constant.
comparing to experiment 1.1, we found better configuration for the L and K parameter, and got higher test accuracy.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**

for depth=3,4, for all values of K, the model wasn't trainable, like we have seen before. 
We got the highest train loss, and the accuracy was the worst.
comparing depth=1 and depth=2, we can see small improvement with the first value of depth, even though it slightly 
over-fitting at the end.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**

1.  first, we added batch-normalization layer, after every convolution layer.  that allow us to train the model in 
higher depths. 
We also added dropout layers, after every pooling layer, and one more dropout layer at the end before the last linear layer.
We have done this in order to deal with over-fitting.  we tuned the parameter p in the dropout layer and fix it on 0.3 
for the layers after the pooling, and 0.2 for the last dropout layer.

2.  first of all, we notice the test acuuracy grows from around 70 to over 80 percent, and the loss also decreased 
dramatically, that's although we made only some small modifications to out model. 
We can also observe that out model doesn't over-fits like it did before, like we showed in experiment 1.
the most effective thing was to add batch-normaliztion. then, we could fo deeper and the model could still be trained,
unlike what happen in experiment 1. 
the best value for depth was L=2, similar to what we saw in experiment 1.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
