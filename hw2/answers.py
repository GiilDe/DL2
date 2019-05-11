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

1. there were some some behaviors which we expected, and some whose we didn't. 
We expected that as we increase the dropout parameter, the train loss will increase, and that is shown in the graphs we got
for dropout=0.4 comparing to dropout=0. we assume to have this behavior because higher value of dropout decrease the 
over-fitting, by restricting the model and making it more general, and that prevents it from getting to over-fitting situation.
On the other hand, we observe different behavior with dropout=0.8, which we didn't expected. In fact, we got better results 
that we assumed with that value of dropout. 
We can conclude from it that sometimes making the model "less sofisticated" can actually improve the performance.

2. The best dropout value was 0.4.  we can observe that as the train loss was growing, the test loss was increasing.
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
**Your answer:**

Yes, it is possible. the accuracy is based on the number of correct classifications, while the loss calculates the exact 
"distance" from the correct classification.  For example, there can be a situation when we have one sample that we
succeeded on increasing the probability of the ground-truth label (so number of correct classification increased, and
therefore the accuracy increased), but on the other examples, we decreased the probability for the right labels, and
therefore the loss also increased.


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

1.  If we compare depth=2 to depth=4, the accuracy increased as we assumed.  later on, when fitting the model on
depth=8 and depth=16, we got very bad results. the model couldn't train in these depths.  we can think of some 
reasons for that. (see question 1.2).
the best result were achieved in depth=4.  

2. for L=8,16 the network wasn't trainable. this may be because we didn't use batch-normalization in this experiment. 
another reason can be over-fitting.  when increasing the depth, we allow the model to fit the train-data more precisely.
to overcome this, we can use dropout layers. 


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


for L=2: the best number of filters per layer is 128.  we got better results from experiment 1.1.
for L=4: the best number of filters per layer is 128.  we got better results from experiment 1.1.
for L=8: the model wasn't trainable, like it was in experiment 1.1.
comparing to experiment 1.1, 

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**

for depth=4, for all values of K, the model wasn't trainable. we got the highest train loss, and the accuracy was 
the worst.
comparing depth=1 and depth=2, we can see small improvement with the last value of depth, especially in the test
accuracy (the train accuracy is almost equal).
for depth=3.....
the best K value 



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**

1.  first, we added batch-normalization.  that allow us to train the model in higher depths. 
We also added dropout layers, in order to deal with over-fitting.  we tuned the parameter p in the dropout layer.

2.  first of all, we notice the test acuuracy grows from around 70 to over 80 percent, although we made only some
small modifications to out model. 
the most effective thing was to add batch-normaliztion. then, we could fo deeper and the model could still be trained.
the best value for depth and dropout were


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
