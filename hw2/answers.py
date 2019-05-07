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



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
