import abc
import torch
from torch import Tensor
import numpy as np
import math

class Optimizer(abc.ABC):
    """
    Base class for optimizers.
    """
    def __init__(self, params):
        """
        :param params: A sequence of model parameters to optimize. Can be a
        list of (param,grad) tuples as returned by the Blocks, or a list of
        pytorch tensors in which case the grad will be taken from them.
        """
        assert isinstance(params, list) or isinstance(params, tuple)
        self._params = params

    @property
    def params(self):
        """
        :return: A sequence of parameter tuples, each tuple containing
        (param_data, param_grad). The data should be updated in-place
        according to the grad.
        """
        returned_params = []
        for x in self._params:
            if isinstance(x, Tensor):
                p = x.data
                dp = x.grad.data if x.grad is not None else None
                returned_params.append((p, dp))
            elif isinstance(x, tuple) and len(x) == 2:
                returned_params.append(x)
            else:
                raise TypeError(f"Unexpected parameter type for parameter {x}")

        return returned_params

    def zero_grad(self):
        """
        Sets the gradient of the optimized parameters to zero (in place).
        """
        for p, dp in self.params:
            dp.zero_()

    @abc.abstractmethod
    def step(self):
        """
        Updates all the registered parameter values based on their gradients.
        """
        raise NotImplementedError()


class VanillaSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            # Update the gradient according to regularization and then
            # update the parameters tensor.
            dp += self.reg*p
            p -= self.learn_rate*dp


class MomentumSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, momentum=0.9):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param momentum: Momentum factor
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.momentum = momentum

        # TODO: Add your own initializations as needed.

        #self.V = np.zeros(len(params))
        self.V = [torch.zeros_like(p[1]) for p in self.params]

    def step(self):
        '''
        for (p, dp), v in zip(self.params, self.V):
            if dp is None:
                continue
            dp += self.reg * p
            v = self.momentum*v - self.learn_rate*dp #TODO does it actually change the element in self.V?
            p += v
        '''

        for (p, dp), idx_param in zip(self.params, range(len(self.params))):
            if dp is None:
                 continue

            # TODO: Implement the optimizer step.
            # update the parameters tensor based on the velocity. Don't forget
            # to include the regularization term.

            dp += self.reg * p
            self.V[idx_param] = self.momentum * self.V[idx_param] - self.learn_rate*dp
            p += self.V[idx_param]


class RMSProp(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, decay=0.99, eps=1e-8):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param decay: Gradient exponential decay factor
        :param eps: Constant to add to gradient sum for numerical stability
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.decay = decay
        self.eps = eps

        # TODO: Add your own initializations as needed.

        #self.R = np.zeros(len(params))
        self.R = [torch.zeros_like(p[1]) for p in self.params]

    def step(self):

        '''
         for (p, dp), r in zip(self.params, self.R):
            if dp is None:
                continue
         dp += self.reg * p
            r = self.decay * r + (1 - self.decay) * dp**2
            p -= (self.learn_rate/math.sqrt(r + self.eps)) * dp

        '''

        for (p, dp), idx_param in zip(self.params, range(len(self.params))):
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            # Create a per-parameter learning rate based on a decaying moving
            # average of it's previous gradients. Use it to update the
            # parameters tensor.

            dp += self.reg * p
            self.R[idx_param] = self.decay * self.R[idx_param] + (1 - self.decay) *  (dp**2)
            p -= (self.learn_rate/torch.sqrt(self.R[idx_param] + self.eps)) * dp
