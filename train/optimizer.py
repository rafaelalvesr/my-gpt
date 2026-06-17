"""
    This module implements optimization algorithms for training the model.
    The optimizers implemented are:
    - Stochastic Gradient Descent (SGD) with learning rate decay
    - AdamW with learning rate decay and weight decay
    Also includes a learning rate scheduler based on cosine annealing and gradient clipping to prevent exploding gradients.

    FUTURE: update classes with inheritance from PyTorch's optim.Optimizer and implement more optimizers like RMSProp, Adagrad, etc.
"""
import math
import numpy as np
from collections.abc import Callable
from typing import Optional
from .tensorvalue import TensorValue

class SGD():
    """
    Stochastic Gradient Descent (SGD) optimizer with learning rate decay.
     - params: list of parameters to optimize
     - lr: learning rate
    """
    def __init__(self, params: list[TensorValue] , lr=1e-3):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.params = params
        self.lr = lr
        self.state = {}
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss. 
                This is used for optimizers that require multiple evaluations of the model, such as line search methods.
        """
        loss = None if closure is None else closure()
        t = self.state.get('t', 0) + 1
        self.state['t'] = t
        step_size = self.lr / math.sqrt(t)

        for param in self.params:
            param.data -= step_size * param.grad


        return loss
    
    def save_dict(self):
        """
        Returns a dictionary containing the states of the optimizer.
        """
        return {'t': self.state.get('t', 0)}
    
    def load_dict(self, state_dict):
        self.state['t'] = int(state_dict['t'])
    
    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        for param in self.params:
            param.grad.fill(0.0)  # Set gradients to zero in-place


class AdamW():
    """
    AdamW optimizer with learning rate decay and weight decay.
     - params: list of parameters to optimize
     """
    def __init__(self, params: list[TensorValue], lr: float=1e-3, beta1: float=0.9, beta2: float=0.999, weight_decay: float=0.001) -> None:
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.state = {}
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss. 
                This is used for optimizers that require multiple evaluations of the model, such as line search methods.
        """
        loss = None if closure is None else closure()
        for param in self.params:
            if param.grad is None:
                continue
            state = self.state.setdefault(param, {})
            if 'm' not in state:
                state['m'] = np.zeros_like(param.data)
            if 'v' not in state:
                state['v'] = np.zeros_like(param.data)
            if 't' not in state:
                state['t'] = 0
            #update first moment and second moment
            m = state['m'] * self.beta1 + (1 - self.beta1) * param.grad
            v = state['v'] * self.beta2 + (1 - self.beta2) * (param.grad ** 2)
            state['m'] = m
            state['v'] = v
            #compute bias-corrected first moment and second moment
            t = state['t'] + 1
            state['t'] = t
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            
            #update in-place the parameters using the AdamW update rule, with decay over time and weight decay
            param.data -= self.lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + self.weight_decay * param.data)

        return loss
    
    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        for param in self.params:
            param.grad.fill(0.0)  # Set gradients to zero in-place

    def save_dict(self):
        """
        Returns a dictionary containing the states of the optimizer.
        """
        pos = {id(p): i for i, p in enumerate(self.params)}
        out = {}
        for p,s in self.state.items():
            i = pos[id(p)]
            out[f"m_{i}"] = s['m']
            out[f"v_{i}"] = s['v']
            out[f"t_{i}"] = np.asarray(s['t'])
        return out
    
    def load_dict(self, state_dict):
        for i,p in enumerate(self.params):
            if f"m_{i}" in state_dict:
                self.state[p] = {
                    'm': state_dict[f"m_{i}"],
                    'v': state_dict[f"v_{i}"],
                    't': int(state_dict[f"t_{i}"])
                }
    
# Learning rate scheduler based on cosine annealing
def learning_rate_schedule(step, T_w=20, T_c=1000, alpha_max=0.1, alpha_min=0.0001):
    """
    Cosine Annealing Learning Rate Scheduler.
    Args:
        step (int): Current step number.
        T_w (int): Number of warm-up steps.
        T_c (int): Total number of steps for cosine annealing.
        alpha_max (float): Maximum learning rate.
        alpha_min (float): Minimum learning rate.
    Returns:
        float: Learning rate for the current step.
    """
    step+=1 #prevent step to be zero
    if step < T_w:
        lr = alpha_max * step / T_w
    elif step <= T_c:
        lr = alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * (step - T_w) / (T_c - T_w)))
    else:
        lr = alpha_min
    return lr

#Gradient clipping function to prevent exploding gradients
def clip_gradients(params, max_norm=1.0):
    """
    Clips the gradients of the parameters to prevent exploding gradients.
    Args:
        params (list): List of parameters with gradients to be clipped.
        max_norm (float): Maximum allowed norm of the gradients.
    """
    total_norm = 0.0
    for param in params:
        if param.grad is not None:
            total_norm += np.sum(param.grad ** 2)
    total_norm = math.sqrt(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in params:
            if param.grad is not None:
                param.grad *= clip_coef