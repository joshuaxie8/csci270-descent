import torch
from levels import Environment


def vanilla_grad_descent(rate, env):
    """Basic gradient descent.
    
    This is completed for you, but it won't work until you
    correctly implement a subroutine called grad_descent
    (a stub implementation is found below, fill in the code).
    
    """
    def vanilla_step_fn(pos):
        return -rate * env.gradient(pos)
    return grad_descent(vanilla_step_fn, env)


def grad_descent(step_fn, env):
    """
    A general-purpose gradient descent algorithm.
    
    step_fn is a function that takes a position (x,y) as input 
    (expressed as a 2-element torch.tensor), and returns the
    relative step to take (also expressed as a 2-element torch.tensor).
    
    env is the environment.
    
    The return value should be a list of the positions (including
    the starting position) visited during the gradient descent. 
    
    """
    # Question ONE
    pass


def momentum_grad_descent(rate, env):
    """Gradient descent with momentum.
    
    This is completed for you, but it won't work until you
    correctly implement the MomentumStepFunction class.
    (a stub implementation is found below, fill in the code).
    
    """
    return grad_descent(MomentumStepFunction(env.gradient, rate, 0.3), env)


class MomentumStepFunction:
    """
    Computes the next step for gradient descent with momentum.

    The __call__ method takes a position (x,y) as its argument (expressed
    as a 2-dimensional torch.tensor), and returns the next relative step
    that gradient descent with momentum would take (also expressed as a
    2-dimensional torch.tensor).
        
    """    
    def __init__(self, loss_gradient, learning_rate, momentum_rate):
        # Question TWO
        pass
        
    def __call__(self, pos):
        # Question TWO
        pass


def adagrad(rate, env):
    """Adaptive gradient descent (adagrad).
    
    This is completed for you, but it won't work until you
    correctly implement the AdagradStepFunction class.
    (a stub implementation is found below, fill in the code).
    
    """
    return grad_descent(AdagradStepFunction(env.gradient, rate), env)


class AdagradStepFunction:
    """
    Computes the next step for adagrad.

    The __call__ method takes a position (x,y) as its argument (expressed
    as a 2-dimensional torch.tensor), and returns the next relative step
    that adagrad would take (also expressed as a
    2-dimensional torch.tensor).
        
    """
    def __init__(self, loss_gradient, learning_rate, delta = 0.0000001):
        # Question THREE
        pass
        
    def __call__(self, pos):
        # Question THREE
        pass       


def rmsprop(rate, decay_rate, env):
    """The RMSProp variant of gradient descent.
    
    This is completed for you, but it won't work until you
    correctly implement the RmsPropStepFunction class.
    (a stub implementation is found below, fill in the code).
    
    """
    return grad_descent(RmsPropStepFunction(env.gradient, rate, decay_rate), env)


class RmsPropStepFunction:
    """
    Computes the next step for RmsProp.

    The __call__ method takes a position (x,y) as its argument (expressed
    as a 2-dimensional torch.tensor), and returns the next relative step
    that RmsProp would take (also expressed as a
    2-dimensional torch.tensor).
        
    """
    def __init__(self, loss_gradient, learning_rate, decay_rate, delta=0.000001):
        # Question FOUR
        pass
        
    def __call__(self, pos):
        # Question FOUR
        pass
