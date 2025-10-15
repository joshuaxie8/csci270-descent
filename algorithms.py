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

    pos = env.current_position()
    result = [pos]
    while (env.status() == Environment.ACTIVELY_SEARCHING):
        pos = pos + step_fn(pos)
        env.step_to(pos)
        result.append(pos)

    return result


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
        self.loss_gradient = loss_gradient
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.prev_step = 0
        
    def __call__(self, pos):
        # Question TWO
        step = -self.learning_rate * self.loss_gradient(pos) + self.momentum_rate*self.prev_step
        self.prev_step = step
        return step


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
        self.loss_gradient = loss_gradient
        self.initial_rate = learning_rate
        self.delta = delta
        self.odometer = 0
        
    def __call__(self, pos):
        self.odometer += self.loss_gradient(pos) ** 2
        rate = self.initial_rate / (self.delta + (self.odometer) ** 0.5)
        return -rate * self.loss_gradient(pos)


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
        self.loss_gradient = loss_gradient
        self.initial_rate = learning_rate
        self.decay_rate = decay_rate
        self.delta = delta
        self.decaying_average = None
        
    def __call__(self, pos):
        if (self.decaying_average is None):
            self.decaying_average = self.loss_gradient(pos) ** 2
        else:
            self.decaying_average = self.decay_rate * self.decaying_average + \
                (1 - self.decay_rate) * self.loss_gradient(pos) ** 2
        rate = self.initial_rate / (self.delta + (self.decaying_average) ** 0.5)
        return -rate * self.loss_gradient(pos)
