import torch
import matplotlib.pyplot as plt

class Environment:
    """
    Interface for the unlit environment.
    
    You can:
        - determine your current (x,y) position
        - determine your current status (actively searching, 
          exceeded_step_limit, found the exit)
        - determine the gradient at a particular (x,y) position
        - step to a new (x,y) position
    
    """
        
    def __init__(self, start_pos, precision, max_steps):
        """
        Initializes the environment.
        
        start_pos is your starting position, a 2-dimensional torch.tensor
        precision is how close you need to be (in both the x and y dimension) 
          to the lowest point in order to see the exit.
        max_steps is the maximum number of steps you're willing to take
          before retreating back to the starting position
        
        """
        self.curr_pos = start_pos
        self.precision = precision
        self.steps = [start_pos]        
        self.max_steps = max_steps
    
    def gradient(self, position):
        """
        Returns the gradient at a particular position.
        
        position is a 2-dimensional torch.tensor, e.g. torch.tensor([x,y])
          where (x,y) is the current position
          
        The return value is also a 2-dimensional torch.tensor.
        
        """
        raise NotImplementedError('Cannot call .gradient() on abstract class.')

    def current_position(self):
        """
        Returns your current (x,y) position.
        
        The return value is a 2-dimensional torch.tensor, e.g.
        torch.tensor([x,y]).
        
        """
        return self.curr_pos
        
    def step_to(self, position):
        """
        Changes your current (x,y) position to the new position.
        
        position is a 2-dimensional torch.tensor, e.g. torch.tensor([x,y]).
        
        """
        self.steps.append(position)
        self.curr_pos = position
        return self.status()

    def can_see_exit(self, position):
        return torch.max(torch.abs(self._goal_position() - position)).item() < self.precision

    ACTIVELY_SEARCHING = 0
    EXCEEDED_STEP_LIMIT = 1
    FOUND_EXIT = 2

    def status(self):
        """
        Returns the current status of your search.
        
        - Environment.ACTIVELY_SEARCHING means that the search is still active.
        - Environment.EXCEEDED_STEP_LIMIT means that you have exceeded the
          maximum number of steps that you are willing to take
        - Environment.FOUND_EXIT means that you have found the lowest point
          of the environment.
        
        """
        if len(self.steps) > self.max_steps + 1:
            return Environment.EXCEEDED_STEP_LIMIT
        elif self.can_see_exit(self.curr_pos):
            return Environment.FOUND_EXIT
        else:
            return Environment.ACTIVELY_SEARCHING
    
    def _goal_position(self):
        """Private method that returns the goal position (lowest point)."""
        raise NotImplementedError('Cannot call .goal_position() on abstract class.')

        

class Level1(Environment):
    
    def __init__(self):
        Environment.__init__(self, torch.tensor([0.0, 0.0]), 0.0001, 100)
    
    def gradient(self, position):
        (x, y) = position
        return torch.tensor([(2*(x-3))/5, (2*(y-4))/10])
    
    def _goal_position(self):
        return torch.tensor([3.0, 4.0])
        
class Level2(Environment):
    
    def __init__(self):
        Environment.__init__(self, torch.tensor([0.0, 0.0]), 0.002, 100)
    
    def gradient(self, position):
        (x, y) = position
        return torch.tensor([(2*(x-5))/1, (2*(y-10))/32])
    
    def _goal_position(self):
        return torch.tensor([5.0, 10.0])

class Level3(Environment):
    
    def __init__(self):
        Environment.__init__(self, torch.tensor([0.0, 0.0]), 0.001, 100)
    
    def gradient(self, position):
        (x, y) = position
        a2 = .4
        b2 = .6
        return torch.tensor([(2*(x+2))/a2, (2*(y-3))/b2])
    
    def _goal_position(self):
        return torch.tensor([-2.0, 3.0])

class Level4(Environment):
    
    def __init__(self):
        Environment.__init__(self, torch.tensor([0.0, 0.0]), 0.001, 100)
    
    def gradient(self, position):
        (x, y) = position
        if x < 3:
            xgrad = -1.
            ygrad = 0.
        elif x > 5:
            xgrad = 1.
            ygrad = 0.
        else:
            xgrad = x-4.
            ygrad = y+7.
        return torch.tensor([xgrad, ygrad])
    
    def _goal_position(self):
        return torch.tensor([4.0, -7.0])

class LevelX(Environment):
    
    def __init__(self):
        Environment.__init__(self, torch.tensor([0.0, 0.0]), 0.002, 10)
    
    def gradient(self, position):
        (x, y) = position
        return torch.tensor([(2*(x-50))/10, (2*(y-50))/20])
    
    def _goal_position(self):
        return torch.tensor([50.0, 50.0])


def visualize_journey(positions, found_destination):
    """
    Plots a series of (x,y) positions. The first point is orange, the intermediate points are gray, and the
    final point is green or red, depending (respectively) on whether the destination has or has not been found.
    
    positions is a list of 2-dimensional torch.tensors
    found_destination is a boolean indicating whether the destination was found
    
    """
    xs = []
    ys = []
    for pt in positions:
        xs.append(pt[0].item())
        ys.append(pt[1].item())
    plt.scatter(xs, ys, color='gray')
    plt.plot(xs, ys, color='gray')
    plt.scatter([0], [0], color='orange')
    if found_destination:
        final_color = 'lime'
    else:
        final_color = 'red'
    plt.scatter([positions[-1][0].item()], [positions[-1][1].item()], color=final_color)
    plt.show()