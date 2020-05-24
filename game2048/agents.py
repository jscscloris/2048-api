import numpy as np
import torch.nn as nn
import torch

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

dic={0:[0,0,0,0,0,0,0,0,0,0,0],2:[0,0,0,0,0,0,0,0,0,0,1],4:[0,0,0,0,0,0,0,0,0,1,0],8:[0,0,0,0,0,0,0,0,1,0,0],
     16:[0,0,0,0,0,0,0,1,0,0,0],32:[0,0,0,0,0,0,1,0,0,0,0],64:[0,0,0,0,0,1,0,0,0,0,0],128:[0,0,0,0,1,0,0,0,0,0,0],
     256:[0,0,0,1,0,0,0,0,0,0,0],512:[0,0,1,0,0,0,0,0,0,0,0],1024:[0,1,0,0,0,0,0,0,0,0,0],2048:[1,0,0,0,0,0,0,0,0,0,0]}

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.net=nn.Sequential(
        nn.Linear(16*11,25),
        nn.ReLU(),
        nn.Linear(25,12),
        nn.ReLU(),
        nn.Linear(12,4),
        nn.Softmax(dim=1)
    )
  def forward(self,x):
    x=x.view(x.size(0),-1)
    out=self.net(x)
    return out


class MyOwnAgent(Agent):
  def step(self):
    model = Net()
    model.load_state_dict(torch.load('model.pkl'))
    bd=np.array(self.game.board)
    bd=bd.flatten()
    data=[dic[n] for n in bd]
    data=torch.Tensor([data]).float()
    out=model.forward(data)
    direction = out.argmax()
    return direction.item()
