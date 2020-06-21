import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

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


map_table={2**i:i for i in range(1,16)}
map_table[0]=0
def grid_ohe(arr):
  ret=np.zeros(shape=(4,4,16),dtype=float)
  for r in range(4):
    for c in range(4):
      ret[r,c,map_table[arr[r,c]]]=1
  return ret


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(16, 64, kernel_size=(4, 1), padding=(2, 0))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 4), padding=(0, 2))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(4, 4), padding=(2, 2))
        self.conv6 = nn.Conv2d(128,128,kernel_size=(2,2))
        self.fc1 = nn.Linear(128 * 4 * 4, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 4)
        self.batch_norm1 = nn.BatchNorm1d(128 * 4 * 4)
        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.batch_norm3 = nn.BatchNorm1d(512)

        self.initialize()

    def forward(self,x):        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.batch_norm1(x)
        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm3(x)
        x = self.fc3(x)

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')

model_64 = Net()
model_64 = model_64.cuda()
model_64.eval()
model_64.load_state_dict(torch.load('64_epoch_200.pkl'))

model_128 = Net()
model_128 = model_128.cuda()
model_128.eval()
model_128.load_state_dict(torch.load('128_epoch_200.pkl'))

model_256 = Net()
model_256 = model_256.cuda()
model_256.eval()
model_256.load_state_dict(torch.load('256_epoch_500.pkl'))

model_512 = Net()
model_512 = model_512.cuda()
model_512.eval()
model_512.load_state_dict(torch.load('512_epoch_200.pkl'))

model_1024 = Net()
model_1024 = model_1024.cuda()
model_1024.eval()
model_1024.load_state_dict(torch.load('1024_epoch_200.pkl'))


class MyOwnAgent(Agent):
  def step(self):    
    bd=np.array(self.game.board)
    bd_ohe=grid_ohe(bd)
    bd_ohe=np.swapaxes(bd_ohe,0,2)
    bd_ohe=np.expand_dims(bd_ohe, axis=0)
    bd_ohe=torch.from_numpy(bd_ohe).float()
    score=self.game.score
    bd_ohe=Variable(bd_ohe).cuda()
    if score<=64:
      out=model_64(bd_ohe)
    elif score==128:
      out=model_128(bd_ohe)
    elif score==256:
      out=model_256(bd_ohe)
    elif score==512:
      out=model_512(bd_ohe)
    elif score==1024:
      out=model_1024(bd_ohe)

    direction = nn.functional.softmax(out,dim=1).argmax()
    return direction.item()
