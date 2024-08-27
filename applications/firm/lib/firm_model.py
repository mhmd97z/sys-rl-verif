from ppo import PPO
from serverless_env import SimEnvironment
from torch import nn as nn
from util import *
import torch

def get_model():
    class ActorNetworkWrapper(nn.Module):
        def __init__(self, input_size=NUM_STATES, hidden_size=HIDDEN_SIZE, 
                output_size=NUM_ACTIONS, base_model=None):
            super(ActorNetworkWrapper, self).__init__()

            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc1.weight.data, self.fc1.bias.data = \
                base_model.fc1.weight.data, base_model.fc1.bias.data

            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc2.weight.data, self.fc2.bias.data = \
                base_model.fc2.weight.data, base_model.fc2.bias.data

            self.fc3 = nn.Linear(hidden_size, output_size)
            self.fc3.weight.data, self.fc3.bias.data = \
                base_model.fc3.weight.data, base_model.fc3.bias.data

            self.relu = nn.ReLU()

        def forward(self, input_):
            # input_ = torch.FloatTensor(input_)
            output = self.relu(self.fc1(input_))
            output = self.relu(self.fc2(output))
            output = self.fc3(output)

            return output


    env = SimEnvironment("data/readfile_sleep_imageresize_output.csv")
    function_name = env.get_function_name()
    initial_state = env.reset(function_name)
    folder_path = "../model/" + str(function_name)
    agent = PPO(env, function_name, folder_path)
    agent.load_checkpoint("../model/ppo.pth.tar")
    return ActorNetworkWrapper(base_model=agent.actor)

def get_params_argmax(input_size):
    # Take sum of the input vars
    c01 = torch.zeros([1, 1, input_size+1])
    c01[0][0][0] = 1

    c02 = torch.zeros([1, 1, input_size+1])
    c02[0][0][0] = 1
    c02[0][0][-1] = 1

    return c01, c02

def get_plain_comparative_firm():
    class MyModel(nn.ModuleList):
        def __init__(self, device=torch.device("cpu")):
            super(MyModel, self).__init__()

            self.input_size = NUM_STATES
            c01, c02 = get_params_argmax(self.input_size)
            self.ft = torch.nn.Flatten()
            
            #################
            # Model
            ################# 
            self.base_model = get_model()
            
            #################
            # Input summation
            #################
            self.input_conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.input_size+1)
            self.input_conv1.weight = torch.nn.Parameter(c01, requires_grad=True)
            self.input_conv1.bias = torch.nn.Parameter(torch.zeros_like(self.input_conv1.bias, requires_grad=True))
            
            self.input_conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.input_size+1)
            self.input_conv2.weight = torch.nn.Parameter(c02, requires_grad=True)
            self.input_conv2.bias = torch.nn.Parameter(torch.zeros_like(self.input_conv2.bias, requires_grad=True))
            
        def forward(self, obs):
            # input processing
            input1 = self.input_conv1(obs)
            input2 = self.input_conv2(obs)
            # the model
            copy1_logits = self.base_model(input1)
            copy2_logits = self.base_model(input2)

            return self.ft(torch.concat((copy1_logits[0], copy2_logits[0]), dim=1))

    return MyModel()
    
if __name__ == "__main__":
    model = get_plain_comparative_firm()
    x = torch.tensor([[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]]) # .to(device="cuda")
    print(model(x))