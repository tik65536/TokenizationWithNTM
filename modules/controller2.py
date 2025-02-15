import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class Controller2(nn.Module):

    def __init__(self, input_dim, ctrl_dim, output_dim, read_data_size,device=None):
        super(Controller2, self).__init__()
        if (device is not None):
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.input_size = input_dim
        self.ctrl_dim = ctrl_dim
        self.output_size = output_dim
        self.read_data_size = read_data_size

        # Controller neural network
        #self.controller_net = nn.LSTMCell(input_dim, ctrl_dim)
        self.controller_net = nn.LSTM(input_dim, ctrl_dim)
        # Output neural network
        self.out_net = nn.Linear(read_data_size, output_dim)
        # Initialize the weights of output net
        nn.init.kaiming_uniform_(self.out_net.weight)

        # Learnable initial hidden and cell states
        self.h_state = torch.zeros([1, ctrl_dim]).to(self.device)
        self.c_state = torch.zeros([1, ctrl_dim]).to(self.device)
        # Layers to learn init values for controller hidden and cell states
        self.h_bias_fc = nn.Linear(1, ctrl_dim)
        self.c_bias_fc = nn.Linear(1, ctrl_dim)
        # Reset
        self.reset()

    def forward(self, x, prev_reads):
        '''Returns the hidden and cell states'''
        x = torch.cat([x, prev_reads], dim=0).to(self.device)
        # Get hidden and cell states
        _, (self.h_state, self.c_state) = self.controller_net.to(self.device)(x, (self.h_state, self.c_state))

        return self.h_state, self.c_state

    def output(self, reads):
        '''Returns the external output from the read vectors'''
        #reads=reads.reshape(1,self.read_data_size-self.ctrl_dim).to(self.device)
        reads=reads.reshape(1,self.read_data_size).to(self.device)
        #out_state = torch.cat([self.h_state, reads], dim=1).to(self.device)
        # Compute output
        #output = torch.sigmoid(self.out_net.to(self.device)(out_state))
        output = self.out_net.to(self.device)(reads)

        return output

    def reset(self, batch_size=1):
        '''Reset/initialize the controller states'''
        # Dummy input
        in_data = torch.tensor([[0.]])
        # Hidden state
        h_bias = self.h_bias_fc(in_data)
        self.h_state = h_bias.repeat(batch_size, 1).to(self.device)
        # Cell state
        c_bias = self.c_bias_fc(in_data)
        self.c_state = c_bias.repeat(batch_size, 1).to(self.device)
