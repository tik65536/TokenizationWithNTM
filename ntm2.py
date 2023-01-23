import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from modules import Memory2, Head2, Controller2

class NTM(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 ctrl_dim,
                 memory_units,
                 memory_unit_size,
                 num_heads):
        super(NTM, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # Create controller
        self.ctrl_dim = ctrl_dim
        self.controller = Controller2(input_dim ,#+ memory_unit_size,
                                     ctrl_dim,
                                     output_dim,
                                     num_heads * memory_unit_size,self.device)

        # Create memory
        self.memory = Memory2(memory_units, memory_unit_size,self.device)
        self.memory_unit_size = memory_unit_size # M
        self.memory_units = memory_units # N

        # Create Heads
        self.num_heads = num_heads
        self.heads = nn.ModuleList([])
        # Change ordering of RW
        for head in range(num_heads):
            self.heads += [
                Head2('w', ctrl_dim, memory_unit_size,self.device),
                Head2('r', ctrl_dim, memory_unit_size,self.device)
            ]

        # Init previous head weights and read vectors
        self.prev_head_rweights = []
        self.prev_head_wweights = []
        self.prev_reads = []
        # Layers to initialize the weights and read vectors
        self.head_weights_fc = nn.Linear(1, self.memory_units)
        self.reads_fc = nn.Linear(1, self.memory_unit_size)

        self.eraseHistory=[]
        self.addHistory=[]

        #self.reset()


    def forward(self, x):
        '''Returns the output of the Neural Turing Machine'''
        # Get controller states
        #prev_reads = torch.Tensor(self.prev_reads)
        ctrl_hidden, ctrl_cell = self.controller(x, self.prev_reads)

        # Read, and Write
        head_rweights = []
        head_wweights = []
        erase=[]
        add=[]
        reads=torch.FloatTensor(self.num_heads,self.memory_unit_size)
        idx=0
        for head, prev_head_rweights,prev_head_wweights in zip(self.heads, self.prev_head_rweights,self.prev_head_wweights):
            # Read
            if head.mode == 'r':
                rweights,_, read_vec ,_ ,_  = head(ctrl_cell, prev_head_rweights,prev_head_wweights, self.memory)
                reads[idx]=read_vec.reshape(self.memory_unit_size,)
                idx+=1
                head_rweights.append(rweights)
                #reads.append(read_vec)
            # Write
            elif head.mode == 'w':
                _,wweights, _ , e ,a = head(ctrl_cell, prev_head_rweights,prev_head_wweights, self.memory)
                erase.append(e.detach().cpu().numpy())
                add.append(a.detach().cpu().numpy())
                head_wweights.append(wweights)

        # Compute output
        output = self.controller.output(reads)

        self.prev_head_rweights = head_rweights
        self.prev_head_wweights = head_wweights
        self.prev_reads = reads
        self.eraseHistory.append(erase)
        self.addHistory.append(add)

        return output


    def reset(self, batch_size=1):
        '''Reset/initialize NTM parameters'''
        # Reset memory and controller
        self.memory.reset(batch_size)
        self.controller.reset(batch_size)
        self.eraseHistory=[]
        self.addHistory=[]
        # Initialize previous head weights (attention vectors)
        self.prev_head_weights = []
        for i in range(len(self.heads)):
            # prev_weight = torch.zeros([batch_size, self.memory.n])
            prev_weight = F.softmax(self.head_weights_fc(torch.Tensor([[0.]])), dim=1)
            self.prev_head_weights.append(prev_weight)

        # Initialize previous read vectors
        reads=torch.FloatTensor(self.num_heads,self.memory_unit_size)
        for i in range(self.num_heads):
            # prev_read = torch.zeros([batch_size, self.memory.m])
            # nn.init.kaiming_uniform_(prev_read)
            reads[i] = self.reads_fc(torch.Tensor([[0.]]))
        self.prev_reads = reads
