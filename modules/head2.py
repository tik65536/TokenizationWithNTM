import torch
import torch.nn.functional as F
from torch import nn


class Head2(nn.Module):

    def __init__(self, mode, ctrl_dim, memory_unit_size,device=None):
        super(Head2, self).__init__()
        # Valid modes are 'r' and 'w' for reading and writing respectively
        if (device is not None):
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.mode = mode
        # Size of each memory vector (key size)
        self.m = memory_unit_size

        self.max_shift = 1

        # Linear Layers for converting controller output to addressing parameters
        self.read_key_fc = nn.Linear(ctrl_dim, self.m)
        self.read_key_strength_fc = nn.Linear(ctrl_dim, 1)
        self.read_interpolation_gate_fc = nn.Linear(ctrl_dim, 1)
        self.read_shift_weighting_fc = nn.Linear(ctrl_dim, 3)
        self.read_sharpen_factor_fc = nn.Linear(ctrl_dim, 1)

        self.write_key_fc = nn.Linear(ctrl_dim, self.m)
        self.write_key_strength_fc = nn.Linear(ctrl_dim, 1)
        self.write_interpolation_gate_fc = nn.Linear(ctrl_dim, 1)
        self.write_shift_weighting_fc = nn.Linear(ctrl_dim, 3)
        self.write_sharpen_factor_fc = nn.Linear(ctrl_dim, 1)
        self.erase_weight_fc = nn.Linear(ctrl_dim, self.m)
        self.add_data_fc = nn.Linear(ctrl_dim, self.m)

        # Reset
        self.reset()

    def forward(self, ctrl_state, prev_rweights,prev_wweights,externalInput, memory):
        '''Extracts the parameters and returns the attention weights
        Args:
            ctrl_state (tensor): output vector from the controller (batch_size, ctrl_dim)
            prev_weights (tensor): previous attention weights (batch_size, N)
            memory (nn.Module): memory module
        '''

        # Extract the parameters from controller state
        #ctrl_state = ctrl_state.to(self.devices)
        rkey = torch.tanh(self.read_key_fc.to(self.device)(ctrl_state))
        rbeta = F.softplus(self.readkey_strength_fc.to(self.device)(ctrl_state))
        rgate = torch.sigmoid(self.read_interpolation_gate_fc.to(self.device)(ctrl_state))
        rshift = F.softmax(self.read_shift_weighting_fc.to(self.device)(ctrl_state), dim=1)
        rgamma = 1 + F.softplus(self.read_sharpen_factor_fc.to(self.device)(ctrl_state))


        wkey = torch.tanh(self.write_key_fc.to(self.device)(ctrl_state))
        wbeta = F.softplus(self.writekey_strength_fc.to(self.device)(ctrl_state))
        wgate = torch.sigmoid(self.write_interpolation_gate_fc.to(self.device)(ctrl_state))
        wshift = F.softmax(self.write_shift_weighting_fc.to(self.device)(ctrl_state), dim=1)
        wgamma = 1 + F.softplus(self.write_sharpen_factor_fc.to(self.device)(ctrl_state))
        erase = torch.sigmoid(self.erase_weight_fc.to(self.device)(ctrl_state))
        #add = torch.tanh(self.add_data_fc.to(self.device)(ctrl_state))

        # ==== Addressing ====
        # Content-based addressing
        read_content_weights = memory.content_addressing(rkey, rbeta)
        write_content_weights = memory.content_addressing(wkey, wbeta)

        # Location-based addressing
        # Interpolation
        prev_rweights= prev_rweights.to(self.device)
        prev_wweights= prev_wweights.to(self.device)
        read_gated_weights = self._gated_interpolation(read_content_weights, prev_rweights, rgate)
        # Convolution
        read_shifted_weights = self._conv_shift(read_gated_weights, rshift)
        # Sharpening
        rweights = self._sharpen(read_shifted_weights, rgamma)

        write_gated_weights = self._gated_interpolation(write_content_weights, prev_wweights, wgate)
        # Convolution
        write_shifted_weights = self._conv_shift(write_gated_weights, wshift)
        # Sharpening
        wweights = self._sharpen(write_shifted_weights, wgamma)

        memory_erased=None
        memory_added=None
        # ==== Read / Write Operation ====
        # Read
        if self.mode == 'r':
            read_vec = memory.read(rweights)
        # Write
        elif self.mode == 'w':
            memory_erased =  (1 - wweights.unsqueeze(2) * erase.unsqueeze(1))
            memory_added = (wweights.unsqueeze(2) * input.unsqueeze(1))
            memory.write(wweights, erase, externalInput)
            read_vec = None
        else:
            raise ValueError("mode must be read ('r') or write ('w')")

        return rweights, wweights, read_vec , memory_erased, memory_added

    def _gated_interpolation(self, w, prev_w, g):
        '''Returns the interpolated weights between current and previous step's weights
        Args:
            w (tensor): weights (batch_size, N)
            prev_w (tensor): weights of previous timestep (batch_size, N)
            g (tensor): a scalar interpolation gate (batch_size, 1)
        Returns:
            (tensor): content weights (batch_size, N)
        '''
        return (g * w) + ((1 - g) * prev_w)

    def _conv_shift(self, w, s):
        '''Returns the convolved weights
        Args:
            w (tensor): weights (batch_size, N)
            s (tensor): shift weights (batch_size, 2 * max_shift + 1)
        Returns:
            (tensor): convolved weights (batch_size, N)
        '''
        batch_size = w.size(0)
        max_shift = int((s.size(1) - 1) / 2)

        unrolled = torch.cat([w[:, -max_shift:], w, w[:, :max_shift]], 1)
        return F.conv1d(unrolled.unsqueeze(1), s.unsqueeze(1))[range(batch_size), range(batch_size)]

    def _sharpen(self, w, gamma):
        '''Returns the sharpened weights
        Args:
            w (tensor): weights (batch_size, N)
            gamma (tensor): gamma value for sharpening (batch_size, 1)
        Returns:
            (tensor): sharpened weights (batch_size, N)
        '''
        w = w.pow(gamma)
        return torch.div(w, w.sum(1).view(-1, 1) + 1e-16)

    def reset(self):
        '''Reset/initialize the head parameters'''
        # Weights
        nn.init.xavier_uniform_(self.key_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.key_strength_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.interpolation_gate_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.shift_weighting_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.sharpen_factor_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.add_data_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.erase_weight_fc.weight, gain=1.4)

        # Biases
        nn.init.normal_(self.key_fc.bias, std=0.01)
        nn.init.normal_(self.key_strength_fc.bias, std=0.01)
        nn.init.normal_(self.interpolation_gate_fc.bias, std=0.01)
        nn.init.normal_(self.shift_weighting_fc.bias, std=0.01)
        nn.init.normal_(self.sharpen_factor_fc.bias, std=0.01)
        nn.init.normal_(self.add_data_fc.bias, std=0.01)
        nn.init.normal_(self.erase_weight_fc.bias, std=0.01)

