import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

class AdaptableNet(nn.Module):
    """
        Initializes an adaptable network.
        
        input_dim (int): gives the dimension of the input
        output_dim (int): gives the dimension of the output
        initial_size (tuple - int): Gives the initial dimension of each hidden layer. Default 1x32 hidden layer.
    """
    def __init__(self, device, input_dim, output_dim, hidden_size=[32,]):
        super(AdaptableNet, self).__init__()
        
        # Save this information for later
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        
        # This is where we will keep all the layers of the network
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_size[0])])
        
        # Hidden layers along with output layer
        for h in range(len(hidden_size)):
            if h == len(hidden_size) - 1:
                self.layers.append(nn.Linear(hidden_size[h], output_dim))
            else:
                self.layers.append(nn.Linear(hidden_size[h], hidden_size[h+1]))
        # Send to the correct device    
        self.layers.to(self.device)
                
      
    """
        Passes an input through a network and returns the output
        
        inp (torch.Tensor, shape: (N, self.input_dim)): input we pass into the network
    """
    def forward(self, inp):
        x = inp
        for l, layer in enumerate(self.layers):
            x = layer(x)
            if l == len(self.layers) - 1:
                x = F.log_softmax(x, dim=1)
            else:
                x = F.relu(x) 
        return x
    
    """
        Increases the number of units in a hidden layer by 1.
    
        h_layer (int): the index of the hidden layer for which we which we will increase.
    """
    def increase_hidden_size(self, h_layer):
        # Assert that we have a valid hidden layer
        assert(h_layer < len(self.hidden_size))
        
        # Get the relevant weight matrices.
        w1 = self.layers[h_layer].weight
        w2 = self.layers[h_layer + 1].weight
        bias = self.layers[h_layer].bias
        
        # add a row/column of 0's to the weight matrices
        with torch.no_grad():
            
            # Construct the new weight matrices (simply append a 0 row/column)
            z1 = torch.Tensor(torch.zeros((1, w1.shape[1]))).to(self.device)
            z2 = torch.Tensor(torch.zeros((w2.shape[0], 1))).to(self.device)
            z3 = torch.Tensor([0]).to(self.device)

            new_mat_1 = nn.Parameter(torch.vstack((w1, z1)))
            new_mat_2 = nn.Parameter(torch.hstack((w2, z2)))
            
            # Set the appropriate weights/biases of our network
            
            self.layers[h_layer].bias = nn.Parameter(torch.cat((bias, z3)))
            self.layers[h_layer].weight = new_mat_1
            self.layers[h_layer + 1].weight = new_mat_2
            self.hidden_size[h_layer] += 1
            
#             if next(self.parameters()).is_cuda:
#                 self.layers[h_layer].cuda()
            
    
    """
        Decreases the number of units in a hidden layer by 1.
    
        h_layer (int): the index of the hidden layer for which we which we will increase.
    """
    def decrease_hidden_size(self, h_layer, mode=2):
        
        # Assert that we have a valid hidden layer, and that we can decrease the number of parameters of this layer.
        assert(h_layer < len(self.hidden_size))
        assert(self.hidden_size[h_layer] > 1)
        
        # Get the relevant weight matrices.
        w1 = self.layers[h_layer].weight
        w2 = self.layers[h_layer + 1].weight
        bias = self.layers[h_layer].bias
        
        # add a row/column of 0's to the weight matrices
        with torch.no_grad():
            
            # Different methods of getting rid of rows based on norms of weight matrices
            if mode == 0:
                row_to_elim = torch.argmin(torch.norm(w1, dim=1))
                col_to_elim = torch.argmin(torch.norm(w2, dim=0))
            elif mode == 1:
                row_to_elim = torch.argmin(torch.norm(torch.hstack((w1, torch.transpose(w2, 0, 1))), dim=1))
                col_to_elim = row_to_elim
            elif mode == 2:
                bias_2d = bias.reshape((bias.shape[0], 1))
                row_to_elim = torch.argmin(torch.norm(torch.hstack((w1, torch.transpose(w2, 0, 1), bias_2d)), dim=1))
                col_to_elim = row_to_elim
            elif mode == 3:
                row_to_elim = np.random.randint(w1.shape[0])
                col_to_elim = np.random.randint(w1.shape[0])
            
            # Construct the new weight matrices (simply append a 0 row/column)
            new_mat_1 = nn.Parameter(torch.vstack((w1[:row_to_elim], w1[row_to_elim + 1:])))
            new_mat_2 = nn.Parameter(torch.hstack((w2[:,:col_to_elim], w2[:,col_to_elim + 1:])))
            
            # Set the appropriate weights/biases of our network
            self.layers[h_layer].bias = nn.Parameter(torch.cat((bias[:row_to_elim], bias[row_to_elim + 1:])))
            self.layers[h_layer].weight = new_mat_1
            self.layers[h_layer + 1].weight = new_mat_2
            self.hidden_size[h_layer] -= 1
            