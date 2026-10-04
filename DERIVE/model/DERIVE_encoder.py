import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
    

class DERIVE_MLP_encoder_WithGraph(nn.Module):
    """
    MLP encoder class for the DERIVE model.
    """
    def __init__(self, params, graph_data, SASA_std_data, WCN_std_data, device):
        """
        Required input parameters:
        - seq_len: (Int) Sequence length of sequence alignment
        - alphabet_size: (Int) Alphabet size of sequence alignment (will be driven by the data helper object)
        - hidden_layers_sizes: (List) List of sizes of DNN linear layers
        - z_dim: (Int) Size of latent space
        - convolve_input: (Bool) Whether to perform 1d convolution on input (kernel size 1, stide 1)
        - convolution_depth: (Int) Size of the 1D-convolution on input
        - nonlinear_activation: (Str) Type of non-linear activation to apply on each hidden layer
        """
        super().__init__()
        self.device            = device
        self.seq_len           = params['seq_len']
        self.alphabet_size     = params['alphabet_size']
        self.input_feature_size = params['input_feature_size']
        self.hidden_layers_sizes = params['hidden_layers_sizes']
        self.z_dim             = params['z_dim']
        self.convolution_depth = params['convolution_input_depth']
        self.GCNConvEnd_size   = params['GCNConvEnd_size']
        self.GCNConvMid_size   = params['GCNConvMid_size']
        self.graph_data        = graph_data
        self.edge_index        = torch.tensor([self.graph_data['edge_index_i_list'], self.graph_data['edge_index_j_list']], dtype=torch.long, requires_grad=False).to(self.device)
        self.edge_weight       = torch.tensor(self.graph_data['edge_weight_list'], dtype=torch.float, requires_grad=False).to(self.device)

        self.conv1 = GCNConv(self.input_feature_size, self.GCNConvMid_size, improved=True, bias=True)
        self.layer_norm1 = nn.LayerNorm(self.GCNConvMid_size, elementwise_affine=False)
        self.conv2 = GCNConv(self.GCNConvMid_size, self.GCNConvEnd_size, improved=True, bias=True)
        self.layer_norm2 = nn.LayerNorm(self.GCNConvEnd_size, elementwise_affine=False)

        in_channels = self.GCNConvEnd_size * 3
        self.channel_size = in_channels + self.input_feature_size
        self.SASA_std_data  = torch.tensor(SASA_std_data, dtype=torch.float, requires_grad=False).to(self.device)
        self.WCN_std_data  = torch.tensor(WCN_std_data, dtype=torch.float, requires_grad=False).unsqueeze(1).to(self.device)

        self.mu_bias_init = 0.1
        self.log_var_bias_init = 1.0


        self.hidden_layers=torch.nn.ModuleDict()
        for layer_index in range(len(self.hidden_layers_sizes)):
            if layer_index==0:
                self.hidden_layers[str(layer_index)] = nn.Linear((self.channel_size*self.seq_len),self.hidden_layers_sizes[layer_index])
                nn.init.constant_(self.hidden_layers[str(layer_index)].bias, self.mu_bias_init)
            else:
                self.hidden_layers[str(layer_index)] = nn.Linear(self.hidden_layers_sizes[layer_index-1],self.hidden_layers_sizes[layer_index])
                nn.init.constant_(self.hidden_layers[str(layer_index)].bias, self.mu_bias_init)
        
        self.fc_mean = nn.Linear(self.hidden_layers_sizes[-1],self.z_dim)
        nn.init.constant_(self.fc_mean.bias, self.mu_bias_init)
        self.fc_log_var = nn.Linear(self.hidden_layers_sizes[-1],self.z_dim)
        nn.init.constant_(self.fc_log_var.bias, self.log_var_bias_init)

        self.nonlinear_activation = nn.ReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(0.1) # We use the default negative_slope: 1e-2
        self.logvar_activation = nn.ELU()

    def Sample_NMuOne(self, std):
        return std * torch.randn_like(std).to(self.device) + 1.0
    
    def forward(self, x):

        idx = x
        x = x.repeat(1,3,1)

        x = self.conv1(x, self.edge_index, edge_weight=self.edge_weight)
        x = self.layer_norm1(x)
        x = self.leaky_relu(x)
        if self.training:
            x  = x * self.Sample_NMuOne(self.SASA_std_data)

        x = self.conv2(x, self.edge_index, edge_weight=self.edge_weight)
        x = self.layer_norm2(x)
        x = self.leaky_relu(x)
        if self.training:
            x  = x * self.Sample_NMuOne(self.WCN_std_data)

        x = torch.cat((x[:,:x.shape[1]//3,:], x[:,x.shape[1]//3:2*x.shape[1]//3,:],x[:,2*x.shape[1]//3:,:]), dim=2)
        x = torch.cat((idx, x), dim=2)
        x_after_graph = x

        x = x.view(-1,self.seq_len*self.channel_size)
        
        for layer_index in range(len(self.hidden_layers_sizes)):
            x = self.nonlinear_activation(self.hidden_layers[str(layer_index)](x))

        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)

        return z_mean, z_log_var, x_after_graph