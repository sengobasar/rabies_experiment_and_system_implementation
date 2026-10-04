import torch
import torch.nn as nn
import torch.nn.functional as F


class DERIVE_decoder(nn.Module):
    """
    Bayesian MLP decoder class for the DERIVE model.
    """

    def __init__(self, params, device):
        """
        Required input parameters:
        - seq_len: (Int) Sequence length of sequence alignment
        - alphabet_size: (Int) Alphabet size of sequence alignment (will be driven by the data helper object)
        - hidden_layers_sizes: (List) List of the sizes of the hidden layers (all DNNs)
        - z_dim: (Int) Dimension of latent space
        - first_hidden_nonlinearity: (Str) Type of non-linear activation applied on the first (set of) hidden layer(s)
        - last_hidden_nonlinearity: (Str) Type of non-linear activation applied on the very last hidden layer (pre-sparsity)
        - dropout_proba: (Float) Dropout probability applied on all hidden layers. If 0.0 then no dropout applied
        - convolution_depth: (Int) Size of the 1D-convolution on output
        - include_temperature_scaler: (Bool) Whether we apply the global temperature scaler
        - include_sparsity: (Bool) Whether we use the sparsity inducing scheme on the output from the last hidden layer
        - num_tiles_sparsity: (Int) Number of tiles to use in the sparsity inducing scheme (the more the tiles, the stronger the sparsity)
        - bayesian_decoder: (Bool) Whether the decoder is bayesian or not
        """
        super().__init__()
        self.device              = device
        self.seq_len             = params['seq_len']
        self.alphabet_size       = params['alphabet_size']
        self.input_feature_size  = params['input_feature_size']
        self.hidden_layers_sizes = params['hidden_layers_sizes']
        self.z_dim               = params['z_dim']
        self.dropout_proba       = params['dropout_proba']
        self.convolution_depth   = params['convolution_output_depth']

        self.first_hidden_nonlinearity = nn.ReLU()
        self.last_hidden_nonlinearity = nn.ReLU()

        if self.dropout_proba > 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout_proba)

        self.mu_bias_init  = 0.1
        self.logvar_init   = -10.0
        self.logit_scale_p = 0.001

        self.channel_size = self.convolution_depth

        self.first_hidden_layer_weight         = nn.Parameter(   torch.zeros(self.hidden_layers_sizes[0], self.z_dim)   )
        self.first_hidden_layer_bias_mean      = nn.Parameter(   torch.zeros(self.hidden_layers_sizes[0]))
        self.first_hidden_layer_bias_log_var   = nn.Parameter(   torch.zeros(self.hidden_layers_sizes[0]))
        nn.init.xavier_normal_(self.first_hidden_layer_weight)  # Glorot initialization
        nn.init.constant_(self.first_hidden_layer_bias_mean,    self.mu_bias_init)
        nn.init.constant_(self.first_hidden_layer_bias_log_var, self.logvar_init)

        self.middle_hidden_layer_weight        = nn.ParameterList()
        self.middle_hidden_layer_bias_mean     = nn.ParameterList()
        self.middle_hidden_layer_bias_log_var  = nn.ParameterList()
        for i in range(len(self.hidden_layers_sizes)-1):
            self.middle_hidden_layer_weight.append(        nn.Parameter( torch.zeros(self.hidden_layers_sizes[i+1], self.hidden_layers_sizes[i]) ) )
            self.middle_hidden_layer_bias_mean.append(     nn.Parameter( torch.zeros(self.hidden_layers_sizes[i+1]) ) )
            self.middle_hidden_layer_bias_log_var.append(  nn.Parameter( torch.zeros(self.hidden_layers_sizes[i+1]) ) )
            nn.init.xavier_normal_(self.middle_hidden_layer_weight[i])  # Glorot initialization
            nn.init.constant_(self.middle_hidden_layer_bias_mean[i],    self.mu_bias_init)
            nn.init.constant_(self.middle_hidden_layer_bias_log_var[i], self.logvar_init)


        self.last_hidden_layer_weight         = nn.Parameter(   torch.zeros(self.channel_size * self.seq_len, self.hidden_layers_sizes[-1])   )
        self.last_hidden_layer_conv_weight    = nn.Parameter(   torch.zeros(self.channel_size,                self.input_feature_size)   )
        self.last_hidden_layer_bias_mean      = nn.Parameter(   torch.zeros(self.input_feature_size * self.seq_len)   )
        self.last_hidden_layer_bias_log_var   = nn.Parameter(   torch.zeros(self.input_feature_size * self.seq_len)   )
        nn.init.xavier_normal_(self.last_hidden_layer_weight)  # Glorot initialization
        nn.init.constant_(self.last_hidden_layer_bias_mean,    self.mu_bias_init)
        nn.init.constant_(self.last_hidden_layer_bias_log_var, self.logvar_init)

        self.temperature_scaler_mean    = nn.Parameter(torch.ones(1))
        self.temperature_scaler_log_var = nn.Parameter(torch.ones(1) * self.logvar_init)

    def sampler(self, mean, log_var):
        """
        Samples a latent vector via reparametrization trick
        """
        eps = torch.randn_like(mean).to(self.device)
        z = torch.exp(0.5 * log_var) * eps + mean
        return z

    def forward(self, z):

        batch_size = z.shape[0]
        if self.dropout_proba > 0.0:
            x = self.dropout_layer(z)
        else:
            x = z

        first_bias = self.sampler(self.first_hidden_layer_bias_mean, self.first_hidden_layer_bias_log_var)
        x = self.first_hidden_nonlinearity(F.linear(x, weight=self.first_hidden_layer_weight, bias=first_bias))

        for i in range(len(self.hidden_layers_sizes) - 1):
            middle_bias = self.sampler(self.middle_hidden_layer_bias_mean[i], self.first_hidden_layer_bias_log_var[i])
            if i == len(self.hidden_layers_sizes) - 2:
                x = self.last_hidden_nonlinearity(F.linear( x, weight=self.middle_hidden_layer_weight[i], bias=middle_bias))
            else:
                x = self.first_hidden_nonlinearity(F.linear(x, weight=self.middle_hidden_layer_weight[i], bias=middle_bias))
            if self.dropout_proba > 0.0:
                x = self.dropout_layer(x)

        W_out = self.last_hidden_layer_weight
        b_out = self.sampler(self.last_hidden_layer_bias_mean,   self.last_hidden_layer_bias_log_var)

        W_out = torch.mm(W_out.view(self.seq_len * self.hidden_layers_sizes[-1], self.channel_size),
                             self.last_hidden_layer_conv_weight)  # product of size (H * seq_len, alphabet)

        W_out = W_out.view(self.seq_len * self.input_feature_size, self.hidden_layers_sizes[-1])

        x = F.linear(x, weight=W_out, bias=b_out)

        temperature_scaler = self.sampler(self.temperature_scaler_mean, self.temperature_scaler_log_var)
        x = torch.log(1.0 + torch.exp(temperature_scaler)) * x

        x = x.view(batch_size, self.seq_len, self.input_feature_size)
        x_recon_log = torch.cat(( F.log_softmax(x[:,:,:self.alphabet_size], dim=-1), x[:,:,self.alphabet_size:] ), dim=2) 

        return x_recon_log


