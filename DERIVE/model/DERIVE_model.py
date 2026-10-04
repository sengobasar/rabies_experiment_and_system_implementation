import os
import numpy as np
import pandas as pd
import time
import tqdm
from scipy.special import erfinv
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import math
from numbers import Number
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from . import DERIVE_encoder, DERIVE_decoder

# nn.CrossEntropyLoss

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)
        
class CustomSampler(torch.utils.data.Sampler):
    def __init__(self, batch_order, batch_size, seq_sample_probs):
        self.batch_order = batch_order
        self.batch_size  = batch_size
        self.seq_sample_probs = seq_sample_probs

    def __iter__(self):
        return iter(np.random.choice(self.batch_order, self.batch_size, p=self.seq_sample_probs).tolist())

    def __len__(self):
        return len(self.indices)
    
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx]
    
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def __len__(self):
        return self.tensor1.shape[0]

    def __getitem__(self, idx):
        return self.tensor1[idx], self.tensor2[idx]

class DERIVE_model(nn.Module):
    """
    Class for the DERIVE model with estimation of weights distribution parameters via Mean-Field VI.
    """
    def __init__(self, *, model_name, training_data, testing_data, graph_data, SASA_data, WCN_data, SASA_std_data, WCN_std_data,
                 encoder_parameters, decoder_parameters, pars_args_result_folder, mylogger, device, 
                 prior_dist, q_dist,
                 alphabet_size=20):
        
        super().__init__()

        self.z_dim       = encoder_parameters['z_dim']
        self.b           = encoder_parameters['TC_b']
        self.prior_dist  = prior_dist
        self.q_dist      = q_dist
        self.model_name  = model_name
        self.device      = device
        self.dtype       = torch.float32
        self.result_folder = pars_args_result_folder
        self.mylogger    = mylogger
        self.alphabet_size    = alphabet_size

        self.testing_data  = testing_data
        self.training_data = training_data
        self.graph_data    = graph_data
        self.SASA_data     = SASA_data
        self.SASA_std_data = SASA_std_data
        self.WCN_data      = WCN_data 
        self.WCN_std_data  = WCN_std_data

        self.alpha_gamma_larger_KL = encoder_parameters['alpha_gamma_larger_KL'] 

        self.training_sample_size, self.seq_len, self.feature_size = training_data.OneHotAndChemicalEncoding_np.shape
        self.Neff          = np.sum(training_data.weights)

        self.encoder_parameters=encoder_parameters
        self.decoder_parameters=decoder_parameters

        encoder_parameters['seq_len']       = self.seq_len
        encoder_parameters['input_feature_size'] = self.feature_size
        encoder_parameters['alphabet_size'] = alphabet_size
        decoder_parameters['seq_len']       = self.seq_len
        decoder_parameters['input_feature_size'] = encoder_parameters['GCNConvEnd_size'] * 3 + self.feature_size
        decoder_parameters['alphabet_size'] = alphabet_size

        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))
        self.relu = nn.ReLU()

        self.encoder = DERIVE_encoder.DERIVE_MLP_encoder_WithGraph(params=encoder_parameters, 
                                                                 graph_data=self.graph_data, 
                                                                 SASA_std_data=self.SASA_std_data,
                                                                 WCN_std_data=self.WCN_std_data,
                                                                 device=self.device)
        self.decoder = DERIVE_decoder.DERIVE_decoder(params=decoder_parameters, device=self.device)

        len_1chain       = SASA_data.shape[0]//3
        len_2chain       = 2*len_1chain
        SASA_data_1chain = [max([SASA_data[x, 0], SASA_data[x + len_1chain, 0], SASA_data[x + len_2chain, 0]]) for x in range(len_1chain)]
        self.SASA_data_1chain = torch.tensor(SASA_data_1chain, dtype=torch.float, requires_grad=False).unsqueeze(0).to(self.device)

        len_1chain       = WCN_data.shape[0]//3
        len_2chain       = 2*len_1chain
        WCN_data_1chain  = [max([WCN_data[x], WCN_data[x + len_1chain], WCN_data[x + len_2chain]]) for x in range(len_1chain)]
        self.WCN_data_1chain = torch.tensor(WCN_data_1chain, dtype=torch.float, requires_grad=False).unsqueeze(0).to(self.device)

        self.logit_sparsity_p = decoder_parameters['logit_sparsity_p']
        
    def sample_latent(self, mu, log_var):
        """
        Samples a latent vector via reparametrization trick
        """
        eps = torch.randn_like(mu).to(self.device)
        z = torch.exp(0.5*log_var) * eps + mu
        return z

    def KLD_diag_gaussians(self, mu, logvar, p_mu, p_logvar):
        """
        KL divergence between diagonal gaussian with prior diagonal gaussian.
        """
        KLD = 0.5 * (p_logvar - logvar) + 0.5 * (torch.exp(logvar) + torch.pow(mu-p_mu,2)) / (torch.exp(p_logvar)+1e-20) - 0.5

        return torch.sum(KLD)

    def annealing_factor(self, annealing_warm_up, training_step):
        """
        Annealing schedule of KL to focus on reconstruction error in early stages of training
        """
        if training_step < annealing_warm_up:
            return training_step/annealing_warm_up
        else:
            return 1

    def get_theta_prior_loss(self):
        """
        KL divergence between the variational distributions and the priors (for the decoder weights).
        """
        theta_prior_loss = 0.0
        zero_tensor = torch.tensor(0.0).to(self.device)

        for i in range(len(self.decoder.hidden_layers_sizes) - 1):
            theta_prior_loss += self.KLD_diag_gaussians(
                self.decoder.state_dict(keep_vars=True)['middle_hidden_layer_bias_mean.' + str(i)].flatten(),
                self.decoder.state_dict(keep_vars=True)['middle_hidden_layer_bias_log_var.' + str(i)].flatten(),
                zero_tensor,
                zero_tensor
            )

        theta_prior_loss += self.KLD_diag_gaussians(
            self.decoder.state_dict(keep_vars=True)['first_hidden_layer_bias_mean'].flatten(),
            self.decoder.state_dict(keep_vars=True)['first_hidden_layer_bias_log_var'].flatten(),
            zero_tensor,
            zero_tensor
        )

        theta_prior_loss += self.KLD_diag_gaussians(
            self.decoder.state_dict(keep_vars=True)['last_hidden_layer_bias_mean'].flatten(),
            self.decoder.state_dict(keep_vars=True)['last_hidden_layer_bias_log_var'].flatten(),
            zero_tensor,
            zero_tensor
        )
        
        return theta_prior_loss
    

    def get_loss_when_train(self, x_recon_log, x, z, kl_global_params_scale, annealing_warm_up, training_step, Neff, z_params,prior_params,data_len):
        """
        Returns mean of negative ELBO, reconstruction loss and KL divergence across batch x.
        """
        BCE1_batch_tensor = F.binary_cross_entropy_with_logits(x_recon_log[:,:,:self.alphabet_size], x[:,:,:self.alphabet_size], reduction='none').sum(dim=2)
        BCE2_batch_tensor = F.mse_loss(x_recon_log[:,:,self.alphabet_size:], x[:,:,self.alphabet_size:], reduction='none').sum(dim=2)
        BCE1_batch_tensor = BCE1_batch_tensor * self.WCN_data_1chain
        BCE2_batch_tensor = BCE2_batch_tensor * self.WCN_data_1chain
        BCE1 = BCE1_batch_tensor.sum(dim=1).sum(dim=0) / x.shape[0]
        BCE2 = BCE2_batch_tensor.sum(dim=1).sum(dim=0) / x.shape[0]
        BCE  = BCE1 + BCE2

        theta_prior_loss_normalized = self.get_theta_prior_loss() / Neff
        warm_up_scale = self.annealing_factor(annealing_warm_up,training_step)

        batch_size = z.shape[0]
        _logqz = self.q_dist.log_density(
            z.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )
        logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * data_len)).sum(1)
        logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * data_len))
        logpz = self.prior_dist.log_density(z, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(z, params=z_params).view(batch_size, -1).sum(1)
        KLD_latent  = torch.relu(logqz_condx - logqz) + self.b * torch.relu(logqz - logqz_prodmarginals) + torch.relu(logqz_prodmarginals - logpz) 
        modified_elbo = BCE + warm_up_scale * (self.alpha_gamma_larger_KL * KLD_latent + kl_global_params_scale * theta_prior_loss_normalized)
        
        return modified_elbo, BCE, KLD_latent, theta_prior_loss_normalized, BCE1, BCE2

    def get_loss_components(self, x):
        """
        Returns tensors of ELBO, reconstruction loss and KL divergence for each point in batch x.
        """
        mu, log_var, x_after_graph = self.encoder(x)
        mu       = mu.unsqueeze(-1)
        log_var  = log_var.unsqueeze(-1)
        z_params = torch.cat([mu, log_var], dim=2)
        z        = self.sample_latent(mu, log_var).squeeze(-1)
        x_recon_log  = self.decoder(z)
        batch_size   = z.shape[0]
        prior_params = self._get_prior_params(batch_size)

        logpz = self.prior_dist.log_density(z, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(z, params=z_params).view(batch_size, -1).sum(1)

        BCE1_batch_tensor = F.binary_cross_entropy_with_logits(x_recon_log[:,:,:self.alphabet_size], x_after_graph[:,:,:self.alphabet_size], reduction='none').sum(dim=2)
        BCE2_batch_tensor = F.mse_loss(x_recon_log[:,:,self.alphabet_size:], x_after_graph[:,:,self.alphabet_size:], reduction='none').sum(dim=2)
        BCE1_batch_tensor = BCE1_batch_tensor.sum(dim=1)
        BCE2_batch_tensor = BCE2_batch_tensor.sum(dim=1)
        BCE_batch_tensor  = BCE1_batch_tensor + BCE2_batch_tensor

        KLD_batch_tensor = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        ELBO_batch_tensor = -( BCE_batch_tensor + torch.relu(logqz_condx - logpz) )

        return ELBO_batch_tensor, BCE_batch_tensor, KLD_batch_tensor, torch.sum(log_var, dim=1), mu, log_var

    def forward(self, x):
        mu, log_var, _ = self.encoder(x)
        return log_var
    
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params
    
    def train_model(self, *, training_data='', training_parameters):
        """
        Training procedure for the DERIVE model.
        If use_validation_set is True then:
            - we split the alignment data in train/val sets.
            - we train up to num_training_steps steps but store the version of the model with lowest loss on validation set across training
        If not, then we train the model for num_training_steps and save the model at the end of training
        """
        if training_data=='':
            training_data = self.training_data

        if torch.cuda.is_available():
            cudnn.benchmark = True

        self.train()
        optimizer = optim.Adam(self.parameters(), lr=training_parameters['learning_rate'], weight_decay = training_parameters['l2_regularization'])
        scheduler = optim.lr_scheduler.CyclicLR( optimizer,
                                                    base_lr=training_parameters['learning_rate']/5.0,
                                                    max_lr =training_parameters['learning_rate'],
                                                    step_size_up=int(training_parameters['num_training_steps']/6),
                                                    step_size_down=int(training_parameters['num_training_steps']/6),
                                                    cycle_momentum=False,
                                                    mode='triangular')

        x_train = torch.tensor(training_data.OneHotAndChemicalEncoding_np, dtype=self.dtype)
        weights_train = training_data.weights
        best_model_step_index = training_parameters['num_training_steps']

        batch_order = np.arange(x_train.shape[0])
        seq_sample_probs = weights_train / np.sum(weights_train)

        self.Neff_training = np.sum(weights_train)        
        start = time.time()
        
        for training_step in tqdm.tqdm(range(1,training_parameters['num_training_steps']+1), desc="Training model"):

            batch_index = np.random.choice(batch_order, training_parameters['batch_size'], p=seq_sample_probs).tolist()
            x = x_train[batch_index].to(self.device)

            data_len = len(batch_order)

            optimizer.zero_grad()

            batch_size = x.shape[0]
            prior_params = self._get_prior_params(batch_size)

            mu, log_var, x_after_graph = self.encoder(x)
            mu       = mu.unsqueeze(-1)
            log_var  = log_var.unsqueeze(-1)
            z_params = torch.cat([mu, log_var], dim=2)  
            z        = self.sample_latent(mu, log_var).squeeze(-1)
            recon_x_log = self.decoder(z)

            neg_ELBO, BCE, KLD_latent, theta_prior_loss_normalized, BCE1, BCE2 = \
                self.get_loss_when_train(recon_x_log, x_after_graph, z, 
                                   training_parameters['kl_global_params_scale'], 
                                   training_parameters['annealing_warm_up'], 
                                   training_step, 
                                   self.Neff_training,
                                   z_params,prior_params,data_len)

            neg_ELBO.mean().backward()
            optimizer.step()
            
            scheduler.step()
            
            if training_step % training_parameters['log_training_freq'] == 0:
                log_message = f"|Train : Update {training_step}, \
                                Negative ELBO : {neg_ELBO.mean():.3f}, \
                                BCE(BCE1+BCE2): {BCE:.3f}, \
                                BCE1(binary_cross_entropy): {BCE1:.3f},\
                                BCE2(mean_square_error): {BCE2:.3f},\
                                KLD_latent: {KLD_latent.mean():.3f}, \
                                theta_prior_loss_normalized: {theta_prior_loss_normalized:.3f}, \
                                Learning Rate: {scheduler.get_last_lr()[0]}, \
                                Time: {time.time() - start:.2f} |"
                self.mylogger.info(log_message)

            if training_step in training_parameters['save_model_params_list']:
                self.save(model_checkpoint=training_parameters['model_checkpoint_location']+os.sep+self.model_name+"_step_"+str(training_step),
                            encoder_parameters=self.encoder_parameters,
                            decoder_parameters=self.decoder_parameters,
                            training_parameters=training_parameters)
                self.eval()
                self.test_model_with_results_save(num_samples=training_parameters["test_model_by_num_samples"], 
                                                     batch_size=256, 
                                                     training_step=training_step)
                self.train()

    def save(self, model_checkpoint, encoder_parameters, decoder_parameters, training_parameters, batch_size=256):
        torch.save({
            'model_state_dict':self.state_dict(),
            'encoder_parameters':encoder_parameters,
            'decoder_parameters':decoder_parameters,
            'training_parameters':training_parameters,
            }, model_checkpoint)

    def test_model_with_results_save(self, *, testing_data='', num_samples=1000, batch_size=256, training_step):

        if testing_data=='':
            test_pd_dict = self.testing_data
        else:
            test_pd_dict = testing_data

        for this_key in test_pd_dict.keys():
            test_pd        = test_pd_dict[this_key] # test_pd.keys(): ['position', 'wt', 'mut', 'mutations', 'aa_str', 'mutations_whole', 'mutations_single', 'whole_count', 'whole_first_seen', 'single_count','single_first_seen', 'OneHotAndChemicalEncoding']
            OneHotAndChemicalEncoding_torch = torch.tensor(np.stack(test_pd.OneHotAndChemicalEncoding.values))
            dataset = MyDataset(OneHotAndChemicalEncoding_torch)
            test_sample_num = OneHotAndChemicalEncoding_torch.shape[0]

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            data_len   = len(dataset)
            prediction_matrix        = torch.zeros((test_sample_num, num_samples))
            prediction_mu_list_matrix     = torch.zeros(test_sample_num, self.encoder_parameters['z_dim'])
            prediction_logvar_list_matrix = torch.zeros(test_sample_num, self.encoder_parameters['z_dim'])

            with torch.no_grad():
                for i, batch in enumerate(tqdm.tqdm(dataloader, 'Looping through mutation batches')):
                    x = batch.type(self.dtype).to(self.device)
                    AAMinNucDis = 0
                    for j in range(num_samples):
                        seq_predictions, BCE_prediction, KL_prediction, logvar_prediction, mu, logvar = self.get_loss_components(x)
                        prediction_matrix[i * batch_size:  i * batch_size + len(x), j] = seq_predictions
                        mu     = mu.squeeze(-1)
                        logvar = logvar.squeeze(-1)
                        if j == 1:
                            prediction_mu_list_matrix[i * batch_size:  i * batch_size + len(x), :] = mu
                            prediction_logvar_list_matrix[i * batch_size:  i * batch_size + len(x), :] = logvar
                mean_predictions = prediction_matrix.mean(dim=1, keepdim=False)
                delta_elbos    = mean_predictions
                evol_indices   = - delta_elbos.detach().cpu().numpy()

            test_pd['evol_indices']   = - evol_indices
            test_pd_subset = test_pd[['evol_indices', 'mutations', 'mutations_whole', 'mutations_single', 'whole_count', 'whole_first_seen', 'single_count','single_first_seen']]
            filename_prefix_path_csv = self.result_folder + '/' + self.model_name + '/' + this_key
            if not os.path.exists(filename_prefix_path_csv):
                os.makedirs(filename_prefix_path_csv)
            test_pd_subset.to_csv(path_or_buf=filename_prefix_path_csv + '/' + 'step_' + str(training_step) + '.csv', index=False)
