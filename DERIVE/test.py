import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime as dt
import logging
import torch
import os

import lib.dist as dist
from generate_candidate_set import GISAID_data_class, MSA_data_class
from DERIVE import DERIVE_model
from lib.flows import FactorialNormalizingFlow


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='For test')

  
    parser.add_argument('--GCN_pretrained_initialization', type=bool, default=False, help='True: use pretrained GCN initialization')
    parser.add_argument('--d0_similarity_threshold', type=float, default=0.01, help='Similarity threshold for the second level sequence re-weighting')
    parser.add_argument('--eta_balancing_strength', type=float, default=1, help='balancing strength for the second level sequence re-weighting')
    parser.add_argument('--z_dim_latent_dimension', type=int, default=64, help='latent space dimension')
    parser.add_argument('--K_flow_iterations', type=int, default=32, help='iterations for the factorized normalizing flow')
    parser.add_argument('--alpha_gamma_larger_KL', type=int, default=2, help='regularization scales for both normalizing flow prior term and disentanglement term')

    parser.add_argument('--GISAID_data_folder', type=str, default='./data/GISAID', help='Folder where GISAID data are stored')
    parser.add_argument('--figs_ForGISAIDData_folder', type=str, default='./figs/GISAID', help='Folder where figs related to GISAID data are stored')
    parser.add_argument('--GISAID_data_name', type=str, default='/all_seqs_scores', help='Name of GISAID with /....')
    parser.add_argument('--MSA_data_folder', type=str, default='./data/MSA', help='Folder where MSAs are stored')
    parser.add_argument('--MSA_data_name', type=str, default='./data/MSA/SARS_CoV_2_Spike.a2m', help='Name of MSAs with /....a2m')
    parser.add_argument('--result_folder', type=str, default='./results', help='Output location of computed evol indices')
    parser.add_argument('--training_logs_location', default='./logs/', type=str, help='Location of DERIVE model parameters')
    parser.add_argument('--model_parameters_location', default='./DERIVE/default_model_params_new.json', type=str, help='Location of DERIVE model parameters')
    parser.add_argument('--model_checkpoint_location', type=str, default='./checkpoints', help='Location where model checkpoints will be stored')
    parser.add_argument('--CUDA_num', default='0', type=str, help='CUDA device number')


    pars_args   = parser.parse_args()
    with_pretrained_graph = pars_args.GCN_pretrained_initialization
    d0          = pars_args.d0_similarity_threshold
    eta         = pars_args.eta_balancing_strength
    z_dim       = pars_args.z_dim_latent_dimension
    K_flow      = pars_args.K_flow_iterations
    alpha_gamma = pars_args.alpha_gamma_larger_KL
    Mydevice    = torch.device("cuda:" + pars_args.CUDA_num  if torch.cuda.is_available() else "cpu")


    ckpt_root = pars_args.model_checkpoint_location
    if not os.path.exists(ckpt_root):
        raise FileNotFoundError("model_checkpoint_location not found: " + ckpt_root)

    ckpt_candidates = []
    for root, dirs, files in os.walk(ckpt_root):
        for f in files:
            lf = f.lower()
            if lf.endswith('.pt') or lf.endswith('.pth') or lf.endswith('.ckpt'):
                full_path = os.path.join(root, f)
                ckpt_candidates.append(full_path)

    if len(ckpt_candidates) == 0:
        raise FileNotFoundError("No checkpoint (.pt/.pth/.ckpt) found under: " + ckpt_root)

    ckpt_candidates = sorted(ckpt_candidates, key=lambda p: os.path.getmtime(p), reverse=True)
    checkpoint_path = ckpt_candidates[0]
    print("Using latest checkpoint: " + checkpoint_path)


    print("Start: data for testing")
    GISAID = GISAID_data_class(pars_args_GISAID_data_folder=pars_args.GISAID_data_folder,
                                pars_args_figs_ForGISAIDData_folder=pars_args.figs_ForGISAIDData_folder,
                                pars_args_GISAID_data_name=pars_args.GISAID_data_name)
    graph_data_dict = GISAID.get_graph_data(pdb_file='./data/PDB/6vxx.pdb')
    WCN_data_np, WCN_std_data_np = graph_data_dict['WCN_list'], graph_data_dict['WCN_std_list']
    SASA_data_np, SASA_std_data_np = GISAID.get_SASA(pdb_file='./data/PDB/6vxx.pdb',  SASA_key_list=['relativeTotal'])
    test_pd_dict = GISAID.get_LeadingWHOVOC_based_1mut_sets()
    GISAID.compute_OneHotAndChemicalEncoding_ForVal(test_pd_dict)
    print("End: data for testing")



    model_params = json.load(open(pars_args.model_parameters_location))
    model_params["training_parameters"]['training_logs_location']    = pars_args.training_logs_location
    model_params["training_parameters"]['model_checkpoint_location'] = pars_args.model_checkpoint_location
    model_params["encoder_parameters"]['alpha_gamma_larger_KL']      = alpha_gamma
    model_params["encoder_parameters"]["hidden_layers_sizes"]        = [4000,1000,300]
    model_params["encoder_parameters"]["z_dim"]                      = z_dim
    model_params["decoder_parameters"]["z_dim"]                      = z_dim

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    model_name = (pars_args.MSA_data_name[1:-4] + "_") + dt.now().strftime("%Y-%m-%d---%H-%M-%S")

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    if not os.path.exists(pars_args.training_logs_location):
        os.makedirs(pars_args.training_logs_location)

    file_handler = logging.FileHandler(pars_args.training_logs_location + model_name + '.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('Dictionary content: {}'.format(json.dumps(model_params, indent=4)))
    logger.info('Using latest checkpoint: {}'.format(checkpoint_path))

    model = DERIVE_model.DERIVE_model(
                    model_name=model_name,
                    testing_data=test_pd_dict,
                    graph_data=graph_data_dict,
                    SASA_data=SASA_data_np,
                    WCN_data=WCN_data_np,
                    SASA_std_data=SASA_std_data_np,
                    WCN_std_data=WCN_std_data_np,
                    encoder_parameters=model_params["encoder_parameters"],
                    decoder_parameters=model_params["decoder_parameters"],
                    mylogger=logger,
                    pars_args_result_folder=pars_args.result_folder,
                    device=Mydevice,
                    prior_dist=FactorialNormalizingFlow(dim= z_dim , nsteps=K_flow), 
                    q_dist=dist.Normal(),
    )

    for name, param in model.named_parameters():
        print(name, param.size())
    model = model.to(Mydevice)
    model.eval()

    print("Loading checkpoint: " + checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=Mydevice)

    if isinstance(ckpt, dict) and ('model_state_dict' in ckpt):
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and ('state_dict' in ckpt):
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", len(missing_keys))
    print("Unexpected keys:", len(unexpected_keys))
    if len(missing_keys) > 0:
        print("Missing keys head:", missing_keys[:20])
    if len(unexpected_keys) > 0:
        print("Unexpected keys head:", unexpected_keys[:20])

    mutations = []
    if isinstance(test_pd_dict, dict):
        for k, v in test_pd_dict.items():
            if isinstance(v, pd.DataFrame):
                if 'mutations' in v.columns:
                    mutations += v['mutations'].dropna().astype(str).tolist()
                elif 'mutation' in v.columns:
                    mutations += v['mutation'].dropna().astype(str).tolist()
                elif 'aa_mutation' in v.columns:
                    mutations += v['aa_mutation'].dropna().astype(str).tolist()
            elif isinstance(v, (list, tuple, set)):
                mutations += [str(x) for x in v]

    uniq_mut = []
    seen = set()
    for m in mutations:
        m = m.strip()
        if m != '' and m not in seen:
            seen.add(m)
            uniq_mut.append(m)

    if not os.path.exists(pars_args.result_folder):
        os.makedirs(pars_args.result_folder)

    mutations_csv_path = pars_args.result_folder + '/' + model_name + "_mutations.csv"
    pd.DataFrame({'mutations': uniq_mut}).to_csv(mutations_csv_path, index=False)
    print("Saved mutations list: " + mutations_csv_path + "  num=" + str(len(uniq_mut)))

    num_samples = 20000
    batch_size  = 256

    if not hasattr(model, 'compute_evol_indices'):
        raise AttributeError("Your DERIVE_model has no function: compute_evol_indices. Please replace this call with your model's test function name.")

    print("Start: compute evol indices")
    with torch.no_grad():
        res = model.compute_evol_indices(
                msa_data=GISAID,
                list_mutations_location=mutations_csv_path,
                num_samples=num_samples,
                batch_size=batch_size
        )

    if not isinstance(res, (list, tuple)) or len(res) < 2:
        raise ValueError("compute_evol_indices() returned unexpected format. Expect (list_valid_mutations, evol_indices, ...).")

    list_valid_mutations = res[0]
    evol_indices = res[1]

    out_csv = pars_args.result_folder + '/' + model_name + "_" + str(num_samples) + "_samples_evol_indices.csv"
    pd.DataFrame({
        'mutations': [str(x) for x in list_valid_mutations],
        'evol_indices': np.asarray(evol_indices, dtype=float)
    }).to_csv(out_csv, index=False)

    print("End: compute evol indices")
    print("Saved: " + out_csv)
