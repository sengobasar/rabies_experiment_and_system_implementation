import argparse
import json
import logging
from datetime import datetime as dt


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='For train')
    parser.add_argument('--GCN_pretrained_initialization', action='store_true', help='Initialize the GCN from a pretrained graph model')
    parser.add_argument('--d0_similarity_threshold', type=float, default=0.01, help='Similarity threshold for the second level sequence re-weighting')
    parser.add_argument('--eta_balancing_strength', type=float, default=1, help='balancing strength for the second level sequence re-weighting')
    parser.add_argument('--z_dim_latent_dimension', type=int, default=64, help='latent space dimension')
    parser.add_argument('--K_flow_iterations', type=int, default=32, help='iterations for the factorized normalizing flow')
    parser.add_argument('--alpha_gamma_larger_KL', type=int, default=2, help='regularization scales for both normalizing flow prior term and disentanglement term')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for model initialization')

    parser.add_argument('--GISAID_data_folder', type=str, default='./data/GISAID', help='Folder where GISAID data are stored')
    parser.add_argument('--figs_ForGISAIDData_folder', type=str, default='./figs/GISAID', help='Folder where figs related to GISAID data are stored')
    parser.add_argument('--GISAID_data_name', type=str, default='/all_seqs_scores', help='Name of GISAID with /....')
    parser.add_argument('--MSA_data_folder', type=str, default='./data/MSA', help='Folder where MSAs are stored')
    parser.add_argument('--MSA_data_name', type=str, default='./data/MSA/P0DTC2_sc0.5_cc0.3_b0.1_pre2020.a2m', help='Name of MSAs with /....a2m')
    parser.add_argument('--result_folder', type=str, default='./results', help='Output location of computed evol indices')
    parser.add_argument('--result_folder_ForTestWhenTrain', type=str, default='./results', help='Output location for test results during pretrained-graph training')
    parser.add_argument('--training_logs_location', default='./logs/', type=str, help='Location of DERIVE model parameters')
    parser.add_argument('--model_parameters_location', default='./DERIVE/default_model_params_new.json', type=str, help='Location of DERIVE model parameters')
    parser.add_argument('--model_checkpoint_location', type=str, default='./checkpoints', help='Location where model checkpoints will be stored')
    parser.add_argument('--CUDA_num', default='0', type=str, help='CUDA device number')

    pars_args   = parser.parse_args()

    # Keep --help usable on systems where training-only dependencies are not installed.
    try:
        import torch
        import lib.dist as dist
        from generate_candidate_set import GISAID_data_class, MSA_data_class
        from model import DERIVE_model
        from torch_geometric.nn import GCNConv
        from lib.flows import FactorialNormalizingFlow
    except ModuleNotFoundError as exc:
        if exc.name == 'torch_geometric':
            raise SystemExit(
                'Training requires PyTorch Geometric. Install a version compatible '
                'with your PyTorch build (see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).'
            ) from exc
        raise
    with_pretrained_graph = pars_args.GCN_pretrained_initialization
    d0          = pars_args.d0_similarity_threshold
    eta         = pars_args.eta_balancing_strength
    z_dim       = pars_args.z_dim_latent_dimension
    K_flow      = pars_args.K_flow_iterations
    alpha_gamma = pars_args.alpha_gamma_larger_KL
    Mydevice    = torch.device("cuda:" + pars_args.CUDA_num  if torch.cuda.is_available() else "cpu")

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

    print("Start: MSA data for training")
    MSA    = MSA_data_class(pars_args_MSA_data_folder=pars_args.MSA_data_folder,
                            pars_args_MSA_data_name=pars_args.MSA_data_name)
    MSA.compute_OneHotAndChemicalEncoding_ForTrain()
    MSA.compute_weights(use_weights=1, d0=d0, eta=eta)
    print("End: MSA data for training")

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

    if with_pretrained_graph:       
        ExtendedLeadingWHOVOC_pd = 0
        with_graph_res = False
        model_params["training_parameters"]['num_training_steps']     = 1000
        model_params["encoder_parameters"]['with_graph_res']  = with_graph_res
        model_name = (pars_args.MSA_data_name[1:-4] + "_") + \
                    dt.now().strftime("%Y-%m-%d---%H-%M-%S")

        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()

        file_handler = logging.FileHandler(pars_args.training_logs_location + model_name + '.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info('Dictionary content: {}'.format(json.dumps(model_params, indent=4)))

        model = DERIVE_model.DERIVE_model(
                        model_name=model_name,
                        training_data=MSA,
                        graph_data=graph_data_dict,
                        SASA_data=SASA_data_np,
                        WCN_data=WCN_data_np,
                        SASA_std_data=SASA_std_data_np,
                        WCN_std_data=WCN_std_data_np,
                        encoder_parameters=model_params["encoder_parameters"],
                        decoder_parameters=model_params["decoder_parameters"],
                        random_seed=pars_args.seed,
                        pars_args_result_folder_ForTestWhenTrain=pars_args.result_folder_ForTestWhenTrain,
                        mylogger=logger,
                        device=Mydevice
                        )
        for name, param in model.named_parameters():
            print(name, param.size())
        model = model.to(Mydevice)

        print("Starting to train model: " + model_name)
        model.train_model(training_parameters=model_params["training_parameters"])

        target_layers = []
        for name, module in model.named_modules():
            if isinstance(module, GCNConv):  
                target_layers.append(f"{name}.lin.weight")
                if hasattr(module, 'bias') and module.bias is not None:
                    target_layers.append(f"{name}.bias")

        print("Target layers:", target_layers)
        GCN_state_dict = {name: param for name, param in model.state_dict().items() if name in target_layers}

        # state_dict_to_save = {name: param for name, param in model.state_dict().items() if name in target_layers}
        # torch.save(state_dict_to_save, 'target_layers.pth')
        # saved_state_dict = torch.load('target_layers.pth')


    ######################################################################################################################################################
    # with_graph_res = True
    ######################################################################################################################################################
    model_name = (pars_args.MSA_data_name[1:-4] + "_") + \
                dt.now().strftime("%Y-%m-%d---%H-%M-%S")

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(pars_args.training_logs_location + model_name + '.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info('Dictionary content: {}'.format(json.dumps(model_params, indent=4)))
    model = DERIVE_model.DERIVE_model(
                    model_name=model_name,
                    training_data=MSA,
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

    print("Starting to train model: " + model_name)
    if with_pretrained_graph: 
        current_state_dict = model.state_dict()
        for name, param in GCN_state_dict.items():
            if name in current_state_dict:
                current_state_dict[name].copy_(param)
        model.load_state_dict(current_state_dict, strict=False)

        for name, param in model.named_parameters():
            if name in target_layers:
                print(f"Loaded {name}: {param}")

    model.train_model(training_parameters=model_params["training_parameters"])

    # print("Saving model: " + model_name)
    # filename_prefix_path = './results/DERIVE_parameters' + '/' + model_name
    # if not os.path.exists(filename_prefix_path):
    #     os.makedirs(filename_prefix_path)
    # model.save(model_checkpoint=  filename_prefix_path + '/' "step" + str(model_params["training_parameters"]['num_training_steps']),
    #             encoder_parameters=model_params["encoder_parameters"],
    #             decoder_parameters=model_params["decoder_parameters"],
    #             training_parameters=model_params["training_parameters"]
    #             )


    print('a')
