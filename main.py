from pathlib import Path
import torch
from Args import Args
import configs
import sys
import torch.backends.cudnn as cudnn
from noise_filter.noise_classes import AddGaussianNoise
import utils
from run import train_an_agent_full, train_first_task

import viz

def main(args:Args):

    args.exp_info_long, args.exp_info_short, args.exp_info_base = utils.get_exp_info(args)

    args.dataset_results_path = utils.get_dataset_results_path(args.dataset, args.perm, args.extra_info)
    args.filepath_main, args.filepath_plot, args.filepath_acc_per_task = utils.get_output_filepath(args.dataset_results_path, args.exp_info_long)
    args.store_path = Path.cwd() / 'store'

    model_subdir_path = args.store_path / args.model_subdir

    if not Path.exists(args.store_path):
        Path.mkdir(args.store_path)
    if not Path.exists(model_subdir_path):
        Path.mkdir(model_subdir_path)
    args.store_path = model_subdir_path
    args.train_d, args.test_d = args.cf.get_train_test_dataset()

    print(f'Dataset: {args.dataset}, # train: {len(args.train_d.to_taskset())}, # test: {len(args.test_d.to_taskset())}')

    if args.output_to_file:
        utils.initialise_file(args.dataset_results_path, args.filepath_main, n_tasks=args.n_tasks)

    overall_acc_per_task_all = {}

    for i in range(args.start_seed, args.start_seed + args.repeat):
        args.seed = i
        utils.seed_everything(args.seed)

        overall_acc_per_task_all[args.seed] = {}

        args.class_order = utils.get_classes_permuted(args)
        utils.data_setup(args)

        print('\n[Phase 3] : Model training')  

        args.current_net= setup_model(args)
        args.old_net = None
        args.agent = 'baseline'
        overall_acc_list = train_first_task(args)

        #since both bican and il2m require exemplars for bias correction, do not include baseline method when bias correction
        #option is enabled
        all_agents = ['baseline', *[k for k in args.agents]] if not args.correct_bias else args.agents
       
        for agent in all_agents:
            args.agent = agent
            print(f'\n| | Training using agent: {args.agent}')
            args.old_net = None
            args.current_net= setup_model(args)

            args.memory_path = utils.get_memory_path(args, args.agent, args.exp_info_long)

            overall_acc_list, overall_acc_per_task = train_an_agent_full(args)
            utils.process_overall_acc(args, overall_acc_list,  agent)
            overall_acc_per_task_all[args.seed][args.agent] = overall_acc_per_task

    if args.output_to_file:
        print(f'Saving results...')
        utils.load_saved_results(args, args.filepath_main)
        viz.output_plot(args.filepath_main, args.filepath_plot, args.exp_info_long)
        
        import pickle as pkl
        with open(args.filepath_acc_per_task, 'wb') as f:
          pkl.dump(overall_acc_per_task_all, f)  
    
    print(f'Experiment completed')
 
def setup_model(args:Args):
    net = args.model_from_config
        
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    return net
                    
if __name__ == '__main__':
    ALL_AGENTS = ['core_high', 'random', 'herding', 'core_low', 'core_high_all', 'load_from_file']
    AGENTS = ['core_high', 'random', 'herding']
    # AGENTS = ['random']
    # AGENTS = ['load_from_file']
    # AGENTS = ['random', 'core_high', 'herding_icarl', 'core_low']

    config = {'CIFAR100':configs.config_cifar100, 'TinyImageNet200': configs.config_tinyimagenet200}

    args = utils.get_args(sys.argv[1:])
    cf = config[args.dataset]

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    args.agents = AGENTS

    args.start_epoch = cf.start_epoch

    args.batch_size = cf.batch_size

    args.optim_type = cf.optim_type
    args.lr = cf.lr
    args.num_workers = cf.num_workers
    args.net_type = cf.net_type
    args.n_classes = cf.n_classes
    args.cf = cf

    args.num_epochs_non_incremental  = cf.num_epochs_non_incremental
    args.num_epochs_incremental = cf.num_epochs_incremental
    args.model_from_config = cf.get_model()
    args.patience_incremental = cf.patience_incremental
    args.patience_non_incremental = cf.patience_non_incremental
    args.transform_train, args.transform_test = args.cf.get_transformations()
    args.n_tasks = (args.n_classes - args.initial_increment) // args.increment + 1
    args.memory = args.increment * (args.n_tasks - 1) * args.memory_per_class
    args.n_classes_incremental = args.n_classes - args.initial_increment

    #add gaussian noise to the transformation list
    if args.noise_type == 'gaussian':
        noise_transform = [AddGaussianNoise(args.noise_mean, args.noise_var)]
        trsf = args.transform_train if args.transform_train is not None else []
        args.transform_train = noise_transform + trsf

    #includes initial non-incremental task
    args.n_tasks = (args.n_classes - args.initial_increment) // args.increment + 1

    _dataset = args.dataset

    args.model_subdir = f'{_dataset}_m{args.memory_per_class}pc_perm{args.perm}'

    main(args)

