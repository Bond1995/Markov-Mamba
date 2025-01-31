import torch

def parse_args(base_parser, args, namespace):
    parser = base_parser
    # General training params
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--acc_steps', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--iterations', default=1000, type=int)
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--warmup_percent', default=0.02, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)
    parser.add_argument('--scheduler', default='cos', choices=['linear', 'cos', 'none'])
    parser.add_argument('--opt', default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--results_base_folder', default="./exps", type=str)
    parser.add_argument('--grad_clip', default=1.0, type=float) 
    # Dataset params
    parser.add_argument('--dataset', default='markov', choices=['markov'])
    parser.add_argument('--vocab_size', default=2, type=int)
    # Model params
    parser.add_argument('--model', default='base', choices=['base'])
    parser.add_argument('--d_model', default=8, type=int)
    parser.add_argument('--d_state', default=8, type=int)
    parser.add_argument('--d_conv', default=4, type=int)
    parser.add_argument('--expand', default=2, type=int)
    parser.add_argument('--nheads', default=1, type=int)
    parser.add_argument('--ngroups', default=1, type=int)
    parser.add_argument('--n_layer', default=1, type=int)
    parser.add_argument('--sequence_length', default=256, type=int)
    parser.add_argument('--dtype', default=torch.float32, type=torch.dtype)
    parser.add_argument('--bias', default=False, type=bool)
    parser.add_argument('--activation', default='relu', choices=['relu', 'silu'])
    parser.add_argument('--layernorm', action='store_true') # If True, adds layer norms
    parser.add_argument('--conv', action='store_true') # If True, adds convolution
    parser.add_argument('--conv_type', default='base', choices=['base', 'fixed', 'onlyx', 'onlyxb'])
    parser.add_argument('--conv_act', action='store_true') # If True, adds convolution activation in Mamba block
    parser.add_argument('--fix_conv', action='store_true') # If True, freezes the convolution layer
    parser.add_argument('--gate', action='store_true') # If True, adds gating in Mamba block and replaces MLP with GatedMLP
    parser.add_argument('--fix_A', action='store_true') # If True, fixes A = 1
    parser.add_argument('--no_mlp', action='store_true') # If True, removes the MLP layer
    parser.add_argument('--mlp_factor', default=4, type=int) # MLP multiplicative factor
    # logging params (wandb)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', default="bias-test", type=str)
    parser.add_argument('--wandb_run_prefix', default="none", type=str)
    parser.add_argument('--eval_seq_prefix', default="0", type=str)
    # Markov args
    parser.add_argument('--p', default=0.5, type=float)
    parser.add_argument('--q', default=0.5, type=float)
    parser.add_argument('--chain', default='random', choices=['switch', 'random', 'random-fixed'])
    parser.add_argument('--type', default='markov', choices=['markov', 'jump-markov'])
    parser.add_argument('--order', default=1, type=int)
    parser.add_argument('--initial', default='uniform', choices=['uniform', 'steady'])
    
    return parser.parse_args(args, namespace)
