import argparse
parser = argparse.ArgumentParser(description='RL')

parser.add_argument(
    '--run_name',
    default='sar',
    help='name for the run - prefix to log files')
parser.add_argument(
    '--gpu_device',       
    type=int, 
    default = int(0), 
    required = False, 
    help = 'visible device in CUDA')
parser.add_argument(
    '--algo',
    type=str,
    default='sar',
    help='augmentation type')

parser.add_argument(
    '--log_dir',
    default='result/logs',
    help='directory to save agent logs')
parser.add_argument(
    '--log_interval',
    type=int,
    default=1,
    help='log interval, one log per n updates')
parser.add_argument(
    '--save_dir',
    default='result/checkpoints',
    help='directory to save agent logs')
parser.add_argument(
    '--save_interval',
    type=int,
    default=1,
    help='save interval, one save per n update')
parser.add_argument(
    '--no_cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')

parser.add_argument(
    '--seed', 
    type=int, 
    default=1, 
    help='random seed')
parser.add_argument(
    '--num_env_steps',
    type=int,
    default=100000000,
    help='number of environment steps to train')
parser.add_argument(
    '--preempt',
    action='store_true',
    default=False,
    help='safe preemption: load the latest checkpoint with same args and continue training')

# Procgen Arguments.
parser.add_argument(
    '--env_name',
    type=str,
    default='starpilot',
    help='environment to train on')
parser.add_argument(
    '--num_processes',
    type=int,
    default=64,
    help='how many training CPU processes to use')
parser.add_argument(
    '--start_level',
    type=int,
    default=0,
    help='start level id for sampling Procgen levels')
parser.add_argument(
    '--num_levels',
    type=int,
    default=200,
    help='number of Procgen levels to use for training')
parser.add_argument(
    '--distribution_mode',
    default='easy',
    help='distribution of envs for procgen')

# PPO Arguments. 
parser.add_argument(
    '--lr', 
    type=float, 
    default=5e-4, 
    help='learning rate')
parser.add_argument(
    '--eps',
    type=float,
    default=1e-5,
    help='RMSprop optimizer epsilon')
parser.add_argument(
    '--alpha',
    type=float,
    default=0.99,
    help='RMSprop optimizer alpha')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.999,
    help='discount factor for rewards')
parser.add_argument(
    '--gae_lambda',
    type=float,
    default=0.95,
    help='gae lambda parameter')
parser.add_argument(
    '--entropy_coef',
    type=float,
    default=0.01,
    help='entropy term coefficient')
parser.add_argument(
    '--value_loss_coef',
    type=float,
    default=0.5,
    help='value loss coefficient (default: 0.5)')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.5,
    help='max norm of gradients)')
parser.add_argument(
    '--num_steps',
    type=int,
    default=256,
    help='number of forward steps in A2C')
parser.add_argument(
    '--ppo_epoch',
    type=int,
    default=3,
    help='number of ppo epochs')
parser.add_argument(
    '--num_mini_batch',
    type=int,
    default=8,
    help='number of batches for ppo')
parser.add_argument(
    '--clip_param',
    type=float,
    default=0.2,
    help='ppo clip parameter')
parser.add_argument(
    '--hidden_size',
    type=int,
    default=256,
    help='state embedding dimension')

# Image aug Arguments.
parser.add_argument(
    '--aug_type',
    type=str,
    default='identity',
    help='augmentation type')
parser.add_argument(
    '--aug_extra_shape', 
    type=int, 
    default=0, 
    help='increase image size by')
parser.add_argument(
    '--image_pad', 
    type=int, 
    default=12, 
    help='increase image size by')
parser.add_argument(
    '--aug_coef', 
    type=float, 
    default=0.1, 
    help='coefficient on the augmented loss')
    
# SAR Arguments.
parser.add_argument(
    '--adv_coef', 
    type=float,
    default=0.1, 
    help='KL-divergence loss on actor-critic coefficient.')
parser.add_argument(
    '--val_sim_coef', 
    type=float,
    default=1.0, 
    help='value of noised embedding to be similar to normal coefficient.')

# Additional experimentations
parser.add_argument(
    '--encoding',
    action='store_true',
    default=False,
    help='extract the represnetation features')