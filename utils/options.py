import argparse
import logging
from operator import truediv
import os
import random
import socket

import numpy as np
import torch

from utils import dist_utils

logger = logging.getLogger()


def add_data_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--do_lower_case',
        default=True,
        type=bool,
        help=('Whether to lower case the input text. True for uncased models, '
              'False for cased models.'))

def add_model_params(parser: argparse.ArgumentParser):
    """Common parameters to initialize an encoder-based model."""
    
    parser.add_argument(
        '--model_cfg',
        default='config/config_768_layer6.json',
        type=str,
        help='Path of the pre-trained model.')
    parser.add_argument(
        '--checkpoint_file',
        default=None,
        type=str,
        help='Trained checkpoint file to initialize the model.')
    parser.add_argument(
        '--projection_dim',
        default=0,
        type=int,
        help='Extra linear layer on top of standard bert/roberta encoder.')
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=128,
        help='Max length of the encoder input sequence.')
    parser.add_argument(
        '--dropout',
        default=0.1,
        type=float,
        help='')
    parser.add_argument(
        '--num_token_types',
        default=20,
        type=int,
        help='Number of possiblen token types')
    parser.add_argument(
        '--ignore_token_type',
        default=True,
        type=bool,
        help='Whether to ignore token types or not')


def add_training_params(parser: argparse.ArgumentParser):
    """Common parameters for training."""
    parser.add_argument(
        '--data_dir',
        # default=None,
        default='data/synthetic',
        # default='data/mz4',
        # default='data/mz4-1k',
        # default='data/mz10',
        type=str,
        help='File pattern for the train set.')
    parser.add_argument(
        '--train_file',
        # default=None,
        default='train_set.json',
        type=str,
        help='File pattern for the train set.')
    parser.add_argument(
        '--dev_file',
        default='test_set.json',
        type=str,
        help='File pattern for the dev set.')
    parser.add_argument(
        '--vocab_file',
        default='data/synthetic/ontology.json',
        # default='data/mz4/ontology.json',
        # default='data/mz4-1k/ontology.json',
        # default='data/mz10/ontology.json',
        type=str,
        help='Path of the pre-trained model.')
    parser.add_argument(
        '--batch_size',
        default=64,
        # default=64,
        type=int,
        help='Amount of questions per batch.')
    parser.add_argument(
        '--dev_batch_size',
        type=int,
        default=512,
        help='amount of questions per batch for dev set validation.') 
    parser.add_argument(
        '--k',
        type=int,
        default=8,
        help='amount of questions per batch for dev set validation.') 
    parser.add_argument(
        '--add_task_1',
        action="store_true",
        # default=True,
        help='amount of questions per batch for dev set validation.')
    parser.add_argument(
        '--add_task_2',
        action="store_true",
        # default=True,
        help='amount of questions per batch for dev set validation.')
    parser.add_argument(
        '--add_task_3',
        action="store_true",
        # default=True,
        help='amount of questions per batch for dev set validation.')
    parser.add_argument(
        '--add_task_4',
        action="store_true",
        # default=True,
        help='amount of questions per batch for dev set validation.')
    parser.add_argument(
        '--add_middle_task_1',
        action="store_true",
        # default=True,
        help='amount of questions per batch for dev set validation.')
    parser.add_argument(
        '--add_middle_task_2',
        action="store_true",
        help='amount of questions per batch for dev set validation.')
    parser.add_argument(
        '--only_gen',
        action="store_true",
        help='amount of questions per batch for dev set validation.')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='random seed for initialization and dataset shuffling.')
    parser.add_argument(
        '--adam_eps',
        default=1e-8,
        type=float,
        help='Epsilon for Adam optimizer.')
    parser.add_argument(
        '--adam_betas',
        default='(0.9, 0.999)',
        type=str,
        help='Betas for Adam optimizer.')
    parser.add_argument(
        '--max_grad_norm',
        default=1.0,
        type=float,
        help='Max gradient norm.')
    parser.add_argument(
        '--max_turn',
        default=20,
        type=int,
        help='Number of steps during decoding.')
    parser.add_argument(
        '--log_batch_step',
        default=1000,
        type=int,
        help='Number of steps to log during training.')
    parser.add_argument(
        '--train_rolling_loss_step',
        default=1000,
        type=int,
        help='Number of steps of interval to save training loss.')
    parser.add_argument(
        '--weight_decay',
        default=0.01,
        type=float,
        help='Weight decay for optimizer.')
    parser.add_argument(
        '--learning_rate',
        default=3e-5,
        type=float,
        help='Learning rate.')
    parser.add_argument(
        '--min_probability', 
        default= 0.009, 
        type=float, 
        required=False,
        help='the minimum probability of inquiring implicit symptom.')
    parser.add_argument(
        '--warmup_steps',
        default=20,
        type=int,
        help='Linear warmup over warmup_steps.')
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of update steps to accumulate before updating parameters.')
    parser.add_argument(
        '--num_train_epochs',
        default=100,
        type=float,
        help='Total number of training epochs to perform.')
    parser.add_argument(
        '--eval_step',
        default=1000,
        type=int,
        help='Batch steps to run validation and save checkpoint.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='saved_checkpoints/synthetic/config_768_layer6',
        help='Output directory for checkpoints.')
    parser.add_argument(
        '--result_output_path',
        type=str,
        default='result.json',
        help="the path of saving the result of testing")
    parser.add_argument(
        '--log_dir',
        type=str,
        default='log/dxy/config_512_layer4/log',
        # default='log/mz4/t5/log',
        # default='log/mz4-1k/t5/log',
        # default='log/mz10/t5/log',
        help='Output directory for checkpoints.')
    parser.add_argument(
        '--model_recover_dir',
        type=str,
        default='',
        help='Output directory for checkpoints.')
    parser.add_argument(
        '--model_recover_path',
        type=str,
        default=None,
        # default='dataset/Chunyu/exp_t5_small_chinese_stage_all/model.35.bin',
        help='Output directory for checkpoints.')
    parser.add_argument(
        '--inference_only',
        action='store_true',
        default=False,
        help='Inference only.')
    parser.add_argument(
        '--prediction_results_file',
        default='dataset/Chunyu/exp/dev_infer_predictions.json',
        type=str,
        help='Path to a file to write prediction results to')

def add_cuda_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='The parameter for distributed training.')
    parser.add_argument(
        '--fp16',
        # default=True,
        default=False,
        type=bool,
        help='Whether to use 16-bit float precision instead of 32-bit.')
    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O2',
        help=('For fp16: Apex AMP optimization level selected.'
              'See details at https://nvidia.github.io/apex/amp.html.'))


def get_encoder_checkpoint_params_names():
    return [
        'do_lower_case',
        'pretrained_model_cfg',
        'projection_dim',
        'max_seq_len',
    ]


def get_encoder_params_state(args):
    """
    Selects the param values to be saved in a checkpoint, so that a trained
    model faile can be used for downstream tasks without the need to specify
    these parameter again.

    Return: Dict of params to memorize in a checkpoint.
    """
    params_to_save = get_encoder_checkpoint_params_names()

    r = {}
    for param in params_to_save:
        r[param] = getattr(args, param)
    return r


def set_encoder_params_from_state(state, args):
    if not state:
        return
    params_to_save = get_encoder_checkpoint_params_names()

    override_params = [
        (param, state[param])
        for param in params_to_save
        if param in state and state[param]
    ]
    for param, value in override_params:
        if param == "pretrained_model_cfg":
            continue
        if hasattr(args, param):
            if dist_utils.is_local_master():
                logger.warning(
                    f'Overriding args parameter value from checkpoint state. '
                    f'{param = }, {value = }')
        setattr(args, param, value)
    return args


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_args_gpu(args):
    """
    Setup arguments CUDA, GPU & distributed training.
    """

    world_size = os.environ.get('WORLD_SIZE')
    world_size = int(world_size) if world_size else 1
    args.distributed_world_size = world_size
    local_rank = args.local_rank
  
    if local_rank == -1:
        # Single-node multi-gpu (or cpu) mode.
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        device = torch.device(device)
        n_gpu = args.n_gpu = torch.cuda.device_count()
    else: 
        # Distributed mode.
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        # set up the master's ip address so this child process can coordinate
        torch.distributed.init_process_group(
            backend='nccl',
            rank=args.local_rank,
            world_size=world_size)
        n_gpu = args.n_gpu = 1
    args.device = device

    if dist_utils.is_local_master():
        logger.info(
            f'Initialized host {socket.gethostname()}'
            f'{local_rank = } {device = } {n_gpu = } {world_size = }'
            f'16-bits training: {args.fp16}')
