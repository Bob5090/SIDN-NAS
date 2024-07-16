import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from time import perf_counter
import importlib
import torch
from pytorch_version.utils import train_DDP

torch.autograd.set_detect_anomaly(True)


def argparsing():
    parser = argparse.ArgumentParser(description='SIDN training / Retraining')
    parser.add_argument('--SIDN', action='store_true', default=False, help='SIDN training / Retraining?')
    parser.add_argument('--epochs', default=250, type=int, help='trining epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=3e-5, type=float, help='learning rate decay')
    parser.add_argument('--in_channel', default=3, type=int, help='model input channel number')
    parser.add_argument('--num_class', default=1, type=int, help='model output channel number')
    parser.add_argument('--num_layers', default=5, type=int, help='model layers')
    parser.add_argument('--activate', default='relu', type=str, help='model activation func')
    parser.add_argument('--multiplier', default=2, type=int, help='parameter multiplier')
    parser.add_argument('--iter', default=3, type=int, help='recurrent iteration')
    parser.add_argument('--train_data', default='data/monuseg/train', type=str, help='data path')
    parser.add_argument('--valid_data', default='data/monuseg/test', type=str, help='data path')
    parser.add_argument('--data_name', default='monuseg', type=str, help='data name')
    parser.add_argument('--exp', default='1', type=str, help='experiment number')
    parser.add_argument('--gene', default=None, type=str, help='searched gene file')
    parser.add_argument('--evaluate_only', action='store_true', help='evaluate only?')
    parser.add_argument('--save_result', action='store_true', default=True, help='save results to exp folder?')
    parser.add_argument('--model_path', default=None, type=str, help='path to model checkpoint')
    parser.add_argument('--valid_dataset', default='monuseg', choices=['monuseg', 'tnbc', 'MRI'], type=str,
                        help='which dataset to validate?')
    parser.add_argument('--backend', default='pytorch', choices=['pytorch'], type=str, help='support pytorch only?')
    parser.add_argument('--dataset', default='nuclei', type=str, help='which dataset to run?')
    parser.add_argument('--save_gene', default=None, type=str, help='path to save genes')
    parser.add_argument('--with_att', action='store_true', default=False, help='do you want attention?')
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--is_newBlock', default=True, type=bool)
    parser.add_argument('--is_newSkip', default=True, type=bool)
    parser.add_argument('--is_parallel', default=False, type=bool)
    parser.add_argument('--is_ddp', default=False, type=bool)
    parser.add_argument('--is_skipATT', default=True, type=bool)
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed training.')
    args = parser.parse_args()

    print()
    print()
    print(args)  # print command line args

    return args


def main(args, CORE):
    # path verification
    if args.model_path is not None:
        if os.path.isfile(args.model_path):
            print('Model path has been verified.')
        else:
            print('Invalid model path! Please specify a valid model file. Program terminating...')
            exit()

    # pipeline starts
    if not args.evaluate_only:
        if args.is_parallel:
            CORE.train_parallel(args)
        elif args.is_ddp:
            world_size = 2
            print('starting ddp...')
            torch.multiprocessing.spawn(train_DDP, args=(args, world_size),
                                        nprocs=world_size, join=True)
            print('start ddp sucessed')
        elif args.valid_dataset == 'MRI':
            CORE.train_MRI(args)
        else:
            CORE.train(args)
    CORE.evaluate(args)


if __name__ == '__main__':
    # parse command line args
    args = argparsing()

    # import dependencies
    CORE = importlib.import_module(args.backend + '_version')
    start = perf_counter()
    main(args, CORE)
    cost = perf_counter() - start
    hour = cost // 3600
    minutes = (cost - hour * 3600) // 60
    print(f'Search end, cost {hour} hours, {minutes} minutes')
