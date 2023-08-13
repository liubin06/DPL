import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Log Information
    parser.add_argument('--log_root', default=r'.\result')
    parser.add_argument('--log', type=bool, default=True)

    # Random Seed
    parser.add_argument('--seed', type=int, default=2022)

    # Training Args
    parser.add_argument('--encoder', default='MF', help='MF LightGCN')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--l2', type=float, default=1e-6, help='l2 penalty')  # 1M必须设定1e-6
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')  #
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate') # 0.1 1
    parser.add_argument('--lr_dc_epoch', type=list, default=[100], help='the epoch which the learning rate decay')  # 20 60 80
    parser.add_argument('--LOSS', default='BCL', help='choose loss function [BPR, Info_NCE, DCL, HCL, DPL]')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dim', type=int, default=32, help='dimension of vector')
    parser.add_argument('--hop', type=int, default=1, help='LightGCN layers')
    parser.add_argument('--temperature', type=float, default=1, help='temperature')
    parser.add_argument('--tau_plus', type=float, default=0.04, help='tau_plus')

    # Dataset
    parser.add_argument('--dataset', default='1M', help='dataset')  # 100k yahoo 1M gowalla yelp2018

    # Sampling Args
    parser.add_argument('--M', type=int, default=5, help='number of positive samples for each user')
    parser.add_argument('--N', type=int, default=5, help='number of negative samples for each user')

    # Evaluation Arg
    parser.add_argument('--topk', type=list, default=[5, 10, 20], help='length of recommendation list')

    return parser.parse_args()


