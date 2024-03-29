import argparse


def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="fruit", help='dataset')
    parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
    parser.add_argument('--beta', type=float, default=0.2, help='beta')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--epoch', type=int, default=50, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.000, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    args = parser.parse_args()
    return args
