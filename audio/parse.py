import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=True, help='does not use GPU')
    parser.add_argument('--dir', type=str, default='MELD_features/', help='dataset directory')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=8, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=2, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--seed', type=int, default=100, metavar='seed', help='seed')
    parser.add_argument('--mu', type=float, default=0, help='class_weight_mu')
    parser.add_argument('--verbose', action='store_true', default=True, help='verbose')
    
    return parser.parse_args()