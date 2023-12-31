import argparse
import pprint

def parse_opt():
    parser = argparse.ArgumentParser()
    pp = pprint.PrettyPrinter(depth=4) # PrettyPrinter is used to print the arguments in a clean way

    parser.add_argument('--no-cuda', action='store_true', default=True, help='does not use CUDA')
    parser.add_argument('--dir', type=str, default='./MELD_features/', help='dataset directory (for .pkl file)')
    parser.add_argument('--n-classes', type=int, default=7, help='number of classes')
    parser.add_argument('--val-split', type=float, default=0, help='validation split')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')

    parser.add_argument('--loss-fn', type=str, default='masked_nll', help='loss function (masked_nll or unmaksed_weighted_nll or masked_mse)')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer (adam or sgd or rmsprop)')

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=3e-4, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=30, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights (true or false)')
    parser.add_argument('--mu', type=float, default=0, help='class weight (mu)')

    parser.add_argument('--seed', type=int, default=42, metavar='seed', help='seed')
    
    parser.add_argument('--feature_type', type=str, default='audio', help='features (text or audio or multimodal)')
    parser.add_argument('--attention', type=str, default='simple', help='attention type (simple or general or general2 or concat or dot)')
    
    parser.add_argument('--verbose', action='store_true', default=True, help='verbose (true or false)')

    args = parser.parse_args("")
    
    pp.pprint(vars(args))
    
    # parser.print_help()
    
    return args