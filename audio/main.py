import torch
import time
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
import warnings
import pandas as pd

# Custom local imports
from dataloader import get_MELD_loaders
from parse import parse_opt
from trainer import build_model, trainer
from config import *

if __name__ == '__main__':
    
    warnings.filterwarnings("ignore") # To ignore the warnings from sklearn.metrics.classification_report
    
    args = parse_opt()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print()
    print('#'*100)
    print()
    model, optimizer, loss_function = build_model(D_m, D_q, D_g, D_r, D_e, D_h, args)

    train_loader, valid_loader, test_loader = get_MELD_loaders(args.dir + 'MELD_features_raw.pkl',
                                                               args.n_classes,
                                                               valid=args.val_split,
                                                               batch_size=args.batch_size,
                                                               num_workers=args.num_workers,)


    print()
    print('#'*100)
    
    print('\nTRAINING PHASE:\n')
    model, metrics, best_label, best_pred, best_mask, best_attn, test_class_report = \
        trainer(args, model, train_loader, valid_loader, test_loader, optimizer, loss_function)

    train_losses = metrics['train']['train_losses']
    train_fscores = metrics['train']['train_fscores']
    train_accs = metrics['train']['train_accs']
    
    val_losses = metrics['val']['val_losses']
    val_fscores = metrics['val']['val_fscores']
    val_accs = metrics['val']['val_accs']
   
    test_losses = metrics['test']['test_losses']
    test_fscores = metrics['test']['test_fscores']
    test_accs = metrics['test']['test_accs']
   
    best_fscore = metrics['best']['best_fscore']
    best_loss = metrics['best']['best_loss']
    best_acc = metrics['best']['best_acc']

    print()
    print('#'*100)
    print('\nTESTING PHASE:\n')
    
    print('[Metrics]\t', end='')
    print('Fscore: {:.3f}\tLoss: {:.3f}\tAccuracy: {:.3f}%\n'.format(best_fscore, best_loss, accuracy_score(best_label,best_pred,sample_weight=best_mask)*100))
    
    print('\nClassification report:')
    print(classification_report(best_label,best_pred,sample_weight=best_mask,digits=4))
    
    print('\nConfusion matrix:')
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))

    print()
    print('#'*100)
    print('\nSAVING MODEL AND METRICS:')
    model_name = 'models/EmoNet_'+str(int(round(best_fscore,0)))+'.pt'
    torch.save(model.state_dict(), args.dir + model_name)
    print('\nModel saved to {}.'.format(model_name))
    
    # Store all training metrics per epoch in a dataframe
    train_metrics = pd.DataFrame(metrics['train'])
    test_metrics = pd.DataFrame(metrics['test'])
    val_metrics = pd.DataFrame(metrics['val'])
    metrics = pd.concat([train_metrics, val_metrics, test_metrics], axis=1)
    metrics.to_csv(args.dir + 'logs/metrics.csv', index=False)
    print('Metrics saved to {}.'.format(args.dir + 'logs/metrics.csv'))
    print('#'*100)