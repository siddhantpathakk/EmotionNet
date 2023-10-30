import torch
import time
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
import warnings
import json
import pandas as pd
# Custom local imports
from dataloader import create_class_weight, get_MELD_loaders
from losses import MaskedNLLLoss
from parse import parse_opt
from model import EmoNet
from trainer import train_or_eval_model


if __name__ == '__main__':
    
    warnings.filterwarnings("ignore") # To ignore the warnings from sklearn.metrics.classification_report
    
    args = parse_opt()
    if args.verbose:
        print(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.verbose:
        if args.cuda:
            print('Running on GPU')
        else:
            print('Running on CPU')
    
    n_classes = 7
    
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    
    global D_s
    D_m = 300
    D_g = D_q = D_r = D_e = 150
    D_h = 100
    
    global seed
    seed = args.seed
    
    model = EmoNet(D_m, D_q, D_g, D_r, D_e, D_h, n_classes=n_classes, dropout=args.dropout)
    
    if args.verbose:
        print('EmoNet model.')
        print(model)
    
    if cuda:
        model.cuda()
    
    if args.class_weight:
        if args.mu > 0:
            loss_weights = torch.FloatTensor(create_class_weight(args.mu))
        else:
            loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696, 0.84847735, 5.42461417, 1.21859721])
        
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    
    
    if args.verbose:
        print(f'Loss function:\t{loss_function.__class__.__name__}')
        print(f'Optimizer:\t{optimizer.__class__.__name__}')
        
    train_losses, train_fscores, train_accs = [], [], []
    test_losses, test_fscores, test_accs = [], [], []
    val_losses, val_fscores, val_accs = [], [], []
    
    best_loss, best_label, best_pred, best_mask = None, None, None, None


    train_loader, valid_loader, test_loader = get_MELD_loaders(args.dir + 'MELD_features_raw.pkl',
                                                               n_classes,
                                                               valid=0.2,
                                                               batch_size=batch_size,
                                                               num_workers=0)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None

    print('\nTraining started..')
    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _,_,_,train_fscore,_,_= train_or_eval_model(model=model, loss_function=loss_function,dataloader=train_loader, epoch=e, optimizer=optimizer, train=True, cuda=cuda)
        valid_loss, valid_acc, _,_,_,val_fscore,_, _= train_or_eval_model(model=model, loss_function=loss_function, dataloader=valid_loader, epoch=e,cuda=cuda)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions, test_class_report = train_or_eval_model(model=model, loss_function=loss_function, dataloader=test_loader, epoch=e, train=False, cuda=cuda)


        train_losses.append(train_loss)
        train_fscores.append(train_fscore)
        train_accs.append(train_acc)
        
        test_losses.append(test_loss)
        test_fscores.append(test_fscore)
        test_accs.append(test_acc)
        
        val_losses.append(valid_loss)
        val_fscores.append(val_fscore)
        val_accs.append(valid_acc)
        

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore, best_loss, best_label, best_pred, best_mask, best_attn = test_fscore, test_loss, test_label, test_pred, test_mask, attentions

        print('Epoch:[{}]/[{}]\ttrain_loss:{:.3f}\ttrain_acc:{:.3f}\ttrain_fscore:{:.3f}\tval_loss:{:.3f}\tval_acc:{:.3f}\tval_fscore:{:.3f}\ttest_loss:{:.3f}\ttest_acc:{:.3f}\ttest_fscore:{:.3f}\ttime:{:.2f} sec'.\
                format(e+1, n_epochs, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,\
                        test_loss, test_acc, test_fscore, round(time.time()-start_time,2)))
        # print (test_class_report)

    print('\nTest performance..')
    print('Fscore: {} accuracy: {}%'.format(best_fscore,
                                    round(accuracy_score(best_label,best_pred,sample_weight=best_mask)*100,2)))
    print(classification_report(best_label,best_pred,sample_weight=best_mask,digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
    
    model_name = 'models/EmoNet_'+str(int(round(best_fscore,0)))+'.pt'
    torch.save(model.state_dict(), args.dir + model_name)
    print('\nModel saved to {}.'.format(model_name))
    
    # Store all training metrics per epoch in a dataframe
    train_metrics = pd.DataFrame({'train_loss': train_losses, 'train_fscore': train_fscores, 'train_acc': train_accs})
    test_metrics = pd.DataFrame({'test_loss': test_losses, 'test_fscore': test_fscores, 'test_acc': test_accs})
    val_metrics = pd.DataFrame({'val_loss': val_losses, 'val_fscore': val_fscores, 'val_acc': val_accs})
    metrics = pd.concat([train_metrics, val_metrics, test_metrics], axis=1)
    metrics.to_csv(args.dir + 'logs/metrics.csv', index=False)
    print('\nMetrics saved to {}.'.format(args.dir + 'logs/metrics.csv'))
    