from audio.dataloader import create_class_weight, get_MELD_loaders
from audio.losses import MaskedNLLLoss
from audio.parse import parse_opt
from audio.model import EmoNet
from audio.trainer import train_or_eval_model

import torch
import time
import numpy as np



if __name__ == '__main__':
    args = parse_opt()
    
    print(args)
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
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
    
    model = EmoNet(D_m, D_q, D_g, D_r, D_e, D_h, 
                   n_classes=n_classes, 
                   residual=args.residual, 
                   norm=args.norm, 
                   dropout=args.dropout)
    
    print('EmoNet model.')
    
    print(model)
    
    if cuda:
        model.cuda()
    
    if args.class_weight:
        if args.mu > 0:
            loss_weights = torch.FloatTensor(create_class_weight(args.mu))
        else:
            loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696, 
                0.84847735, 5.42461417, 1.21859721])
        
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    
    print(f'Loss function:\t{loss_function.__class__.__name__}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    
    print(f'Optimizer:\t{optimizer.__class__.__name__}')
    
    train_loader, valid_loader, test_loader = get_MELD_loaders(batch_size=batch_size,
                                                               num_workers=0)
    
    valid_losses, valid_fscores = [], []
    test_losses, test_fscores = [], []
    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        
        ## train step ##
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True)
        
        ## eval step ##
        valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
        
        ## test step ##
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)
            
        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        test_losses.append(test_loss)
        test_fscores.append(test_fscore)   
        
        print('epoch: {}, train_loss: {}, acc: {}, fscore: {}, valid_loss: {}, acc: {}, \
            fscore: {}, test_loss: {}, acc: {}, fscore: {}, time: {} sec'.format(e+1, train_loss, train_acc, train_fscore, \
                valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))


    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()

    score1 = test_fscores[0][np.argmin(valid_losses)]
    score2 = test_fscores[0][np.argmax(valid_fscores[0])]    
    scores = [score1, score2]
    scores = [str(item) for item in scores]
    
    print ('Test Scores: Weighted F1')
    print('@Best Valid Loss: {}'.format(score1))
    print('@Best Valid F1: {}'.format(score2))
 