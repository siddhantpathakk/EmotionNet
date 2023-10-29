import torch
import time
import numpy as np

# Custom local imports
from dataloader import create_class_weight, get_MELD_loaders
from losses import MaskedNLLLoss
from parse import parse_opt
from model import EmoNet
from trainer import train_or_eval_model


if __name__ == '__main__':
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
    
    train_loader, test_loader = get_MELD_loaders(batch_size=batch_size, num_workers=0)
    
    train_losses, train_fscores = [], []
    test_losses, test_fscores = [], []
    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for epoch in range(n_epochs):
        start_time = time.time()
        
        ## train step ##
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model=model, 
                                                                              loss_function=loss_function, 
                                                                              dataloader=train_loader, 
                                                                              epoch=epoch, 
                                                                              optimizer=optimizer,
                                                                              train=True, 
                                                                              cuda=cuda, 
                                                                              seed=seed)
        
        ## test step ##
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model=model,
                                                                                                             loss_function=loss_function, 
                                                                                                             dataloader=test_loader, 
                                                                                                             epoch=epoch, 
                                                                                                             train=False,
                                                                                                             cuda=cuda,
                                                                                                             seed=seed)
            
        train_losses.append(train_loss)
        train_fscores.append(train_fscore)
        test_losses.append(test_loss)
        test_fscores.append(test_fscore)   
        
        print('epoch: {}, train_loss: {}, acc: {}, fscore: {}, \
            test_loss: {}, acc: {}, fscore: {}, time: {} sec'.format(epoch+1, \
            train_loss, train_acc, train_fscore, \
                test_loss, test_acc, test_fscore, \
                    round(time.time()-start_time, 2)))


    train_fscores = np.array(train_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()

    score1 = test_fscores[0][np.argmin(train_losses)]
    score2 = test_fscores[0][np.argmax(train_fscores[0])]    
    scores = [score1, score2]
    scores = [str(item) for item in scores]
    
    print ('Test Scores: Weighted F1')
    print('@Best Train Loss: {}'.format(score1))
    print('@Best Train F1: {}'.format(score2))
 