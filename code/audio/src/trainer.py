import time
import numpy as np,random
import torch
torch.cuda.empty_cache()

from sklearn.metrics import classification_report, f1_score, accuracy_score

from src.losses import *
from src.model import EmotionNet
from src.dataloader import create_class_weight


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
          

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False, cuda=False, feature_type='audio'):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas_f, alphas_b, vids = [], [], []
    assert not train or optimizer!=None
    
    if cuda:
        model.to('cuda')
    else:
        model.to('cpu')
    
    if train:
        model.train()
    else:
        model.eval()
        
    for data in dataloader:
        
        if train:
            optimizer.zero_grad()
            
        textf, acouf, qmask, umask, label = [d.to('cuda') for d in data[:-1]] if cuda else data[:-1]

        # log_prob, alpha_f, alpha_b = model(acouf, qmask,umask) # seq_len, batch, n_classes
        
        if feature_type == "audio":
            log_prob, alpha_f, alpha_b = model(acouf, qmask,umask) # seq_len, batch, n_classes
        elif feature_type == "text":
            log_prob, alpha_f, alpha_b = model(textf, qmask,umask) # seq_len, batch, n_classes
        else:
            log_prob, alpha_f, alpha_b = model(torch.cat((textf,acouf),dim=-1), qmask,umask) # seq_len, batch, n_classes

        
        lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            optimizer.step()
        else:
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    avg_accuracy = round(accuracy_score(labels,preds,sample_weight=masks)*100,2)
    avg_fscore = round(f1_score(labels,preds,sample_weight=masks,average='weighted')*100,2)
    class_report = classification_report(labels,preds,sample_weight=masks,digits=4)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas_f, alphas_b, vids], class_report


def build_model(D_m, D_q, D_g, D_r, D_e, D_h, args):
    
    seed_everything(args.seed)
    model = EmotionNet(D_m, D_q, D_g, D_r, D_e, D_h, n_classes=args.n_classes, dropout=args.dropout, attention=args.attention)
    
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2)
    else:
        raise Exception('Unknown optimizer {}'.format(args.optimizer))
        
    if args.class_weight:
        if args.mu > 0:
            loss_weights = torch.FloatTensor(create_class_weight(args.mu))
        else:
            loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696, 0.84847735, 5.42461417, 1.21859721])
        
        
        if args.loss_fn == 'masked_nll':
            loss_function = MaskedNLLLoss(loss_weights.cuda() if args.cuda else loss_weights)
            
        elif args.loss_fn == 'unmasked_weighted_nll':
            loss_function = UnMaskedWeightedNLLLoss(loss_weights.cuda() if args.cuda else loss_weights)
            
        elif args.loss_fn == 'masked_mse':
            loss_function = MaskedMSELoss()
    
    
    else:
        if args.loss_fn == 'masked_nll':
            loss_function = MaskedNLLLoss()
        elif args.loss_fn == 'unmasked_weighted_nll':
            loss_function = UnMaskedWeightedNLLLoss()
        elif args.loss_fn == 'masked_mse':
            loss_function = MaskedMSELoss()
    
    if args.verbose:
        print(model)
        print(f'Loss function:\t{loss_function.__class__.__name__}')
        print(f'Optimizer:\t{optimizer.__class__.__name__} with initial lr = {args.lr}, l2 = {args.l2}')
    
    return model, optimizer, loss_function



def load_model_from_ckpt(model_ckpt="audio/MELD_features/models/EmoNet_31.pt"):

    global D_s
    D_m = 300
    D_g = D_q = D_r = D_e = 150
    D_h = 100
    n_classes = 7

    model = EmotionNet(D_m, D_q, D_g, D_r, D_e, D_h, n_classes=n_classes)
    model.load_state_dict(torch.load(model_ckpt))
    
    return model


def run_inference(model, f, qmask, umask):
    
    log_prob, _, _ = model(f, qmask,umask)

    lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) # batch*seq_len, n_classes
    pred_ = torch.argmax(lp_,1) # batch*seq_len
    return pred_.data.cpu().numpy()
    

def trainer(args, model, train_loader, valid_loader, test_loader, optimizer, loss_function):
    
    train_losses, train_fscores, train_accs = [], [], []
    test_losses, test_fscores, test_accs = [], [], []
    val_losses, val_fscores, val_accs = [], [], []
    best_fscore, best_loss, best_label, best_pred, best_mask, best_attn = None, None, None, None, None, None
    
    for e in range(1, args.epochs+1):
        start_time = time.time()
        train_loss, train_acc, _,_,_,train_fscore,_,_= train_or_eval_model(model=model, loss_function=loss_function,dataloader=train_loader, epoch=e, optimizer=optimizer, train=True, cuda=args.cuda, feature_type=args.feature_type)
        valid_loss, valid_acc, _,_,_,val_fscore,_, _= train_or_eval_model(model=model, loss_function=loss_function, dataloader=valid_loader, epoch=e,cuda=args.cuda, feature_type=args.feature_type)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions, test_class_report = train_or_eval_model(model=model, loss_function=loss_function, dataloader=test_loader, epoch=e, train=False, cuda=args.cuda, feature_type=args.feature_type)


        train_losses.append(train_loss)
        train_fscores.append(train_fscore)
        train_accs.append(train_acc)
        
        test_losses.append(test_loss)
        test_fscores.append(test_fscore)
        test_accs.append(test_acc)
        
        val_losses.append(valid_loss)
        val_fscores.append(val_fscore)
        val_accs.append(valid_acc)
        

        if best_fscore == None or best_fscore < test_fscore :
            best_fscore, best_loss, best_label, best_pred, best_mask, best_attn = test_fscore, test_loss, test_label, test_pred, test_mask, attentions


        if e % 10 == 0 or e == 1:
            print(f'Epoch [{e}]/[{args.epochs}]\t',
                
                f'Train Loss: {train_loss:.3f}\t',
                f'Train Acc: {train_acc:.3f}%\t',
                f'Train F1: {train_fscore:.3f}\t',
                
                f'Val Loss: {valid_loss:.3f}\t',
                f'Val Acc: {valid_acc:.3f}%\t',
                f'Val F1: {val_fscore:.3f}\t',
                
                f'Test Loss: {test_loss:.3f}\t',
                f'Test Acc: {test_acc:.3f}%\t',
                f'Test F1: {test_fscore:.3f}\t',
                
                f'Time: {time.time()-start_time:.2f} sec '
                )
        
        #### early stopping
        min_delta = 0.01
        patience = 5
        
        if e > 1:
            if test_losses[-1] - test_losses[-2] > min_delta:
                patience_cnt += 1
            else:
                patience_cnt = 0
                
            if patience_cnt > patience:
                print(f"[Early stopping at epoch: {e}]")
                break
        else:
            patience_cnt = 0


    train_metrics = {'train_losses': train_losses, 
                     'train_fscores': train_fscores, 
                     'train_accs': train_accs
                     }
    
    test_metrics = {'test_losses': test_losses, 
                    'test_fscores': test_fscores, 
                    'test_accs': test_accs
                    }
    
    val_metrics = {'val_losses': val_losses, 
                   'val_fscores': val_fscores, 
                   'val_accs': val_accs
                   }
    
    best_metrics = {'best_loss': best_loss, 
                    'best_fscore': best_fscore, 
                    'best_acc': accuracy_score(best_label,best_pred,sample_weight=best_mask)*100
                    }
    
    metrics = {'train': train_metrics, 
               'val': val_metrics, 
               'test': test_metrics, 
               'best': best_metrics
               }
    
    return model, metrics, best_label, best_pred, best_mask, best_attn, test_class_report