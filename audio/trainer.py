import numpy as np,random
import torch
from sklearn.metrics import f1_score, accuracy_score

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False, seed=42, cuda=False):
    losses, preds, labels, masks  = [], [], [], []
    alphas, alphas_f, alphas_b = [], [], []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed)
    
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        # print(len(data))
        # r1, r2, r3, r4, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        
        _, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
                
        log_prob, alpha_f, alpha_b = model(acouf, qmask, umask)

        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        if train:
            total_loss = loss
            total_loss.backward()
            optimizer.step()
        else:
            alphas_f += alpha_f
            alphas_b += alpha_b

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
        
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, [avg_fscore], [alphas, alphas_f, alphas_b]


