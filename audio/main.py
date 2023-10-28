import torch
import torch.optim as optim
import time

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 

from model import BiModel
from losses import MaskedNLLLoss
from dataloader import get_MELD_loaders
from train import train_or_eval_model

cuda = torch.cuda.is_available()
if cuda:
    print('Running on GPU')
else:
    print('Running on CPU')
    

# choose between 'sentiment' or 'emotion'
classification_type = 'emotion'
feature_type = 'multimodal'

data_path = 'DialogueRNN_features/MELD_features/'
batch_size = 30
n_classes = 3
n_epochs = 1
active_listener = True
attention = 'general'
class_weight = False
l2 = 0.00001
lr = 0.0005

if feature_type == 'text':
    print("Running on the text features........")
    D_u = 600
elif feature_type == 'audio':
    print("Running on the audio features........")
    D_u = 300
else:
    print("Running on the multimodal features........")
    D_u = 900
    
D_g = 150
D_s = 150
D_i = 150
D_e = 100
D_h = 100

D_a = 100 # concat attention

loss_weights = torch.FloatTensor([1.0,1.0,1.0])

if classification_type.strip().lower() == 'emotion':
    n_classes = 7
    loss_weights = torch.FloatTensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0])

model = BiModel(D_m, D_g, D_p, D_e, D_h,
                n_classes=n_classes,
                listener_state=True)

if cuda:
    model.cuda()
if class_weight:
    loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
else:
    loss_function = MaskedNLLLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=lr,
                       weight_decay=l2)

train_loader, valid_loader, test_loader =\
        get_MELD_loaders(data_path + 'MELD_features_raw.pkl', n_classes,
                            valid=0.0,
                            batch_size=batch_size,
                            num_workers=0)

best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None


for e in range(n_epochs):
    start_time = time.time()
    train_loss, train_acc, _,_,_,train_fscore,_,_= train_or_eval_model(model, loss_function,
                                           train_loader, e, optimizer, True)
    valid_loss, valid_acc, _,_,_,val_fscore,_= train_or_eval_model(model, loss_function, valid_loader, e)
    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions, test_class_report = train_or_eval_model(model, loss_function, test_loader, e)

    if best_fscore == None or best_fscore < test_fscore:
        best_fscore, best_loss, best_label, best_pred, best_mask, best_attn =\
                test_fscore, test_loss, test_label, test_pred, test_mask, attentions

    print('epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
            format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,\
                    test_loss, test_acc, test_fscore, round(time.time()-start_time,2)))
    print (test_class_report)


print('Test performance..')
print('Fscore {} accuracy {}'.format(best_fscore,
                                 round(accuracy_score(best_label,best_pred,sample_weight=best_mask)*100,2)))
print(classification_report(best_label,best_pred,sample_weight=best_mask,digits=4))
print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))