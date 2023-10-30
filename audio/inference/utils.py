from audio.inference.encoder import *
from audio.model import *
import torch

def load_conversation(idx, pred_emotions=None):
    num_to_label_map = {0: 'neutral', 1: 'surprise', 2: 'fear', 3: 'sadness', 4: 'joy', 5: 'disgust', 6: 'anger'}

    list_of_speakers = set(''.join(str(e) for e in speaker) for speaker in videoSpeakers[idx])
    num_of_speakers = len(list_of_speakers)
    nums = [i+1 for i in range(num_of_speakers)]
    speaker_dict = dict(zip(list_of_speakers, nums))

    print(f"Loading dialogue #{idx} with {num_of_speakers} speakers, {len(videoSentence[idx])} utterances.\n")

    i=1
    sentence_count=0
    for sentence, speaker, label in zip(videoSentence[idx], videoSpeakers[idx], videoLabels[idx]):
        speaker = ''.join(str(e) for e in speaker)
        
        if pred_emotions is not None:
            print(f'[#{i}] Person {speaker_dict[speaker]} ({num_to_label_map[label]} - {num_to_label_map[pred_emotions[sentence_count]]})\t:\t{sentence}')
        else:
            print(f'[#{i}] Person {speaker_dict[speaker]} ({num_to_label_map[label]}) :\t{sentence}')

        i+=1
        sentence_count+=1


def create_qmask(N, X, U):
    qmask = []
    for i in range(U):
        utterance_mask = [0] * N + [0] * (9 - N)  # initialize the mask with zeros
        speaker = X[i] - 1  # Convert to 0-based index
        utterance_mask[speaker] = 1
        qmask.append(utterance_mask)
    return qmask

def create_umask(U):
    return [1] * U

def create_masks(X, U):
    N = len(set(X))
    qmask = create_qmask(N, X, U)
    umask = create_umask(U)
    return qmask, umask


# end to end function to get features for a dialogue
def get_features_for_dialogue(dialogue_dir, X):
    acouf = make_embs_for_dialogue(dialogue_dir)
    qmask, umask = create_masks(X, len(acouf))
    return torch.FloatTensor(acouf), \
        torch.FloatTensor(qmask), \
            torch.FloatTensor(umask)

def load_model_from_ckpt(model_ckpt="audio/MELD_features/models/EmoNet_31.pt"):
    # load model
    global D_s
    D_m = 300
    D_g = D_q = D_r = D_e = 150
    D_h = 100
    n_classes = 7

    model = EmotionNet(D_m, D_q, D_g, D_r, D_e, D_h, n_classes=n_classes)
    model.load_state_dict(torch.load(model_ckpt))
    
    return model


def run_inference(model, acouf, qmask, umask):
    
    log_prob, _, _ = model(acouf, qmask,umask)
    
    lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) # batch*seq_len, n_classes

    pred_ = torch.argmax(lp_,1) # batch*seq_len
    
    return pred_.data.cpu().numpy()
    
    