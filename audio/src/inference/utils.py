from audio.src.inference.encoder import *
from audio.src.model import *
import torch
torch.cuda.empty_cache()

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

    