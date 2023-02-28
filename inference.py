from trainer import *
from hmm_syllables import *
from utils import *
from scipy.stats import moyal
from torchtext.vocab import vocab as torch_vocab
import sys

def generate_babytalk_sentences(n_sent=10, output_file=None):
    sampled_lens = list(map(int, moyal.rvs(loc = 4.752048401023519, scale = 2.123603020197126, size=n_sent)))

    vocab = torch.load('vocab_obj.pth')

    model = HMM_syllable(M=len(vocab), N=6)
    model.load_state_dict(torch.load("model_tokens.pt"))
    model.eval()
    file = open(output_file, "w") if output_file else sys.stdout
    for i in range(n_sent):
        sampled_x, sampled_z = model.sample(T=sampled_lens[i])
        if output_file:
            file.write(str(sampled_lens[i])+":"+decode(sampled_x, vocab) + "\n")
        else:
            print(sampled_z, decode(sampled_x, vocab))
    if output_file:
        file.close()
