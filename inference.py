from trainer import *
from hmm_syllables import *
from utils import *
from scipy.stats import gumbel_r
from torchtext.vocab import vocab as torch_vocab
import sys

def generate_babytalk_sentences(n_sent=10, output_file=None):
    sampled_lens = list(map(int, gumbel_r.rvs(loc = 7.33330589149, scale = 9.011387779280227, size=n_sent)))

    vocab = torch.load('data/babylm_data/babylm_10M/vocab_obj.pth')
    vocab.insert_token("<unk>", len(vocab))
    vocab.set_default_index(vocab["<unk>"])
    model = HMM_syllable(M=len(vocab), N=6)
    model.load_state_dict(torch.load("checkpoints/model_tokens{}.pt".format(33085)))
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
