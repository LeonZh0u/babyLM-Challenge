from trainer import *
from hmm_syllables import *
from utils import *
import numpy as np
import pandas as pd
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
import pickle
from scipy.stats import lognorm
import sys
from torchtext.vocab import vocab as torch_vocab
from collections import Counter


def generate_babytalk_sentences(n_sent=10, output_file=None):
    sampled_lens = list(map(int, lognorm.rvs(0.5491595459713867, loc=-
                                             0.4303603242253272, scale=4.628530660114688, size=n_sent)))

    vocab = torch.load('vocab_obj.pth')
    vocab.insert_token(" ", len(vocab))
    vocab.set_default_index(vocab['<unk>'])

    model = HMM_syllable(M=len(vocab), N=6)
    model.load_state_dict(torch.load("model_tokens.pt"))
    model.eval()
    file = open(output_file, "w") if output_file else sys.stdout
    for i in range(n_sent):
        sampled_x, sampled_z = model.sample(T=sampled_lens[i])
        if output_file:
            file.write(decode(sampled_x, vocab) + "\n")
        else:
            print(sampled_z, decode(sampled_x, vocab))
    if output_file:
        file.close()
