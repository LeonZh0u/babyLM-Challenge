import sys
# sys.path.insert(0,'/Users/leon/Downloads/babylm/AOCHILDES')
sys.path.insert(0,'/nlp/data/zhliyang/babyLM-Challenge/AOCHILDES')
from AOCHILDES.aochildes.dataset import AOChildesDataSet
import torch
import numpy as np
from tqdm import tqdm # for displaying progress bar

import pickle
import stanza
import torch.utils.data
from collections import Counter
from sklearn.model_selection import train_test_split
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from trainer import *
from hmm_syllables import *
from nltk.tokenize import SyllableTokenizer
import nltk
from nltk import word_tokenize
nltk.download('punkt')


SSP = SyllableTokenizer()
v2 = torch.load('vocab_obj.pth')
v2.insert_token(" ", len(v2))
v2.set_default_index(v2['<unk>'])

def encode(s):
    """
    Convert a string into a list of integers
    """
    x = sum([SSP.tokenize(word)+[" "] for word in word_tokenize(s)],[]) + ["\n"]
    return [v2[xx] for xx in x]

def decode(x):
    """
    Convert list of ints to string
    """
    s = " ".join([v2.lookup_token(xx) for xx in x])
    return s



# transcripts = AOChildesDataSet().load_transcripts()


class TextDataset(torch.utils.data.Dataset):
  def __init__(self, lines):
    self.lines = lines # list of strings
    collate = Collate() # function for generating a minibatch from strings
    self.loader = torch.utils.data.DataLoader(self, batch_size=32, shuffle=True, collate_fn=collate)

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    return line

class Collate:
  def __init__(self):
    pass

  def __call__(self, batch):
    """
    Returns a minibatch of strings, padded to have the same length.
    """
    x = []
    batch_size = len(batch)
    for index in range(batch_size):
      x_ = batch[index]
      # convert letters to integers
      x.append(encode(x_))
    # pad all sequences with 0 to have same length
    x_lengths = [len(x_) for x_ in x]
    T = max(x_lengths)
    for index in range(batch_size):
      x[index] += [0] * (T - len(x[index]))
      x[index] = torch.tensor(x[index])

    # stack into single tensor
    x = torch.stack(x)
    x_lengths = torch.tensor(x_lengths)
    return (x,x_lengths)
  
""" for i, text in enumerate(tqdm(transcripts)):
  for sent in nlp(text).sentences:
    for token in sent.tokens:
      tokens.append(token.text) """

"""
with open("tokens", "wb") as fp:   #Pickling
   pickle.dump(tokens, fp) """


def main():
    transcripts = AOChildesDataSet().load_transcripts()
    unk_token = '<unk>'
    nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)

    sentences = []
    """
    for i, text in enumerate(tqdm(transcripts)):
      sentences += split_into_sentences(text)
    
    with open("sentences", "wb") as fp:   #Pickling
      pickle.dump(sentences, fp)
    print("**********************")
    all_tokens = []
    for transcript in tqdm(transcripts):
      all_tokens += word_tokenize(transcript)

    v2 = vocab(OrderedDict(Counter(all_tokens)), specials=[unk_token])
    
    v2.insert_token("\n", len(v2))
    v2.set_default_index(v2['<unk>'])
    torch.save(v2, 'vocab_obj_tokens.pth')
    """
    with open("sentences", "rb") as fp:   # Unpickling
      sentences = pickle.load(fp)
    print(sentences[:10])
    train_lines, valid_lines = train_test_split(sentences, test_size=0.1, random_state=42)
    train_dataset = TextDataset(train_lines)

    model = HMM_syllable(M=len(v2), N=6) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)
    train_loss = 0
    num_samples = 0
    model.train()
    print_interval = 50

    for idx, batch in enumerate(tqdm(train_dataset.loader)):
        x,T = batch
        batch_size = len(x)
        num_samples += batch_size
        log_probs = model(x,T)
        loss = -log_probs.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy().item() * batch_size
        if idx % print_interval == 0:
            print("loss:", loss.item())
            for _ in range(5):
                sampled_x, sampled_z = model.sample(stop_token_index = v2["\n"])
                print(sampled_z)
                print(decode(sampled_x))


    torch.save(model.state_dict(), "model_tokens.pt")
    # model = HMM_syllable(M=len(v2), N=6) 
    # model.load_state_dict(torch.load("model_tokens.pt"))
    print("saved")

if __name__ == "__main__":
  main()