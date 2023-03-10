from nltk import word_tokenize
import nltk
from nltk.tokenize import SyllableTokenizer
from hmm_syllables import *
from inference import generate_babytalk_sentences
from trainer import *
import torch.utils.data
from utils import *
from tqdm import tqdm
import torch
from aochildes.dataset import AOChildesDataSet
from pathlib import Path
import aochildes
from string import punctuation
nltk.download('punkt')

# transcripts = AOChildesDataSet().load_transcripts()


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, lines, vocab):
        self.lines = lines  # list of strings
        # function for generating a minibatch from strings
        collate = Collate(vocab=vocab)
        self.loader = torch.utils.data.DataLoader(
            self, batch_size=32, shuffle=True, collate_fn=collate)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        return line


class Collate:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        """
        Returns a minibatch of strings, padded to have the same length.
        """
        x = []
        text = []
        batch_size = len(batch)
        for index in range(batch_size):
            x_ = batch[index]
            # convert letters to integers
            x.append(encode(x_, self.vocab))
            text.append(x_)
        # pad all sequences with 0 to have same length
        x_lengths = [len(x_) for x_ in x]
        T = max(x_lengths)
        for index in range(batch_size):
            x[index] += [0] * (T - len(x[index]))
            x[index] = torch.tensor(x[index])

        # stack into single tensor
        x = torch.stack(x)
        x_lengths = torch.tensor(x_lengths)
        return ((x, x_), x_lengths)

def train_hmm_syllables(model_dir, dataset_dir, pretrain_dir):
    
    transcripts = load_transcripts(dataset_dir)
    print("**********************BUILD VOCAB**********************")
    vocab = build_or_load_vocab(transcripts)
    vocab.insert_token("<unk>", len(vocab))
    vocab.set_default_index(vocab["<unk>"])
    print("**********************Split Sentences**********************")
    train_lines = build_or_load_train_lines(transcripts)
    print(train_lines[:10])
    train_dataset = TextDataset(train_lines, vocab)

    print("**********************START TRAINING**********************")
    model = HMM_syllable(M=len(vocab), N=6)
    if os.path.exists(pretrain_dir)::
        model.load_state_dict(torch.load(pretrain_dir))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.00001)
    train_loss = 0
    num_samples = 0
    model.train()
    print_interval = 500
    save_interval = 5000
    for idx, batch in enumerate(tqdm(train_dataset.loader)):
        (x, text), T = batch
        batch_size = len(x)
        num_samples += batch_size
        log_probs = model(x, T)
        loss = -log_probs.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy().item() * batch_size
        if idx % print_interval == 0:
            print("loss:", loss.item())
            for _ in range(5):
                sampled_x, sampled_z = model.sample()
                print(sampled_z)
                print(decode(sampled_x, vocab))
        if idx % save_interval == 0:
            torch.save(model.state_dict(), model_dir+"model_tokens{}.pt".format(idx))
    torch.save(model.state_dict(), model_dir+"model_tokens{}.pt".format(idx))
    # model = HMM_syllable(M=len(vocab), N=6)
    # model.load_state_dict(torch.load("model_tokens.pt"))
    print("saved")


if __name__ == "__main__":
    #aochildes.configs.Dirs.transcripts = Path("data/aochildes/")
    
    model_dir = os.getcwd() + '/checkpoints/'
    dataset_dir = os.getcwd() + "/data/babylm_data/babylm_10M"
    pretrain_dir = os.getcwd() + '/checkpoints/model_tokens30000'
    train_hmm_syllables(model_dir, dataset_dir, pretrain_dir)
    generate_babytalk_sentences(10, "babytalk_corpus.txt")
