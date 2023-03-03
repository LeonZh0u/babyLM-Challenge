import re
import torch
from nltk.tokenize import SyllableTokenizer
from nltk import word_tokenize
import os
import pickle
import multiprocessing
from torchtext.vocab import vocab as torch_vocab
from collections import OrderedDict, Counter
from tqdm import tqdm
from string import punctuation
import re

SSP = SyllableTokenizer()

def read_text_file(file_path):
        file_content = []
        with open(file_path, 'r') as f:
            for line in f:
                line = re.sub(r'[^A-Za-z0-9 ]+', '', line)
                file_content.append(line)
        return file_content

def load_transcripts(path):
    file_content = []
    os.chdir(path)
    print(os.listdir())
    for file in os.listdir():
        if file.endswith(".train"):
            file_path = f"{path}/{file}"
            print(file_path)
            file_content+=read_text_file(file_path)
    return file_content

def build_or_load_vocab(transcript, unk_token="<unk>"):
    if os.path.exists('vocab_obj.pth'):
        return torch.load('vocab_obj.pth')
    all_tokens = []
    pool = multiprocessing.Pool(None)

    for out in tqdm(pool.map((word_tokenize), transcript), total=len(transcript)):
        all_tokens += out

    all_syllables = []
    for out in tqdm(pool.map(SSP.tokenize, all_tokens), total=len(all_tokens)):
        all_syllables += out
    
    vocab = torch_vocab(OrderedDict(Counter(all_syllables)))

    vocab.insert_token(" ", len(vocab))
    torch.save(vocab, 'vocab_obj.pth')
    return vocab

def build_or_load_train_lines(transcripts):
    if os.path.exists('sentences'):
        with open("sentences", "rb") as fp:   # Unpickling
            sentences = pickle.load(fp)
        return sentences
    
    sentences = []
    for text in tqdm(transcripts):
        sentences += split_into_sentences(text)

    with open("sentences", "wb") as fp:  # Pickling
        pickle.dump(sentences, fp)
    return sentences

def encode(s, vocab):
    """
    Convert a string into a list of integers
    """
    s = s.strip(punctuation)
    s = re.sub(r'[^A-Za-z0-9 ]+', '', s)
    x = sum([SSP.tokenize(word)+[" "]
            for word in word_tokenize(s)], []) +[" "]
    return [vocab[xx] for xx in x]


def decode(x, vocab):
    """
    Convert list of ints to string
    """
    s = "".join([vocab.lookup_token(xx) for xx in x])
    return s


def log_domain_matmul(log_A, log_B):
    """
    log_A : m x n
    log_B : n x p
    output : m x p matrix

    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
    """
    m = log_A.shape[0]
    n = log_A.shape[1]
    p = log_B.shape[1]

    # log_A_expanded = torch.stack([log_A] * p, dim=2)
    # log_B_expanded = torch.stack([log_B] * m, dim=0)
# fix for PyTorch > 1.5 by egaznep on Github:
    log_A_expanded = torch.reshape(log_A, (m, n, 1))
    log_B_expanded = torch.reshape(log_B, (1, n, p))

    elementwise_sum = log_A_expanded + log_B_expanded
    out = torch.logsumexp(elementwise_sum, dim=1)

    return out


def maxmul(log_A, log_B):
    """
    log_A : m x n
    log_B : n x p
    output : m x p matrix

    Similar to the log domain matrix multiplication,
    this computes out_{i,j} = max_k log_A_{i,k} + log_B_{k,j}
    """
    m = log_A.shape[0]
    n = log_A.shape[1]
    p = log_B.shape[1]

    log_A_expanded = torch.stack([log_A] * p, dim=2)
    log_B_expanded = torch.stack([log_B] * m, dim=0)

    elementwise_sum = log_A_expanded + log_B_expanded
    out1, out2 = torch.max(elementwise_sum, dim=1)

    return out1, out2


alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    if "..." in text:
        text = text.replace("...", "<prd><prd><prd>")
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms+" "+starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" +
                  alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets +
                  "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" "+suffixes+"[.] "+starters, " \\1<stop> \\2", text)
    text = re.sub(" "+suffixes+"[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    return sentences
