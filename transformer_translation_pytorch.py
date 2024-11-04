import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, blue, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

spacy_ger = spacy.load("de")
spact_eng = spacy.load("en")

# Tokenizers
def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

# what is Field???
german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, valid_data, test_data = Multi30k.splits(exts=(".de",".en"),fields=(german, english))

#
german.build.vocab(train_data, max_size=10000, min_freq=2)
english.build.vocab(train_data, max_size=10000, min_freq=2)

class Transformer(nn.Module):
    def __init__(
            self,
            embedding_size,
            src_vocab_size,#?
            trg_vocab_size,#?
            src_pad_idx,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,#?
            dropout,
            max_len,#? maybe the max length of a sentence?
            device,                 
                 ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(embedding_size,num_heads,num_encoder_layers, num_decoder_layers, forward_expansion, dropout)

        