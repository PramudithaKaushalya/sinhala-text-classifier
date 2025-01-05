import torch
from torch.utils.data import Dataset
from collections import Counter
from nltk.tokenize import word_tokenize

class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        # Convert text to indices
        text_indices = [self.vocab.get(token, self.vocab['<PAD>']) for token in text]
        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def tokenize_text(text):
    # Tokenize the text using NLTK or other tokenizer
    return word_tokenize(text.lower())

def build_vocab(data, min_freq=1):
    # Build vocabulary from the dataset
    counter = Counter()
    for text, _ in data:
        counter.update(text)

    vocab = {word: idx for idx, (word, count) in enumerate(counter.items()) if count >= min_freq}
    vocab['<PAD>'] = len(vocab)  # Add padding token

    return vocab

def collate_fn(batch):
    texts, labels = zip(*batch)
    # Pad sequences to same length
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return padded_texts, torch.stack(labels)
