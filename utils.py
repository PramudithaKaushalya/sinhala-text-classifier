import torch
from nltk.tokenize import word_tokenize

def tokenize_text(text):
    """
    Tokenize the input text using NLTK.
    """
    return word_tokenize(text.lower())


def build_vocab(data, min_freq=1):
    """
    Build a vocabulary from the dataset.
    """
    from collections import Counter
    counter = Counter()
    for text, _ in data:
        counter.update(text)

    vocab = {word: idx for idx, (word, count) in enumerate(counter.items()) if count >= min_freq}
    vocab['<PAD>'] = len(vocab)  # Add padding token

    return vocab


def collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length.
    """
    texts, labels = zip(*batch)
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return padded_texts, torch.stack(labels)
