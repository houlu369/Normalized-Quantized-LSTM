import os
import torch

from collections import Counter

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        if path.split('/')[1] in [ 'warpeace', 'text8']:
            self.train = self.tokenizeChar(os.path.join(path, 'train.txt'))
            self.valid = self.tokenizeChar(os.path.join(path, 'valid.txt'))
            self.test = self.tokenizeChar(os.path.join(path, 'test.txt'))
        else:
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file for a word level language model."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def tokenizeChar(self, path):
        """Tokenizes a text file for a character language model."""
        assert os.path.exists(path)
        data = open(path, 'rb').read()
        chars =  list(set(data))
        data_size, vocab_size = len(data), len(chars)
        # Add chars to the dictionary
        for char in chars:
            self.dictionary.add_word(char)
        # Process your text file
        ids = torch.LongTensor(data_size)
        token = 0
        for char in data:
            ids[token] = self.dictionary.word2idx[char]
            token += 1

        return ids
