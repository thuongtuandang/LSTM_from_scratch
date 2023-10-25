import numpy as np


class Dataset:
    def __init__(self, input_length=2, output_length=1) -> None:
        self.input_length = input_length
        self.output_length = output_length
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
    
    def load(self, data_path="data/robert_frost.txt", train_test_ratio=0.8):
        # TODO: load data to self.x_train ....
        data = self._processing()

    def _processing(self, inputs):
        vocab = []
        for k, v in input.items():
            v = k.split()
            for word in v:
                if word not in vocab:
                    vocab.append(word)
        self.vocab_size = len(vocab)
        v = []
        for word in inputs.split():
            v.append(self._oneHotEncode(word))
        return v

    def _oneHotEncode(self, word):
        v = np.zeros(self.vocab_size)
        v[self._word_to_idx(word)] = 1
        return v

    def _word_to_idx(self, word):
        return self.vocab_size.index(word)