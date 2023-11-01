import argparse
import string
import numpy as np

from utils import oneHotEncode


class Dataset:
    def __init__(self, input_length=2, output_length=1) -> None:
        self.input_length = input_length
        self.output_length = output_length
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.word_dict = []
    
    def __repr__(self) -> str:
        info =   "-----DATASET INFO-----\n"
        info += f"- Input/Output length: {self.input_length} / {self.output_length}\n"
        info += f"- Vocabulary size: {self.vocab_size}\n"
        info += "----------------------\n"
        info += f"- x_train / y_train shape: {self.x_train.shape} / {self.y_train.shape}\n"
        info += f"- x_test / y_test shape: {self.x_test.shape} / {self.y_test.shape}\n"
        info += "----------------------\n"
        info += f"- Sample x:\n{self.x_train[0]}\n"
        info += f"- Sample y:\n{self.y_train[0]}\n"
        return info

    def load(self, data_path="data/robert_frost.txt", train_test_ratio=0.8):
        word_lines = []
        for line in open(data_path):
            line = line.translate(str.maketrans('','', string.punctuation))
            tokens = line.split()
            line_arr = []
            for word in tokens:
                word = word.lower()
                if word not in self.word_dict:
                    self.word_dict.append(word)
                line_arr.append(word)
            if len(line_arr) > 0: word_lines.append(line_arr)
        self.word_dict.sort()
        inputs = [[self.word_dict.index(word) for word in line] for line in word_lines]

        train_size = int(len(inputs) * train_test_ratio)
        train_data = inputs[:train_size]
        test_data = inputs[train_size:]
        self.x_train, self.y_train = self._processing(train_data)
        self.x_test, self.y_test = self._processing(test_data)
    
    def word_embedding(self, inputs):
        v = []
        for x in inputs.split():
            index = self.word_dict.index(x)
            v.append(oneHotEncode(index, self.vocab_size))
        return v

    @property
    def vocab_size(self):
        return len(self.word_dict)
    
    def _processing(self, inputs):
        X = []
        y_true = []
        for line in inputs:
            for i in range(len(line) - self.input_length - self.output_length):
                chunk = line[i : i + self.input_length + self.output_length]
                x = chunk[:self.input_length]
                y =chunk[-self.output_length:]
                X.append(list(map(lambda i: oneHotEncode(i, self.vocab_size), x)))
                y_true.append(list(map(lambda i: oneHotEncode(i, self.vocab_size), y)))
        
        return np.array(X), np.array(y_true)

if __name__ == "__main__":
    dataset = Dataset()
    dataset.load(data_path="notebooks/test.txt")
    print(dataset)