import numpy as np
from models.model import BaseModel
from utils import softmax


class RNN(BaseModel):

    def __init__(self, input_size, output_size, hidden_size = 64) -> None:
        super().__init__()
        self.Whh = np.random.rand(hidden_size, hidden_size)/1000
        self.Wxh = np.random.rand(hidden_size, input_size)/1000
        self.Why = np.random.rand(output_size, hidden_size)/1000

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size,1))
    
    def _forward(self, inputs):
        self.inputs = inputs
        self.hs = []
        h = np.zeros((self.Whh.shape[0],1))
        self.hs.append(h)

        for x in inputs:
            x = x.reshape(-1,1)
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.hs.append(h)
        n = len(inputs)
        y_pred = self.Why @ self.hs[n] + self.by
        self.outputs = softmax(y_pred)

    def _backward(self, dy, learning_rate = 0.03):
        
        dWhh = np.zeros_like(self.Whh)
        dWxh = np.zeros_like(self.Wxh)
        dbh = np.zeros_like(self.bh)
        
        n = len(self.inputs)
        # These derivatives can be easily computed
        dby = dy
        dWhy = dy @ self.hs[n].T
        dh = self.Why.T @ dy

        for t in reversed(range(n)):
            temp = (1 - self.hs[t+1]**2) * dh
            dbh += temp
            dWxh += temp * self.inputs[t].T
            dWhh += temp * self.hs[t].T
        
        for d in [dWhh, dWxh, dWhy, dbh, dby]:
            np.clip(d, -1, 1, out = d)
        
        # Update
        self.Whh -= learning_rate*dWhh
        self.Wxh -= learning_rate*dWxh
        self.Why -= learning_rate*dWhy
        self.bh -= learning_rate*dbh
        self.by -= learning_rate*dby

    def _process(self, data, learning_rate = 0.03):
        items = list(data.items())
        accuracy = 0
        for x, y_true in items:
            # inputs = createInputs(x) # Transform a sentence to a matrix
            true_index = int(y_true) # True label
            # Compute h and probability for each sentence
            self.forward(x) 
            probs = self.outputs
            accuracy += int(np.argmax(probs) == true_index) # Accuracy
            dy = probs
            dy[true_index] -= 1 # The formula for dy
            # Update parameters for each sentence
            self.backward(dy, learning_rate) 
        self.accuracy = float(accuracy/len(data)) #Accuracy
    
    def fit(self, data, max_iter = 1001, learning_rate = 0.03):
        for i in range(max_iter):
            self.process(data)
            if(i % 100 == 0):
                print(f"Step: {i}")
                print(f"accuracy for training data: {self.accuracy}")

    def predict(self, data):
        items = list(data.items())
        accuracy = 0
        for x, y_true in items:
            # inputs = createInputs(x)
            true_index = int(y_true)
            self.forward(x)
            probs = self.outputs
            accuracy += int(np.argmax(probs) == true_index)
        print(f"Accuracy for test data: {float(accuracy/len(data))}")

    def score(self, x, y):
        pass