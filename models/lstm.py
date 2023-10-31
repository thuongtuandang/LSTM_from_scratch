import numpy as np
from dataset import Dataset
from models.model import BaseModel
from utils import hadamard, sigmoid, softmax

class LSTM(BaseModel):
    def __init__(self, input_size, output_size, hidden_size = 64):
        # Initialize W
        self.Wf = np.random.rand(hidden_size, input_size)/1000
        self.Wi = np.random.rand(hidden_size, input_size)/1000
        self.Wc = np.random.rand(hidden_size, input_size)/1000
        self.Wo = np.random.rand(hidden_size, input_size)/1000
        self.Wy = np.random.rand(output_size, hidden_size)/1000
        # Initialize U
        self.Uf = np.random.rand(hidden_size, hidden_size)/1000
        self.Ui = np.random.rand(hidden_size, hidden_size)/1000
        self.Uc = np.random.rand(hidden_size, hidden_size)/1000
        self.Uo = np.random.rand(hidden_size, hidden_size)/1000
        # Initialize b
        self.by = np.zeros((output_size,1))
        self.bf = np.zeros((hidden_size,1))
        self.bi = np.zeros((hidden_size,1))
        self.bc = np.zeros((hidden_size,1))
        self.bo = np.zeros((hidden_size,1))
    
    def _forward(self, inputs):
        self.inputs = []
        # Initialize arrays h and c
        self.hs = []
        self.cs = []
        # Initialize for gates
        self.fs = []
        self.iss = []
        self.os = []
        # css is the array for the gate c tilde
        self.css = []

        h = np.zeros((self.Wf.shape[0], 1))
        self.hs.append(h)
        c = np.zeros((self.Wf.shape[0], 1))
        self.cs.append(c)

        # Now compute
        for t, x in enumerate(inputs):
            x = np.array(x)
            x = x.reshape(-1,1)
            self.inputs.append(x)
            f = sigmoid(self.Uf @ self.hs[t] + self.Wf @ x + self.bf)
            c_tilde = np.tanh(self.Uc @ self.hs[t] + self.Wc @ x + self.bc)
            i = sigmoid(self.Ui @ self.hs[t] + self.Wi @ x + self.bi)
            o = sigmoid(self.Uo @ self.hs[t] + self.Wo @ x + self.bo)
            # Compute the memory cell
            c = hadamard(c_tilde, i) + hadamard(self.cs[t], f)
            # Compute the hidden state
            h = hadamard(o, np.tanh(c))
            # Append values for memory cell and hidden state
            self.hs.append(h)
            self.cs.append(c)
            # Append values for gates
            self.css.append(c_tilde)
            self.fs.append(f)
            self.os.append(o)
            self.iss.append(i)
        
        n = len(inputs)
        y_pred = self.Wy @ self.hs[n] + self.by
        self.outputs = softmax(y_pred)
        return self.outputs
    
    def _backward(self, dy, learning_rate = 0.03):
        n = len(self.inputs)
        dby = dy
        dWy = dy @ self.hs[n].T
        # That is dL/dh[n]
        dh = self.Wy.T @ dy

        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWo = np.zeros_like(self.Wo)
        dWc = np.zeros_like(self.Wc)

        dUf = np.zeros_like(self.Uf)
        dUi = np.zeros_like(self.Ui)
        dUo = np.zeros_like(self.Uo)
        dUc = np.zeros_like(self.Uc)

        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo)
        dbc = np.zeros_like(self.bc)

        for t in reversed(range(n-1)):
            tmpf = self.fs[t+1] * (1 - self.fs[t+1])
            tmpi = self.iss[t+1] * (1 - self.iss[t+1])
            tmpo = self.os[t+1] * (1 - self.os[t+1])

            tmpc = 1 - np.tanh(self.cs[t])**2
            tmpc_tilde = 1 - np.tanh(self.css[t+1])**2

            dhtdct = self.os[t+1] @ tmpc.T
            dhtdot = np.tanh(self.cs[t+1])

            dotdht_1 = self.Uo @ tmpo
            dotdUo = tmpo @ self.hs[t].T
            dotdWo = tmpo @ self.inputs[t+1].T
            dotdbo = np.sum(tmpo, axis=1, keepdims=True)

            dctdct_1 = self.fs[t+1]
            dctdft = self.cs[t]
            dctdit = self.css[t+1]
            dctdct_tilde = self.iss[t+1]

            dftdUf = tmpf @ self.hs[t].T
            dftdht_1 = self.Uf @ tmpf
            dftdWf = tmpf @ self.inputs[t+1].T
            dftdbf = np.sum(tmpf, axis = 1, keepdims=True)

            ditdUi = tmpi @ self.hs[t].T
            ditdht_1 = self.Ui @ tmpi
            ditdWi = tmpi @ self.inputs[t+1].T
            ditdbi = np.sum(tmpi, axis = 1, keepdims=True)

            dc_tildedUc = tmpc_tilde @ self.hs[t].T
            dc_tildedht_1 = self.Uc @ tmpc_tilde
            dc_tildedWc = tmpc_tilde @ self.inputs[t+1].T
            dc_tildedbc = np.sum(tmpc_tilde, axis = 1, keepdims = True)
            

            dhtdht_1 = dhtdct @ dctdct_tilde @ dc_tildedht_1.T + dhtdct @ dctdft @ dftdht_1.T + dhtdct @ dctdit @ ditdht_1.T + dhtdot @ dotdht_1.T
            # temp is dL/dht
            temp = dhtdht_1 @ dh

            dUo += temp @ (dhtdot.T @ dotdUo)
            dWo += temp @ (dhtdot.T @ dotdWo)
            dbo += temp @ (dhtdot.T @ dotdbo)

            dUf += temp @ (dhtdct @ dctdft).T @ dftdUf
            dWf += temp @ (dhtdct @ dctdft).T @ dftdWf
            dbf += temp @ (dhtdct @ dctdft).T @ dftdbf

            dUi += temp @ (dhtdct @ dctdit).T @ ditdUi
            dWi += temp @ (dhtdct @ dctdit).T @ ditdWi
            dbi += temp @ (dhtdct @ dctdit).T @ ditdbi

            dUc += temp @ (dhtdct @ dctdct_tilde).T @ dc_tildedUc
            dWc += temp @ (dhtdct @ dctdct_tilde).T @ dc_tildedWc
            dbc += temp @ (dhtdct @ dctdct_tilde).T @ dc_tildedbc

            # Update dh and clip its values
            dh = temp
            for d in [dh]:
                np.clip(d, 1e-15, 1 - 1e-15, out = d)
        

        # Clip to prevent gradient vanishing/exploding
        for d in [dUo, dWo, dbo, dUf, dWf, dbf, dUi, dWi, dbi, dUc, dWc, dbc, dby, dWy]:
            np.clip(d, -1, 1, out = d)

        # Update parameters
        self.Uf -= learning_rate * dUf
        self.Wf -= learning_rate * dWf
        self.bf -= learning_rate * dbf

        self.Wi -= learning_rate * dWi
        self.Ui -= learning_rate * dUi
        self.bi -= learning_rate * dbi

        self.Wo -= learning_rate * dWo
        self.Uo -= learning_rate * dUo
        self.bo -= learning_rate * dbo

        self.Wc -= learning_rate * dWc
        self.Uc -= learning_rate * dUc
        self.bc -= learning_rate * dbc

        self.by -= learning_rate * dby
        self.Wy -= learning_rate * dWy
    
    def _process(self, X, y, learning_rate=0.03, run_backward=False) -> float:
        accuracy = 0
        results = []
        for i, x in enumerate(X):
            probs = self._forward(x)
            results.append(np.argmax(probs))
            # Accuracy
            accuracy += (np.argmax(probs) == np.argmax(y[i]))

            # Calculate backward
            if run_backward:
                dy = probs.copy()
                dy[np.argmax(y[i])] -= 1
                self._backward(dy, learning_rate)

        return results, float(accuracy/len(X))
    
    def fit(self, X, y, max_iter = 101, learning_rate = 0.03):
        for i in range(max_iter):
            _, accuracy = self._process(X, y, learning_rate, run_backward=True)
            if(i % 10 == 0):
                print(f"Step: {i}")
                print(f"accuracy for training data: {accuracy}")

    def predict(self, X, y):
        return self._process(X, y, run_backward=False)

    def score(self, x, y):
        pass