class BaseModel:

    def _forward(self, inputs):
        pass

    def _backward(self, inputs, dy, learning_rate):
        pass

    def fit(self, X, y, max_iter = 1001, learning_rate = 0.03):
        pass

    def predict(self, X, y):
        pass

    def score(self, x, y):
        pass