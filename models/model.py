class BaseModel:

    def _forward(self):
        pass

    def _backward(self):
        pass

    def fit(self, data):
        pass

    def predict(self, data):
        pass

    def score(self, x, y):
        pass