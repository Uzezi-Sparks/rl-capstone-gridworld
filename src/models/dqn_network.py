import numpy as np

class DQNNetwork:
    """
    Lightweight MLP for DQN on discrete grid environments.
    No PyTorch/TensorFlow needed — pure numpy for compatibility.
    
    Architecture: Input(state) -> Hidden(64) -> Hidden(64) -> Output(Q-values)
    """

    def __init__(self, input_size, output_size, hidden_size=64, lr=0.001):
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size

        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        self.x = np.array(x, dtype=float)
        self.h1 = self.relu(self.x @ self.W1 + self.b1)
        self.h2 = self.relu(self.h1 @ self.W2 + self.b2)
        self.out = self.h2 @ self.W3 + self.b3
        return self.out

    def backward(self, loss_grad):
        """Backprop through network, update weights with SGD"""
        dout = loss_grad
        dW3 = self.h2.reshape(-1, 1) * dout
        db3 = dout
        dh2 = dout @ self.W3.T * (self.h2 > 0)
        dW2 = self.h1.reshape(-1, 1) * dh2
        db2 = dh2
        dh1 = dh2 @ self.W2.T * (self.h1 > 0)
        dW1 = self.x.reshape(-1, 1) * dh1
        db1 = dh1

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def get_weights(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = weights

    def save(self, path):
        np.save(path, np.array(self.get_weights(), dtype=object))
        print(f"Network saved: {path}")

    def load(self, path):
        weights = np.load(path, allow_pickle=True)
        self.set_weights(list(weights))
        print(f"Network loaded: {path}")