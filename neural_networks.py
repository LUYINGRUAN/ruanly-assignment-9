import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # Learning rate
        self.activation_fn = activation  # Activation function

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.1
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.weights_hidden_output = np.random.randn(hidden_dim, output_dim) * 0.1
        self.bias_output = np.zeros((1, output_dim))

        # Store activations and gradients for visualization
        self.hidden_activations = None
        self.gradients_input_hidden = None

    def activation(self, x):
        if self.activation_fn == "tanh":
            return np.tanh(x)
        elif self.activation_fn == "relu":
            return np.maximum(0, x)
        elif self.activation_fn == "sigmoid":
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function.")

    def activation_derivative(self, x):
        if self.activation_fn == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == "relu":
            return np.where(x > 0, 1, 0)
        elif self.activation_fn == "sigmoid":
            sigmoid_x = 1 / (1 + np.exp(-x))
            return sigmoid_x * (1 - sigmoid_x)
        else:
            raise ValueError("Unsupported activation function.")

    def forward(self, X):
        self.input = X
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.activation(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = np.tanh(self.final_input)  # Output for binary classification
        self.hidden_activations = self.hidden_output
        return self.output

    def backward(self, X, y):
        # Compute output error
        output_error = self.output - y
        output_gradient = output_error * (1 - self.output ** 2)

        # Compute hidden layer error
        hidden_error = np.dot(output_gradient, self.weights_hidden_output.T)
        hidden_gradient = hidden_error * self.activation_derivative(self.hidden_input)

        # Update weights and biases
        self.weights_hidden_output -= self.lr * np.dot(self.hidden_output.T, output_gradient)
        self.bias_output -= self.lr * np.sum(output_gradient, axis=0, keepdims=True)
        self.weights_input_hidden -= self.lr * np.dot(X.T, hidden_gradient)
        self.bias_hidden -= self.lr * np.sum(hidden_gradient, axis=0, keepdims=True)

        # Store gradients for visualization
        self.gradients_input_hidden = hidden_gradient

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features
    hidden_features = mlp.hidden_activations
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title("Hidden Layer Activations")

    # Input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid)
    preds = (preds > 0).astype(int).reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, alpha=0.6, cmap='bwr')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title("Input Layer Decision Boundary")

    # Visualize gradients
    gradients = np.abs(mlp.gradients_input_hidden)
    ax_gradient.quiver(X[:, 0], X[:, 1], gradients[:, 0], gradients[:, 1], angles='xy', scale_units='xy', scale=0.1)
    ax_gradient.set_title("Gradient Visualization")

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num // 5, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=5)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
