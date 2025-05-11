import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import norm

data = loadmat('DAGGER\data22.mat')
X_i = data['X_i']
X_n = data['X_n']

N = 400
alpha = 0.0005
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(x))

def sigmoid_derivative(x):
    f2 = 1 / (1 + np.exp(x))
    return -f2 * (1 - f2)

def adam_optimizer_with_diagnostics(
    gradient_func, 
    z_init, 
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    max_iter=3000,
    lambda_power=0.01,
):
    z = z_init.copy()
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    P = np.zeros_like(z)
    J_history = []

    for t in range(1, max_iter + 1):
        J, grad = gradient_func(z)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        P = (1 - lambda_power) * P + lambda_power * (grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        normalization_factor = np.sqrt(P + epsilon)
        normalized_grad = m_hat / normalization_factor
        z -= learning_rate * normalized_grad
        J_history.append(J)

    return z, J_history

def compute_J_gradient(X_n_cropped, z, T, A_1, B_1, A_2, B_2):
    w1 = A_1 @ z + B_1[:, np.newaxis]
    z1 = relu(w1)
    w2 = A_2 @ z1 + B_2[:, np.newaxis]
    x = sigmoid(w2)
    Tx_minus_Xn = np.dot(T, x) - X_n_cropped
    J_value = np.log10(norm(Tx_minus_Xn)**2)
    Tx_minus_Xn = np.dot(T, x) - X_n_cropped
    norm_squared = np.linalg.norm(Tx_minus_Xn)**2 + epsilon
    u2 = (2 / np.log(10)) * (np.dot(T.T, Tx_minus_Xn) / norm_squared)
    v2 = u2 * sigmoid_derivative(w2)
    u1 = A_2.T @ v2
    v1 = u1 * relu_derivative(w1)
    u0 = A_1.T @ v1
    gradient = N * u0 + 2 * z
    return J_value, gradient

T = np.eye(N, 784)
data_21 = loadmat('data21.mat')
A_1, B_1 = data_21['A_1'], data_21['B_1'].flatten()
A_2, B_2 = data_21['A_2'], data_21['B_2'].flatten()

cropped_images = []
reconstructed_images = []
error_history = []

for i in range(X_n.shape[1]):
    X_n_cropped = np.dot(T, X_n[:, i].reshape(-1, 1))
    X_n_cropped_padded = np.zeros(784)
    X_n_cropped_padded[:X_n_cropped.shape[0]] = X_n_cropped.flatten()
    cropped_images.append(X_n_cropped_padded)
    z_init = np.random.normal(0, 1, (10, 1))
    z_opt, errors = adam_optimizer_with_diagnostics(
        lambda z: compute_J_gradient(X_n_cropped, z, T, A_1, B_1, A_2, B_2),
        z_init
    )
    error_history.append(errors)
    w1 = A_1 @ z_opt + B_1[:, np.newaxis]
    z1 = relu(w1)
    w2 = A_2 @ z1 + B_2[:, np.newaxis]
    reconstructed_x = sigmoid(w2)
    reconstructed_images.append(reconstructed_x)

plt.figure(figsize=(10, 5))
for i, errors in enumerate(error_history):
    plt.plot(errors, label=f'Image {i+1}')
plt.xlabel('Iteration')
plt.ylabel('Error (J(z))')
plt.legend()
plt.title('Error vs. Iteration')
plt.show()

plt.figure(figsize=(12, 6))
for i in range(X_n.shape[1]):
    plt.subplot(4, 3, i * 3 + 1)
    plt.imshow(X_i[:, i].reshape(28, 28).T, cmap='gray')
    plt.title('Original')
    plt.subplot(X_n.shape[1], 3, 3*i + 2)
    plt.imshow(cropped_images[i].reshape(28, 28).T, cmap='gray')
    plt.title(f'Noisy (N={N})')
    plt.subplot(4, 3, i * 3 + 3)
    plt.imshow(reconstructed_images[i].reshape(28, 28).T, cmap='gray')
    plt.title('Reconstructed')

plt.tight_layout()
plt.show()