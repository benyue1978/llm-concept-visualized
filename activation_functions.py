import matplotlib.pyplot as plt
import numpy as np


# 定义常见激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def swish(x):
    return x * sigmoid(x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))

# 定义输入区间
x = np.linspace(-6, 6, 400)

# 重新计算
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_swish = swish(x)
y_gelu = gelu(x)

# 创建子图
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

axs[0, 0].plot(x, y_sigmoid, color="blue")
axs[0, 0].set_title("Sigmoid")
axs[0, 0].grid(True)

axs[0, 1].plot(x, y_tanh, color="orange")
axs[0, 1].set_title("Tanh")
axs[0, 1].grid(True)

axs[0, 2].plot(x, y_relu, color="green")
axs[0, 2].set_title("ReLU")
axs[0, 2].grid(True)

axs[1, 0].plot(x, y_leaky_relu, color="red")
axs[1, 0].set_title("Leaky ReLU")
axs[1, 0].grid(True)

axs[1, 1].plot(x, y_swish, color="purple")
axs[1, 1].set_title("Swish")
axs[1, 1].grid(True)

axs[1, 2].plot(x, y_gelu, color="brown")
axs[1, 2].set_title("GELU")
axs[1, 2].grid(True)

plt.tight_layout()
plt.show()