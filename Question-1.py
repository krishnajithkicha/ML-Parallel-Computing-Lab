# --- Imports ---
import numpy as np
import struct
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Load images ---
def load_images(path):
    with open(path, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num, rows, cols)

# --- Load labels ---
def load_labels(path):
    with open(path, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# --- Load dataset ---
x_train = load_images("train-images.idx3-ubyte")
y_train = load_labels("train-labels.idx1-ubyte")
x_test  = load_images("test-images.idx3-ubyte")
y_test  = load_labels("test-labels.idx1-ubyte")

# --- Normalize images ---
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# --- One-hot encode labels ---
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# --- Train / validation split ---
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)

# --- Output shapes ---
print("Training:", x_train.shape, y_train.shape)
print("Validation:", x_val.shape, y_val.shape)
print("Test:", x_test.shape, y_test.shape)
