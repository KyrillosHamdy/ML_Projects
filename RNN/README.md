# 📘 Recurrent Neural Network from Scratch (NumPy)

implementing a **vanilla Recurrent Neural Network (RNN)** from scratch using only **NumPy**, including forward propagation, backpropagation through time (BPTT), training loop, and text generation.

---

## 🚀 Features

* RNN implemented from scratch with no external deep learning frameworks.
* One-hot encoding for character-level text modeling.
* Cross-entropy loss function.
* Backpropagation through time (BPTT).
* Gradient descent parameter updates.
* Sampling/generation from trained RNN.

---

## 🧮 Mathematical Formulas

### 1. Hidden State Update

For each time step ( t ):

$$
a^{\langle t \rangle} = \tanh(W_{ax} x^{\langle t \rangle} + W_{aa} a^{\langle t-1 \rangle} + b_a)
$$

* $$W_{ax} \in \mathbb{R}^{n_a \times n_x}$$
* $$W_{aa} \in \mathbb{R}^{n_a \times n_a}$$
* $$b_a \in \mathbb{R}^{n_a \times 1}$$
* $$x^{\langle t \rangle} \in \mathbb{R}^{n_x \times 1}$$  (one-hot encoded input)

---

### 2. Output Prediction

$$
z^{\langle t \rangle} = W_{ya} a^{\langle t \rangle} + b_y
$$

$$
\hat{y}^{\langle t \rangle} = \text{softmax}(z^{\langle t \rangle})
$$

* $$W_{ya} \in \mathbb{R}^{n_y \times n_a}$$
* $$b_y \in \mathbb{R}^{n_y \times 1}$$
* $$\hat{y}^{\langle t \rangle} \in \mathbb{R}^{n_y \times 1}$$ (predicted probability distribution)

---

### 3. Loss Function (Cross-Entropy)

$$
\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{t=1}^{T_x} y^{\langle t \rangle (i)} \cdot \log\left( \hat{y}^{\langle t \rangle (i)} \right)
$$

Where:

* $$m$$ = number of training examples
* $$T_x$$ = sequence length
* $$y^{\langle t \rangle}$$ = one-hot true label

---

### 4. Backpropagation Through Time (BPTT)

For each time step $$t$$:

$$
dW_{ya} = \sum_t \left( \frac{\partial \mathcal{L}}{\partial \hat{y}^{\langle t \rangle}} \cdot {a^{\langle t \rangle}}^T \right)
$$

$$
dW_{ax} = \sum_t \left( dz^{\langle t \rangle} \cdot {x^{\langle t \rangle}}^T \right)
$$

$$
dW_{aa} = \sum_t \left( dz^{\langle t \rangle} \cdot {a^{\langle t-1 \rangle}}^T \right)
$$

$$
d b_a = \sum_t dz^{\langle t \rangle}, \quad d b_y = \sum_t d y^{\langle t \rangle}
$$

Where:

$$
dz^{\langle t \rangle} = (1 - {a^{\langle t \rangle}}^2) \odot \left( W_{ya}^T d y^{\langle t \rangle} + d a^{\langle t \rangle} \right)
$$

---

### 5. Parameter Update (Gradient Descent)

$$
\theta = \theta - \eta \cdot d\theta
$$

where $$\eta$$ is the learning rate.

---

## 📂 Project Structure

```
model.py   # Full RNN implementation
README.md  # Documentation (this file)
```

---

## 🔧 Usage

### 1. Install Requirements

No external libraries are required besides NumPy:

```bash
pip install numpy
```

### 2. Training Example

Inside `model.py`:

```python
text = "hello world"
chars, char_to_ix, ix_to_char = build_vocab(text)
x, y = one_hot_encode(text, char_to_ix, len(chars))

trained_params = train_rnn(x, y, n_a=32, num_epoches=2000, lr=0.05)
```

### 3. Generate Text

```python
print(sample(trained_params, char_to_ix, ix_to_char, seed_char="h"))
```

---

## 📝 Future Improvments

*  Implement GRU/LSTM versions
*  Save/load trained models
