# LSTM from Scratch in Python

This repository contains a NumPy-based implementation of a Long Short-Term Memory (LSTM) network from scratch in Python. It demonstrates the main components and equations behind the LSTM cell and forward propagation through time.

## Overview

LSTM is a type of recurrent neural network (RNN) architecture that mitigates the vanishing gradient problem by introducing memory cells and gating mechanisms, which regulate the flow of information.

This implementation covers key formulas for:

- Forget gate
- Update gate (also called input gate)
- Candidate memory cell
- Output gate
- Memory cell state update
- Hidden state update
- Prediction via softmax output layer

***

## Key Functions and Formulas

### Activation functions

- **Sigmoid function:**

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Used for gating mechanisms to output values between 0 and 1.

- **Softmax function:**

$$
\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
$$

Implemented with a numerical stability trick by subtracting the max value.

***

### LSTM Cell Computation

At each time step $$ t $$, given the input $$ x_t $$, previous hidden state $$ a_{t-1} $$, and previous cell state $$ c_{t-1} $$:

1. Concatenate the previous hidden state and input:

$$
\text{concat} = [a_{t-1}, x_t]
$$

2. Compute the candidate memory cell:

$$
\tilde{c}_t = \tanh(W_c \cdot \text{concat} + b_c)
$$

3. Compute the update gate:

$$
u_t = \sigma(W_u \cdot \text{concat} + b_u)
$$

4. Compute the forget gate:

$$
f_t = \sigma(W_f \cdot \text{concat} + b_f)
$$

5. Compute the output gate:

$$
o_t = \sigma(W_o \cdot \text{concat} + b_o)
$$

6. Update the cell state:

$$
c_t = u_t \odot \tilde{c}_t + f_t \odot c_{t-1}
$$

7. Compute the next hidden state:

$$
a_t = o_t \odot \tanh(c_t)
$$

8. Compute the prediction output by passing the hidden state through a softmax layer:

$$
y_t = \text{softmax}(W_y \cdot a_t + b_y)
$$

***

## Code Usage

- `lstm_cell(parameters, x_t, a_prev, c_prev)`: Implements the calculations above for one time step.
- `lstm_forward(x, a0, parameters)`: Computes the LSTM cell forward pass for a sequence of inputs $$x$$.

### Arguments

- `x_t`: Input at time $$t$$, shape $$(n_x, m)$$.
- `a_prev`: Previous hidden state, shape $$(n_a, m)$$.
- `c_prev`: Previous cell state, shape $$(n_a, m)$$.
- `parameters`: Dictionary storing weights and biases for gates and output.

***

## Parameters Dictionary

- $$W_c, W_u, W_f, W_o, W_y$$: Weight matrices for candidate cell, update, forget, output gates, and output layer.
- $$b_c, b_u, b_f, b_o, b_y$$: Corresponding bias vectors.

***

## Example

Below is a snippet of the main LSTM cell function with the core formulas:

```python
concat = np.concatenate((a_prev, x_t))
cand = np.tanh(np.dot(Wc, concat) + bc)        # Candidate cell state
Gu = sigmoid(np.dot(Wu, concat) + bu)          # Update gate
Gf = sigmoid(np.dot(Wf, concat) + bf)          # Forget gate
Go = sigmoid(np.dot(Wo, concat) + bo)          # Output gate
c_next = Gu * cand + Gf * c_prev                # Cell state update
a_next = Go * np.tanh(c_next)                    # Hidden state update
y_t_pred = softmax(np.dot(Wy, a_next) + by)    # Output prediction
```

***

## Future improvements
- Backpropagation
- Training
