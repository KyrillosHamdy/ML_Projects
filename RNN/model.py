import numpy as np

# parameter initialization
def initialize_parameters(n_a, n_x, n_y):
    """
    n_a -- hidden state size
    n_x -- input size
    n_y -- output size

    Return parameters => dictionary containing weights and biases

    """
    np.random.seed(1)

    Waa = np.random.randn(n_a, n_a) * 0.01
    Wax = np.random.randn(n_a, n_x) * 0.01
    Wya = np.random.randn(n_y, n_a) * 0.01
    ba = np.zeros((n_a, 1))
    by = np.zeros((n_y, 1))

    parameters = {'Waa': Waa, 'Wax': Wax, 'Wya': Wya, 'ba': ba, 'by': by }

    return parameters

# Softmax Function
def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Forward propagation for single time step
def rnn_step_forward(parameters, x_t, a_prev):
    """
    parameters -- dictionary containing weights and biases
    a_prev -- hidden state from previous step (n_a, 1)
    x_t -- input at time step t (n_x, 1)

    Return:
    - a_next -- next hidden state
    - y_hat -- prediction at this time step
    - cache -- values stored for backprop

    """

    a_next = np.tanh(np.dot(parameters['Wax'], x_t) + np.dot(parameters['Waa'], a_prev) + parameters['ba'])
    z = np.dot(parameters['Wya'], a_next) + parameters['by']
    y_hat  = softmax(z)

    cache = (a_next, a_prev, x_t, parameters)

    return a_next, y_hat, cache

# Forward propagation over the entire sequence
def rnn_forward(x, a0, parameters):
    """
    x -- input sequence (n_x, m, T_x) (features, batch, time)
    a0 -- initial hidden state (n_a, m)
    parameters -- dictionary of parameters

    Return:
    - a -- hidden states for all time steps
    - y_hat -- prediction for all time steps
    - caches -- values for back prop

    """
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape

    a = np.zeros((n_a, m, T_x))
    y_hat = np.zeros((n_y, m, T_x))
    caches = []

    a_next = a0

    for t in range(T_x):
       a_next, y_hat_t, cache = rnn_step_forward(parameters, x[:, :, t], a_next)
       a[:, :, t] = a_next
       y_hat[:, :, t] = y_hat_t
       caches.append(cache)

    return a, y_hat, caches

# Compute Loss: Cross-entropy loss
def compute_loss(y_hat, y):
    """
    y_hat -- predictions (n_y, m, T_x)
    y -- true label(n_y, m, T_x)

    Return:
    loss -- scalar

    """
    loss = -np.sum(y * np.log(y_hat + 1e-8)) / y.shape[1]
    return loss

# Bachward propagation for single time step
def rnn_step_backward(dy, da_next, cache):
    """
    dy -- gradient of loss with respect to prediction
    da_next -- gradient with respect to next hidden state
    cache -- values from forward prop

    Return:
    - grads -- dictionary of gradients wrt Waa, wax, Wya, ba, by
    - da_prev -- gradient wrt previous hidden state

    """
    (a_next, a_prev, x_t, parameters) = cache
    Waa, Wax, Wya, ba, by =  parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['ba'], parameters['by']

    # grads wrt output layer
    dWya = np.dot(dy, a_next.T)
    dby = np.sum(dy, axis = 1, keepdims=True)

    # backprop into hidden state
    da = np.dot(Wya.T, dy) + da_next

    # gradient of tanh
    dz = (1 - a_next ** 2) * da

    # grads wrt parameters of hidden state
    dWax = np.dot(dz, x_t.T)
    dWaa = np.dot(dz, a_prev.T)
    dba = np.sum(dz, axis = 1, keepdims = True)

    # grads wrt previous hidden state
    da_prev = np.dot(Waa.T, dz)

    grads = {'dWya' : dWya, 'dby': dby, 'dWax': dWax, 'dWaa': dWaa, 'dba': dba}

    return grads, da_prev 

# Backward Propagation Through Time (BPTT)
def rnn_backward(y_hat, y, caches):
    """
    y_hat -- predictions
    y -- true labels
    caches -- from forward pass

    Return:
    grads -- accmulated gradients

    """ 
    # extract dims
    (a_next, a_prev, x_t, parameters) = caches[0] 
    n_a, m = a_next.shape
    T_x = y.shape[2]

    # initialize all grads as zeros
    dWaa = np.zeros_like(parameters["Waa"])
    dWax = np.zeros_like(parameters["Wax"])
    dWya = np.zeros_like(parameters["Wya"])
    dba = np.zeros_like(parameters["ba"])
    dby = np.zeros_like(parameters["by"])
    da_next = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        dy = y_hat[:, :, t] - y[:, :, t]
        grads_t, da_next = rnn_step_backward(dy, da_next, caches[t])
        dWya += grads_t['dWya']
        dby += grads_t['dby']
        dWax += grads_t['dWax']
        dWaa += grads_t['dWaa']
        dba += grads_t['dba']

    grads = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": dba, "dby": dby}

    return grads

# Update Parameters
def update_parameters(parameters, grads, lr = 0.01):
    parameters["Waa"] -= lr * grads["dWaa"]
    parameters["Wax"] -= lr * grads["dWax"]
    parameters["Wya"] -= lr * grads["dWya"]
    parameters["ba"] -= lr * grads["dba"]
    parameters["by"] -= lr * grads["dby"]
    
    return parameters

# Training Loob
def train_rnn(x, y, n_a, num_epoches = 100, lr = 0.01):
    n_x, m, T_x = x.shape
    n_y = y.shape[0]

    parameters = initialize_parameters(n_a, n_x, n_y)
    a0 = np.zeros((n_a, m))

    for epoch in range(num_epoches):
        a, y_hat, caches = rnn_forward(x, a0, parameters)
        loss = compute_loss(y_hat, y)
        grads = rnn_backward(y_hat, y, caches)
        parameters = update_parameters(parameters, grads, lr)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss:.4f}")
    
    return parameters


# ----- Preprocessing -----
def build_vocab(text):
    chars = sorted(list(set(text)))
    char_to_ix = {ch:i for i,ch in enumerate(chars)}
    ix_to_char = {i:ch for i,ch in enumerate(chars)}
    return chars, char_to_ix, ix_to_char

def one_hot_encode(text, char_to_ix, vocab_size):
    T_x = len(text)-1  # last char is target only
    x = np.zeros((vocab_size, 1, T_x))
    y = np.zeros((vocab_size, 1, T_x))
    for t in range(T_x):
        x[char_to_ix[text[t]], 0, t] = 1
        y[char_to_ix[text[t+1]], 0, t] = 1
    return x, y

# ----- Example Training -----
text = "hello world"
chars, char_to_ix, ix_to_char = build_vocab(text)
vocab_size = len(chars)

x, y = one_hot_encode(text, char_to_ix, vocab_size)

trained_params = train_rnn(x, y, n_a=32, num_epoches=2000, lr=0.05)

# ----- Sampling from trained RNN -----
def sample(parameters, char_to_ix, ix_to_char, seed_char, length=20):
    vocab_size = len(char_to_ix)
    a_prev = np.zeros((parameters['Waa'].shape[0], 1))

    x = np.zeros((vocab_size, 1))
    x[char_to_ix[seed_char]] = 1

    output = seed_char
    for t in range(length):
        a_prev, y_hat, _ = rnn_step_forward(parameters, x, a_prev)
        idx = np.random.choice(range(vocab_size), p=y_hat.ravel())
        char = ix_to_char[idx]
        output += char
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
    return output

print("\nGenerated text:\n", sample(trained_params, char_to_ix, ix_to_char, seed_char="h"))