import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    e_z = np.exp(z - np.max(z, axis = 0, keepdims = True)) # numerical trick that prevents large values
    return e_z / np.sum(e_z, axis = 0, keepdims = True)

def lstm_cell(parameters, x_t, a_prev, c_prev):
    """
    Arguments:
    parameters -- weights and biases
    x_t -- input at time t, (n_x, m)
    a_prev -- previous hidden state, (n_a, m)
    c_prev -- previous cell state, (n_a, m)

    Returns:
    a_next -- next hidden state, (n_a, m)
    c_next -- next cell state, (n_a, m)
    y_t_pred -- predicted output at time t, (n_y, m)
    cache -- values for backprop

    """

    Wc = parameters['Wc']
    Wu = parameters['Wu']
    Wf = parameters['Wf']
    Wo = parameters['Wo']
    Wy = parameters['Wy']
    bc = parameters['bc']
    bu = parameters['bu']
    bf = parameters['bf']
    bo = parameters['bo']
    by = parameters['by']

    concat = np.concatenate(a_prev, x_t)

    cand = np.tanh(np.dot(Wc, concat) + bc)

    Gu = sigmoid(np.dot(Wu, concat) + bu)
    Gf = sigmoid(np.dot(Wf, concat) + bf)
    Go = sigmoid(np.dot(Wo, concat) + bo)

    c_next = Gu * cand + Gf * c_prev
    a_next = Go * np.tanh(c_next)
    y_t_pred = softmax(np.dot(Wy, a_next) + by)

    cache = (c_next, a_next, a_prev, c_prev, cand, x_t, Gu, Gf, Go, parameters)

    return c_next, a_next, y_t_pred

def lstm_forward(x, a0, parameters):
    """
    Arguments:
    x -- input sequence, (n_x, m, Tx)
    a0 -- initial hidden state, (n_a, m)
    parameters -- dictionary of weights and biases

    Returns:
    a -- hidden states for all time steps, (n_a, m, Tx)
    c -- hidden cells for all time staps, (n_c, m, Tx)
    y -- predictions for all time steps, (n_y, m, Tx)
    caches -- list of caches for backprop  

    """

    caches = []
    Tx = x.shape[2]
    n_a, m = a0.shape
    n_y = parameters['Wy'].shape[0]

    a = np.zeros((n_a, m, Tx))
    c = np.zeros((n_a, m, Tx))
    y = np.zeros((n_y, m, Tx))

    for t in range(Tx):
        x_t = x[:, :, t]
        c_prev, a_prev, y_t_pred = lstm_cell(parameters, x_t, a_prev, c_prev)
        y[:, :, t] = y_t_pred
        a[:, :, t] = a_prev 
        c[:, :, t] = c_prev
        caches.append(cache) 
    
    return a, c, y, caches
