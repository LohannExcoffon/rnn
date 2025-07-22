import numpy as np

class RNN():
    def __init__(self, input_size=1, hidden_size=100, output_size=1, learning_rate = 0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # init weights w/ rand and biases
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, inputs):
        h_prev = np.zeros((self.hidden_size, 1))
        xs, hs, ys = {}, {}, {} #preallocate dicts to store states
        hs[-1] = h_prev # no previous hidden layer
        for t in range(len(inputs)):
            x_t = np.array([[inputs[t]]]) #current time step
            xs[t] = x_t

            #matrix mult to calculate next hidden state based on input and previous hidden state
            h_t = self.tanh(np.dot(self.W_xh, x_t) + np.dot(self.W_hh, hs[t-1]) + self.b_h)
            hs[t] = h_t

            y_t = np.dot(self.W_hy, h_t) + self.b_y #get output state with updated hidden state
            ys[t] = y_t
        return xs, hs, ys
        
    def backward(self, xs, hs, ys, dy):
        dW_xh = np.zeros_like(self.W_xh) #preallocate to store gradients
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros_like(hs[0]) #no following hidden state
        for t in reversed(range(len(xs))): #back propagation loop
            if t not in dy:
                continue
            dW_hy += np.dot(dy[t], hs[t].T)
            db_y += dy[t]

            dh = np.dot(self.W_hy.T, dy[t]) + dh_next
            dt = dh * self.dtanh(hs[t])

            db_h += dt
            dW_xh += np.dot(dt, xs[t].T)
            dW_hh += np.dot(dt, hs[t-1].T)
            
            dh_next = np.dot(self.W_hh.T, dt)
        #clip gradients
        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dparam, -1, 1, out=dparam)
        return dW_xh, dW_hh, dW_hy, db_h, db_y

    def update_params(self, dW_xh, dW_hh, dW_hy, db_h, db_y):
        self.W_xh -= self.learning_rate * dW_xh
        self.W_hh -= self.learning_rate * dW_hh
        self.W_hy -= self.learning_rate * dW_hy
        self.b_h -= self.learning_rate * db_h
        self.b_y -= self.learning_rate * db_y
