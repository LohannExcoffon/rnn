class RNN():
    def __init__(self, input_size=1, hidden_size=100, output_size=1, learning_rate = 0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # init weights w/ rand
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
        h_prev = np.zeros((hidden_size, 1))
        xs, hs, ys = {}, {}, {}
        hs[-1] = h_prev
    
        for t in range(len(inputs)):
            x_t = np.array([[inputs[t]]])
            xs[t] = x_t
    
            h_t = self.tanh(np.dot(self.W_xh, x_t) + np.dot(self.W_hh, hs[t-1]) + self.b_h)
            hs[t] = h_t
    
            y_t = np.dot(self.W_hy, h_t) + self.b_y
            ys[t] = y_t
    
        return xs, hs, ys
