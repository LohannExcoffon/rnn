import numpy as np
from RNN import RNN

sequence_length = 9
X = [] #preallocate arrays
y = []
for i in range(200): #200 training data points
    start = i * 0.1
    # generates consecutive sin values
    seq = np.sin(np.linspace(start, start + (sequence_length * 0.1), sequence_length))
    X.append(seq[:-1]) # take all but last of series and make training data
    y.append(seq[-1]) # take last value and that is solution to the series above
X = np.array(X)
y = np.array(y)
