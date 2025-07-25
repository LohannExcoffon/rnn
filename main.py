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

def loss_and_derivatives(ys, target):
    loss = 0
    dy = {}
    t = len(ys) - 1
    y = ys[t]
    loss += 0.5 * (y - target) ** 2
    dy[t] = (y - target)
    return loss, dy

def predict(start, length=5):
    h = np.zeros((rnn.hidden_size, 1))
    x = np.array([[start]])
    preds = []

    for _ in range(length):
        h = rnn.tanh(np.dot(rnn.W_xh, x) + np.dot(rnn.W_hh, h) + rnn.b_h)
        y = np.dot(rnn.W_hy, h) + rnn.b_y
        preds.append(y.item())
        x = y
    return preds



# --------------- TRAIN ---------------
def train(model, epochs=100):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            inputs = X[i]
            target = np.array([[y[i]]]) #reshape as needed

            xs, hs, ys = model.forward(inputs)
            loss, dy = loss_and_derivatives(ys, target)
            dW_xh, dW_hh, dW_hy, db_h, db_y = model.backward(xs, hs, ys, dy)
            model.update_params(dW_xh, dW_hh, dW_hy, db_h, db_y)
            total_loss += loss.sum()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")

# --------------- TEST ---------------
def test(model, input):
    # get predictions
    xs, hs, ys = rnn.forward(input)
    pred = ys[len(input) - 1]
    print('Prediction is:', pred[0][0])
    return pred[0][0]
    


test_input = np.sin(np.linspace(1, 3.6, 10))
true_pred = test_input[-1]
test_input = test_input[:-1]
print(test_input)
print(true_pred)
epochs = 300
sizes = [50, 100, 200, 300, 400, 500, 1000]
errors = {}
for size in sizes:
    rnn = RNN(input_size=1, hidden_size=size)
    train(rnn, epochs)
    output = test(rnn, test_input)
    errors[size] = (true_pred - output)**2
print(errors)
    