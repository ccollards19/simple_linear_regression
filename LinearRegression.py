import numpy as np

class LinearRegression: 
    def __init__(self): 
        self.parameters = {}
        self.parameters['m'] = 0
        self.parameters['p'] = 0

    def approximate_value(self, value):
        return value * self.parameters['m'] + self.parameters['p']

    def forward_propagation(self, train_input):
        return np.multiply(self.parameters['m'], train_input) + self.parameters['p']

    # mean square
    def cost(self, test_input, test_output):
        predictions = self.forward_propagation(test_input)
        return np.mean((test_output - predictions) ** 2)

    # train model using descending graadient method
    def train(self, train_input, train_output, learning_rate, iters): 
        # Initialize random parameters
        if self.parameters['m'] == 0 and self.parameters['p'] == 0:
            size = train_input.size
            i = np.random.randint(0, size/2)
            j = np.random.randint(size/2, size)
            self.parameters['m'] = (train_output[j] - train_output[i]) / (train_input[j] - train_input[i])
            self.parameters['p'] = train_output[i] - self.parameters['m']*train_input[i] + 0.1
            # print("{}X + {}".format(self.parameters['m'], self.parameters['p']))
            # print("||||||||||||||||||||||||||||||||||")
        # train model
        for i in range(iters): 
            predictions = self.forward_propagation(train_input) 
            # backward propagation
            # print(predictions)
            # print(train_output)
            # print("||||||||||||||||||||||||||||||||||")
            diff = (predictions-train_output) 
            # print(diff)
            # print("||||||||||||||||||||||||||||||||||")
            dm = 2 * np.mean(np.multiply(train_input, diff)) # derivate for p
            # print(dm)
            # print("||||||||||||||||||||||||||||||||||")
            dp = 2 * np.mean(diff) # derivate for m
            # print(dp)
            # print("||||||||||||||||||||||||||||||||||")
            self.parameters['m'] = self.parameters['m'] - learning_rate * dm 
            self.parameters['p'] = self.parameters['p'] - learning_rate * dp
            # print("new {}X + {}".format(self.parameters['m'], self.parameters['p']))
            # print("||||||||||||||||||||||||||||||||||")

        return self.parameters['m'], self.parameters['p']
