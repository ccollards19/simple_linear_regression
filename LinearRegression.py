import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

class LinearRegression: 
    def __init__(self, train_input, train_output): 
        self.m = 0
        self.p = 0
        self.min = 0
        self.rng = 1
        self.train_input = train_input
        self.train_output = train_output

    def normalize(self):
        print(self.train_input)
        self.min = np.min(self.train_input)
        self.rng = np.max(self.train_input) - self.min
        self.train_input = np.divide(self.train_input - self.min, self.rng)
        self.train_input = self.train_input 
        print(self.train_input)
        

    def precision(self):
        return np.mean((np.multiply(self.m, self.train_input) + self.p) - self.train_output)

    # train model using descending gradient method
    def train(self, learning_rate, iters): 
        for epoch in range(iters):
            predictions = np.multiply(self.train_input, self.m) + self.p
            diff = predictions - self.train_output 
            dm = np.mean(np.multiply(diff, self.train_input)) 
            dp = np.mean(diff)
            self.m -= learning_rate * dm
            self.p -= learning_rate * dp
            print(f"epoch = {epoch}, m = {self.m}, p = {self.p}, error = {self.precision()}")
        return self.m, self.p, self.min, self.rng

def main():
    try :
        #get training data
        #data = pd.read_csv("data.csv")
        data = pd.read_csv("data_for_lr.csv")
        data = data.dropna()
        train_input = np.array(data[data.columns[0]])
        train_output = np.array(data[data.columns[1]])
        #train the model
        model = LinearRegression(train_input, train_output)
        model.normalize()
        m, p, minimum, rng= model.train(0.1, 1000)
        #save the weights and the normalisation parameters in a csv
        with open('weights.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([m, p, minimum, rng])
        #visualise the result
        train_input = np.divide(train_input - minimum, rng)
        plt.scatter(train_input, train_output)
        plt.plot(train_input, np.multiply(m, train_input) + p, 'r')
        plt.show()
    except e:
        print("ERROR\n");

if __name__ == "__main__":
    main()
