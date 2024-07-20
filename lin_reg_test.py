import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LinearRegression import LinearRegression as ln


def main():
    data = pd.read_csv("data.csv")
    # data = pd.read_csv("data_for_lr.csv")
    data = data.dropna()
    train_input = np.array(data[data.columns[0]]) 
    train_output = np.array(data[data.columns[1]])
    plt.scatter(train_input, train_output)
    model = ln()
    model.train(train_input, train_output, 0.00000000001, 100000)
    plt.plot(train_input, model.forward_propagation(train_input), 'r')
    plt.show()


if __name__ == "__main__":
    main()
