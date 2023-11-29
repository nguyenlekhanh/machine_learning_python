#pip install pandas
#pip install matplotlib

import pandas as pd
import matplotlib.pyplot as plt


dataframe = pd.read_csv('Advertising.csv')
#print(dataframe)


X = dataframe.values[:, 2]
y = dataframe.values[: , 4]

# plt.scatter(X, y, marker='o')
# plt.show()

def predict(new_radio, weight, bias):
    return weight * new_radio + bias


#MSE
def cost_function(X, y, weight, bias):
    n = len(X)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i] - (weight*X[i] + bias))**2

    return sum_error / 2


#gradient decenst
def update_weight(X, y, weight, bias, learning_rate):
    n = len(X)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2*X[i] * (y[i] - (weight*X[i] + bias))
        bias_temp += -2 * (y[i] - (weight*X[i] + bias))
    
    weight -= (weight_temp/n) * learning_rate
    bias -= (bias_temp/n) * learning_rate

    return weight, bias


def train (X, y, weight, bias, learning_rate, iter):

    cost_his = []

    for i in range(iter):

        weight, bias = update_weight(X, y, weight, bias, learning_rate)

        cost = cost_function(X, y, weight, bias)

        cost_his.append(cost)


    return weight, bias, cost_his


weight, bias, cost_his = train(X, y, 0.03, 0.0014, 0.001, 3000)


#print(weight)
#print(bias)
#print(cost_his)
print(predict(37, weight, bias))


repeat = [i for i in range(3000)]
plt.plot(repeat, cost_his)
plt.show()