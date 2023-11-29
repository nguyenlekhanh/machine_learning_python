import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_classification.csv')

#print(data)

#print 2d array
#print(data.values)

#classify true, false
true_x = []
true_y = []
false_x = []
false_y = []

for item in data.values:
    if item[2] == 1.:
        true_x.append(item[0])
        true_y.append(item[1])
    else:
        false_x.append(item[0])
        false_y.append(item[1])

#show classify true and false base on pass column
# plt.scatter(true_x, true_y, marker='o', c='b')
# plt.scatter(false_x, false_y, marker='s', c='r')
# plt.show()

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def classify(p):
    if p >= 0.5:
        return 1
    else:
        return 0
    
def predict(features, weights):
    z = np.dot(features, weights)
    return sigmoid(z)


def cost_function(features, labels, weights):
    n = len(labels)
    predictions = predict(features, weights)

    cost_class1 = - labels * np.log(predictions)
    cost_class2 = - (1-labels) * np.log(1 - predictions)
    cost = cost_class1 + cost_class2

    return cost.sum() / n


def update_weight(features, labels, weights, learning_rate):
    n = len(labels)

    predictions = predict(features, weights)
    gradient = np.dot(features.T, (predictions - labels))

    gradient = gradient/n
    gradient = gradient * learning_rate
    weights = weights - gradient
    return weights


def train(features, labels, weights, learning_rate, iter):
    cost_history = []
    for i in range(iter):
        #weights include weight0 = bias
        weights = update_weight(features, labels, weights, learning_rate)
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)

    return weights, cost_history


# data = [
#     [4.855064, 9.639962, 1],
#     [8.62544, 0.058927, 0],
#     [3.828192, 0.723199, 0],
#     [7.150955, 3.89942, 1],
#     [6.4779, 8.198181, 1]
# ]


# columns = ['feature1', 'feature2', 'label']
# data = pd.read_csv('data_classification.csv')
# dataframe = pd.DataFrame(data, columns=columns)
# print(data)

# Extract features (X) and labels (y)
X = data[['feature1', 'feature2']].values
y = data['label'].values
#print(X)
#print(y)

# Initialize weights and other parameters
initial_weights = np.zeros(X.shape[1] + 1)  # +1 for bias
learning_rate = 0.01
iterations = 16000

#print(initial_weights)

# Call the train function
trained_weights, cost_history = train(np.insert(X, 0, 1, axis=1), y, initial_weights, learning_rate, iterations)

# Print the trained weights and cost history
print("Trained Weights:", trained_weights)
#print("Cost History:", cost_history)

# Example usage for prediction
new_data_point = np.array([4.0, 1.0])  # Replace with your new data
new_data_point_with_bias = np.insert(new_data_point, 0, 1)
prediction = predict(new_data_point_with_bias, trained_weights)

# Classify the prediction
classification = classify(prediction)

print(f"Prediction: {prediction}, Classified as: {classification}")

# Visualize the decision boundary (optional)
plt.scatter(true_x, true_y, marker='o', c='b', label='Class 1')
plt.scatter(false_x, false_y, marker='s', c='r', label='Class 0')

x_boundary = np.linspace(min(data['feature1']), max(data['feature1']), 100)
y_boundary = (-trained_weights[0] - trained_weights[1] * x_boundary) / trained_weights[2]
plt.plot(x_boundary, y_boundary, label='Decision Boundary', c='g')

plt.legend()
plt.show()