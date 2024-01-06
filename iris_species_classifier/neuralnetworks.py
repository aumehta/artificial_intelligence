import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd 
import numpy as np
iris = pd.read_csv('irisdata.csv')

class_2_and_3 = iris[iris['species'].isin(['versicolor', 'virginica'])].copy()
class_2_and_3.loc[:, 'binary_species'] = class_2_and_3['species'].map({'versicolor': 0, 'virginica': 1})
species_to_color = {'versicolor': 'red', 'virginica': 'blue'}
species_to_marker = {'versicolor': 'o', 'virginica': '+'}

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Single neuron neural network 
def compute_neural_network(data, weights, bias):
     return sigmoid(np.dot(data, weights) + bias)

#Prediction method
def predict(data, weights, bias): 
    output =  compute_neural_network(data,weights,bias)
    prediction = [] 
    for i in output: 
        if i > 0.5: 
            prediction.append(1)
        else:
            prediction.append(0)
    return np.array(prediction)
     
#Function to calculate MSE 
def calc_MSE(data, weights, bias, true_values):
    predicted_values = predict(data,weights,bias)
    mse_value = 0 
    for index, value in enumerate(predicted_values):
        mse_value = mse_value + ((predicted_values[index] - true_values[index])**2)
    return mse_value/2

#Function to plot data 
def plotData():
    for species, group in class_2_and_3.groupby('species'):
        plt.scatter(group['petal_length'], group['petal_width'], c=species_to_color[species],
                    marker=species_to_marker[species], label=f'{species} Iris')

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('Scatter Plot of Versicolor and Virginica Iris Classes')
    plt.legend()

#Function to plot decision boundary 
def plot_decision_boundary(weights, bias, label, color, title):
    plotData()

    # Compute decision boundary using intercept and slope
    x_vals = np.array([2.5, 7])
    y_vals_intercept_slope = -(weights[0] * x_vals + bias) / weights[1]

    # Plot decision boundary
    plt.plot(x_vals, y_vals_intercept_slope, color=color, label=label, linewidth=2)

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title(title)
    plt.legend()

#Function for gradient summation 
def gradient_summation(weight, bias, data, true_values):
    class_values = predict(data, weight, bias)
    bias_gradient = np.sum(class_values - true_values)
    #Takes sum over all values
    gradient = np.sum((class_values - true_values)[:, np.newaxis] * data, axis=0)
    return gradient, bias_gradient

# Updates weight and bias for each gradient 
def update_weights_and_bias(weight, bias, data, goal, step_size):
    gradient, bias_gradient = gradient_summation(weight, bias, data, goal)
    weight -= step_size * gradient
    bias -= step_size * bias_gradient
    return weight, bias

# Plots gradient boundary 
def gradient_boundary(data, weights, bias, true_values, learning_rate, iterations):
    for _ in range(iterations):
        plot_decision_boundary(weights, bias, f'Iteration {_}', 'black', f'Boundary at Iteration {_}')
        plt.show()
        weights, bias = update_weights_and_bias(weights, bias, data, true_values, learning_rate)
    # Plot the final boundary
    plot_decision_boundary(weights, bias, 'Final Boundary', 'black', 'Final Decision Boundary')
    plt.show()

def main():
    # # These are some random weights and bias:
    large_weights = np.array([-10, 10])
    large_bias = 30
    iris_values = iris[iris['species'] != 'setosa']
    iris_values = iris_values.iloc[:, :-1]
    iris_values = iris_values.iloc[:, 2:4]
    large_mse = calc_MSE(iris_values, large_weights, large_bias, class_2_and_3['binary_species'].to_numpy())
    print("The large error mse is " + str(large_mse))
    
    # # Plot decision boundary for random weights
    plot_decision_boundary(large_weights, large_bias, 'Large Error', 'green', 'Plot of Versicolor and Virginica Iris Classes for Large Error')
    plt.show()

    # These are close to optimal weights and bias:
    small_weights = np.array([0.054, 0.051])
    small_bias =  -0.32
    small_mse = calc_MSE(iris_values, small_weights, small_bias, class_2_and_3['binary_species'].to_numpy())
    print("The small error mse is " + str(small_mse))
    # Plot decision boundary for close to optimal weights
    plot_decision_boundary(small_weights, small_bias, 'Small Error', 'purple', 'Plot of Versicolor and Virginica Iris Classes for Small Error')
    # Show the plot
    plt.show()
    # Set the number of iterations (adjust as needed)
    iterations = 1
    # Call boundary_diff with large weights and bias
    gradient_boundary(iris_values.to_numpy(), small_weights, small_bias, class_2_and_3['binary_species'].to_numpy(), 0.0001, iterations)

    
    
if __name__ == "__main__":
    main()

