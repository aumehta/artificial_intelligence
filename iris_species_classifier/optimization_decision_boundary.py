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

#Function to predict values 
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

#Function to sum gradient values 
def gradient_summation(weight, bias, data, true_values):
    class_values = predict(data, weight, bias)
    bias_gradient = np.sum(class_values - true_values)
    gradient = np.sum((class_values - true_values)[:, np.newaxis] * data, axis=0)
    return gradient, bias_gradient

def update_weights_and_bias(weight, bias, data, goal, step_size):
    gradient, bias_gradient = gradient_summation(weight, bias, data, goal)
    weight -= step_size * gradient
    bias -= step_size * bias_gradient
    return weight, bias

#Gradient descent function 
def gradient_descent(data, initial_weights, initial_bias, true_values, learning_rate, number_iterations):
    weight_value = initial_weights
    bias_value = initial_bias
    list_errors = []
    list_weights = []
    list_bias = []
    for _ in range(number_iterations):
        iteration_mse = calc_MSE(data, weight_value, bias_value, true_values)
        #Adds corresponding errors, weight, and bias to list 
        list_errors.append(iteration_mse)
        list_weights.append(weight_value.copy())
        list_bias.append(bias_value)
        #Stopping criterion
        if iteration_mse < 8:
            break
        weight_grad, bias_grad = gradient_summation(weight_value, bias_value, data, true_values)
        # Use vectorized operations for weight and bias updates
        weight_value -= learning_rate * weight_grad
        bias_value -= learning_rate * bias_grad
    middle_weights = get_middle_element(list_weights)
    middle_bias = get_middle_element(list_bias)
    return weight_value, bias_value, middle_weights, middle_bias, list_errors

def get_middle_element(arr):
    return np.array(arr)[len(arr) // 2]

def plot_learningcurve(error_list):        
    # Plotting the learning curve
    plt.plot( np.arange(len(error_list)), error_list)
    # Adding title and labels
    plt.title("Error Reduction")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error") 
    # Display the plot
    plt.show()

def plot_decision_boundary(weights, bias, title):
    # plotData()
    # Compute decision boundary using intercept and slope
    x_vals = np.array([2.5,7])
    y_vals_intercept_slope = -(weights[0]*x_vals+bias)/weights[1]
    # Plot decision boundary
    plt.plot(x_vals, y_vals_intercept_slope, color='green', label='Decision Boundary', linewidth=2)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_classifier_output(data, weights, bias):    
    predicted_classes = predict(data, weights,bias)
    plt.scatter(data[predicted_classes == 0][:, 0], data[predicted_classes == 0][:, 1], label='Versicolor', color='red', marker='o')
    plt.scatter(data[predicted_classes == 1][:, 0], data[predicted_classes == 1][:, 1], label='Virginica', color='blue', marker='+')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('Data with Linear Boundaries')
    plt.legend()

def initialize_random_parameters(low_w, high_w, low_b, high_b, size):
    weights = np.random.uniform(low_w, high_w, size)
    bias = np.random.uniform(low_b, high_b)
    return weights, bias

def main():
    iris_values = iris[iris['species'] != 'setosa']
    iris_values = iris_values.iloc[:, :-1]
    iris_values = iris_values.iloc[:, 2:4]
    low_weight, high_weight = -10.0, 10.0
    low_bias, high_bias = -10, 10
    size_of_weights = 2  # Assuming you want two weights

    initial_weights, initial_bias = initialize_random_parameters(low_weight, high_weight, low_bias, high_bias, size_of_weights)
    learning_rate = 0.01
    max_iterations = 1000
    final_weights, final_bias, middle_weights, middle_bias, error_list = gradient_descent(
        iris_values.to_numpy(), initial_weights, initial_bias, class_2_and_3['binary_species'].to_numpy(), learning_rate, max_iterations
    )
    
    plot_classifier_output(np.array(iris_values),initial_weights, initial_bias)
    plot_decision_boundary(initial_weights, initial_bias, 'Initial Decision Boundary on Scatter Plot of Versicolor and Virginica Iris Classes')
    plot_classifier_output(np.array(iris_values),middle_weights, middle_bias)
    plot_decision_boundary(middle_weights, middle_bias, 'Intermediate Decision Boundary on Scatter Plot of Versicolor and Virginica Iris Classes')
    plot_classifier_output(np.array(iris_values),final_weights, final_bias)
    plot_decision_boundary(final_weights, final_bias, 'Final Decision Boundary on Scatter Plot of Versicolor and Virginica Iris Classes')
    plot_learningcurve(error_list)
    plt.show()

if __name__ == "__main__":
    main()