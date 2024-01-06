import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
iris = pd.read_csv('irisdata.csv')

class_2_and_3 = iris[iris['species'].isin(['versicolor', 'virginica'])]

species_to_color = {'versicolor': 'red', 'virginica': 'blue'}
species_to_marker = {'versicolor': 'o', 'virginica': '+'}


# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Single neuron neural network 
def compute_neural_network(data, weights, bias):
     return sigmoid(np.dot(data, weights) + bias)

def predict(data, weights, bias): 
    output =  compute_neural_network(data,weights,bias)
    prediction = [] 
    for i in output: 
        if i > 0.5: 
            prediction.append(1)
        else:
            prediction.append(0)
    return np.array(prediction)

def plotData():
    for species, group in class_2_and_3.groupby('species'):
        plt.scatter(group['petal_length'], group['petal_width'], c=species_to_color[species],
                    marker=species_to_marker[species], label=f'{species} Iris')

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('Scatter Plot of Versicolor and Virginica Iris Classes')
    plt.legend()

def plot_decision_boundary(weights, bias):
    # plotData()
    # Compute decision boundary using intercept and slope
    x_vals = np.array([2.5,7])
    y_vals_intercept_slope = -(weights[0]*x_vals+bias)/weights[1]

    # Plot decision boundary
    plt.plot(x_vals, y_vals_intercept_slope, color='green', label='Decision Boundary', linewidth=2)

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('Scatter Plot of Versicolor and Virginica Iris Classes')
    plt.legend()
    plt.show()


def plot_classifier_output(data, weights, bias):
    predicted_classes = predict(data, weights,bias)

    # Plot points with different colors based on predicted classes
    plt.scatter(data[predicted_classes == 1][:, 0], data[predicted_classes == 1][:, 1], label='Versicolor', color='red', marker='o')
    plt.scatter(data[predicted_classes == 0][:, 0], data[predicted_classes == 0][:, 1], label='Virginica', color='blue', marker='+')

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('Data with Linear Boundaries')
    plt.legend()


def plot_learning_curve(weights, bias):
    # Generate input space
    petal_length_vals = np.linspace(3, 7, 200)
    petal_width_vals = np.linspace(1, 3, 200)
    # Create a meshgrid for the input space
    petal_length, petal_width = np.meshgrid(petal_length_vals, petal_width_vals)
    input_space = np.c_[petal_length.flatten(), petal_width.flatten()]
    # Compute the output for each point in the input space
    output_space = compute_neural_network(input_space, weights, bias)
    # Reshape the output to match the shape of the meshgrid
    output_space = output_space.reshape(petal_length.shape)
    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(petal_length, petal_width, output_space, cmap='viridis', alpha=0.8)
    # Set labels and title
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.set_zlabel('Neural Network Output')
    ax.set_title('Neural Network Output and Iris Data')
    plt.show()

def main(): 
    iris_values = iris[iris['species'] != 'setosa']
    iris_values = iris_values.iloc[:, :-1]
    iris_values = iris_values.iloc[:, 2:4]
    weights = np.array([-2,-3])
    bias = 14.5
    plotData()
    plt.show() 
    plotData() 
    plot_decision_boundary(weights,bias)
    plt.show() 
    plot_learning_curve(weights, bias)
    plot_classifier_output(np.array(iris_values),weights, bias)
    plot_decision_boundary(weights,bias)
    plt.show()

if __name__ == "__main__":
    main()




