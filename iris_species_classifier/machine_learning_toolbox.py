import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np 

# Load in the Iris dataset
iris_df = pd.read_csv('irisdata.csv')

features = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
species = iris_df['species'].values

# Split the dataset into training and testing sets
train_features, test_features, species_train, species_test = train_test_split(features, species, test_size=0.3, random_state=42)

# Standardize the input features and scale them 
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

#List containing all of the non-linear activation functions 
activation_functions = ['logistic', 'relu', 'tanh']
epoch_iteration = [50, 100, 150, 200]

for activation in activation_functions:
    # Build and train the neural network model
    clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=200, activation=activation, solver='adam', random_state=42)
    fit_data = clf.fit(train_features, species_train)

    # Predictions on the test set
    y_pred = clf.predict(test_features)
    
    # Evaluate the model
    accuracy = accuracy_score(species_test, y_pred)
    precision = precision_score(species_test, y_pred, average='weighted')
    f1 = f1_score(species_test, y_pred, average='weighted')
    
    # Print the evaluation metrics
    print(f"\nEvaluation Metrics for Activation Function: {activation}")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("F1 Score:", f1)
    
    # Plot the loss curve at specified epochs
    epochs = len(fit_data.loss_curve_)
    iterations = range(0, epochs)
    plt.plot(iterations, fit_data.loss_curve_, label=activation)
    for epoch in epoch_iteration:
         if epoch < epochs:
            plt.scatter(epoch, fit_data.loss_curve_[epoch], color='lightcoral', marker='o', label=f'Epoch {epoch}')

    plt.title(f'Loss Curve during Training for Activation Functions')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

plt.show()

def plot_functions():
    # Define the activation functions
    def relu(x):
        return np.maximum(0, x)

    def logistic(x):
        return 1 / (1 + np.exp(-x))

    def tanh(x):
        return np.tanh(x)

    # Generate x values
    x_values = np.linspace(-5, 5, 100)

    # Compute y values for each activation function
    relu_values = relu(x_values)
    logistic_values = logistic(x_values)
    tanh_values = tanh(x_values)

    # Plot the activation functions on the same graph 
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(x_values, relu_values, label='ReLU')
    plt.title('ReLU Function')
    plt.xlabel('x')
    plt.ylabel('ReLU(x)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(x_values, logistic_values, label='Logistic (Sigmoid)')
    plt.title('Logistic (Sigmoid) Function')
    plt.xlabel('x')
    plt.ylabel('Logistic(x)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(x_values, tanh_values, label='Tanh')
    plt.title('Tanh Function')
    plt.xlabel('x')
    plt.ylabel('tanh(x)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_functions()