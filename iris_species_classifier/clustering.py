import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import load_iris
import pandas as pd 
import numpy as np

iris = pd.read_csv('irisdata.csv')
class_2_and_3 = iris[iris['species'].isin(['versicolor', 'virginica'])].copy()
class_2_and_3.loc[:, 'binary_species'] = class_2_and_3['species'].map({'versicolor': 0, 'virginica': 1})
species_to_color = {'versicolor': 'red', 'virginica': 'blue'}
species_to_marker = {'versicolor': 'o', 'virginica': '+'}
# Extract numerical features only
numerical_data = iris.iloc[:, :4].to_numpy()


#Initialize uk to k random points 
# assign clusters
#estimate means
#loop from step 2 until converged
#Randomly select k centroids from the entire dataset
def create_centroids(data, k):
    indices = numpy.random.choice(len(data), k, False)
    arr = [] 
    for i in indices:
        arr.append(data[i])
    return np.array(arr)

#Assign data to clusters using the distortion 
def cluster_assignment(data, centroids):
    # Assign each data point to the cluster with the nearest centroid using Euclidean distance
    cluster_assignment = {}
    for row in data:
        distanceArray = [] 
        for centroid in centroids:
            distanceArray.append(distance(row, centroid))
        minDistance = np.argmin(distanceArray)
        chosenCluster = tuple(centroids[minDistance])
        cluster_assignment[chosenCluster] = cluster_assignment.get(chosenCluster, []) + [row]
    return cluster_assignment

#Function to calculate distance 
def distance(datapoint, cluster):
    euclidean_distance = np.sqrt(sum((x - y) ** 2 for x, y in zip(datapoint, cluster)))
    return euclidean_distance

#Function to calculate distortion 
def calculate_distortion(cluster_assignment):
    distortion = 0
    for cluster, assigned_data in cluster_assignment.items():
        for data_point in assigned_data:
            distortion += distance(data_point, cluster) ** 2
    return distortion

#Function to update centroids 
def update_centroids(cluster_assignment):
    centroids = []
    for cluster, assigned_data in cluster_assignment.items():
        if len(assigned_data) > 0:
            cluster_mean = np.mean(assigned_data, axis=0)
            centroids.append(tuple(cluster_mean))
    return centroids

#Iteration function 
def iterate(numerical_data, k, centroids):
    distortions = []  # Store distortions at each iteration
    iteration = 0
    previous_centroids = None
    while True:
        assignment = cluster_assignment(numerical_data, centroids)
        distortion = calculate_distortion(assignment)
        distortions.append(distortion)
        centroids = update_centroids(assignment)
        plot_clusters(centroids, assignment, k, f'{iteration} Iteration Scatter Plot Petal Width vs. Petal Length')
        plt.show()
        #Convergence stopping condition 
        if previous_centroids is not None and np.array_equal(previous_centroids, centroids):
            break
        previous_centroids = centroids.copy()
        iteration += 1
    plot_learning_curve(distortions, k, f'K-Means Learning Curve for K={k}')
    if k == 2:
        plot_clusters(centroids, assignment, k, f'{iteration} Iteration Scatter Plot Petal Width vs. Petal Length')
        plot_decision_boundary(np.asarray([0.56, 0.56]), -2.22, "Decision Boundary K=2 Optimized Parameters")
        plt.show()
    elif k == 3:
        plot_clusters(centroids, assignment, k, f'{iteration} Iteration Scatter Plot Petal Width vs. Petal Length')
        plot_decision_boundary(np.asarray([0.56, 0.56]), -1.53, "Decision Boundary K=3 Optimized Parameters")
        plot_decision_boundary(np.asarray([0.06, 0.05]), -0.40, "Decision Boundary K=3 Optimized Parameters")
        plt.show()
    return assignment

#Function to plot learning curve 
def plot_learning_curve(distortions, k, title):
    color = plt.cm.Accent(2)
    plt.scatter(range(1, len(distortions) + 1), distortions, label=f'K={k}', color=color)
    plt.plot(range(1, len(distortions) + 1), distortions, linestyle='-', color=color)
    plt.xlabel('Iteration')
    plt.ylabel('Distortion (Objective Function Value)')
    plt.title(title)
    plt.legend()
    plt.show()

#Function to plot individual clusters
def plot_clusters(centroids, assignment, k, title):
    for i, (center, data_points) in enumerate(assignment.items()):
        color = plt.cm.Accent(i / k)
        last_elements = np.array(data_points)[:, -2:]
        plt.scatter(*zip(*last_elements), label=f'Cluster {i + 1}', color=color)

    centroids = np.array(centroids)[:, -2:]
    plt.scatter(*zip(*centroids), marker='X', s=100, c='black', label='Cluster Centers')
    plt.title(title)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()

# Function to plot data on to plot
def plotData():
    for species, group in class_2_and_3.groupby('species'):
        plt.scatter(group['petal_length'], group['petal_width'], c=species_to_color[species],
                    marker=species_to_marker[species], label=f'{species} Iris')

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('Scatter Plot of Versicolor and Virginica Iris Classes')
    plt.legend()

#Function to plot decision boundary
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

def main():
    #When k =2
    centroids_k2 = create_centroids(numerical_data, 2)
    iterate(numerical_data, 2, centroids_k2) 
    #When k = 3
    centroids_k3 = create_centroids(numerical_data, 3)
    iterate(numerical_data, 3, centroids_k3)
if __name__ == "__main__":
    main()