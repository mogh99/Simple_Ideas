import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

from Arguments import argument_parser

global data, upnormal_cluster, args

# Generate the normal clusters using make_blobs
def generate_data(X, y):
    y = y.reshape(len(y), 1)
    data = np.hstack((X, y))

    columns = [f"x{i}" for i in range(0, X.shape[1])]
    columns.append("y")

    df = pd.DataFrame(data, columns=columns)

    return df

# Generate gussian noise for each cluster
def generate_noise(centroids, radiuses):
    # Add noise for each cluster but all with the same label
    gaussian_noise = pd.DataFrame()
    columns = [f"x{i}" for i in range(0, args.features)]
    columns.append("y")

    # 1. Generate for each cluster a noise that start from the center
    for centroid in centroids:
        noise = np.random.normal(loc=centroid, scale=args.noise_std, size=(args.num_noise, args.features))
        
        y = np.array([upnormal_cluster if args.same_labels else 1 for i in range(0, noise.shape[0])])
        y = y.reshape(noise.shape[0], 1)
        noise = np.hstack((noise, y))

        noise = pd.DataFrame(noise, columns=columns)

        gaussian_noise = gaussian_noise.append(noise, ignore_index=True)

    # 2. Remove the noise from any cluster n-sphere and leave the noise outside the n-sphere
    # radius ^ 2 = (x1 - c1)^2 + (x2 - c2)^2 + ... + (xn - cn)^2
    # n-sphere is a sphere with n-dimensions
    for cluster, centroid in enumerate(centroids):
        value = 0
        radius = radiuses[cluster]**2

        for index, feature in enumerate(data.columns[:-1]):
            value += (gaussian_noise[feature] - centroid[index])**2
        
        gaussian_noise = gaussian_noise.loc[(value > radius)]
    
    return gaussian_noise

def find_radiuses(data, centroids, radius_threshold):
    radiuses = {}

    # For each cluster find the furthest distance from the centroid for all the features.
    for cluster, centroid in enumerate(centroids):
        furthests = []
        for index, feature in enumerate(data.columns[:-1]):
            feature_max, feature_min = data.loc[data.y == cluster, feature].max(), data.loc[data.y == cluster, feature].min()
            feature_max_radius, feature_min_radius = feature_max - centroid[index] + radius_threshold, centroid[index] - feature_min + radius_threshold

            furthests += [feature_max_radius, feature_min_radius]
        
        radiuses[cluster] = max(furthests)

    return radiuses

def onClick(event):
    if event.inaxes:
        new_point = [event.xdata, event.ydata, upnormal_cluster if args.same_labels else 1]
        data.loc[len(data)] = new_point
    
    plt.scatter(data.x0,data.x1,c=data.y)
    plt.show()


def make_meshgrid(x, y, h=.02):
    # h is the step size in the mesh
    x_min, x_max = x.min() - 5, x.max() + 5
    y_min, y_max = y.min() - 5, y.max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h)
                        ,np.arange(y_min, y_max, h))
    return xx, yy

if __name__ == "__main__":
    args = argument_parser()

    # Generate data as clusters and give them the same label.
    # TODO: If the data was read through a file we need to make sure the dataset has y feature and the centroid are known this issue can be solved using k-means algorithm 
    if args.path:
        data = pd.read_csv("JuhaAlgorithm\JuhaDataset.csv")
    else:
        X, y, centroids = make_blobs(n_features=args.features, n_samples=args.samples, center_box=args.ranges, centers=args.clusters, return_centers=True)
        data = generate_data(X, y)
    
    upnormal_cluster = args.clusters

    # Generate gaussian noise for each cluster
    if args.noise:
        # 1. Find the furthest point in each cluster and calculate the radius.
        radiuses = find_radiuses(data, centroids, args.radius_threshold)
        # 2. Generate the noise and remove the points that are inside the circle.
        gaussian_noise = generate_noise(centroids, radiuses)

        data = data.append(gaussian_noise)

    # Plot the normal data and allow the user to generate upnormal data only if the number of features equal 2.
    if args.features == 2:
        fig, ax = plt.subplots()
        plt.scatter(data.x0, data.x1, c=data.y)
        # If the noise was not generated automatically allow the user to generate noise
        if args.noise:
            for index, centroid in enumerate(centroids):
                ax.add_patch(plt.Circle(tuple(centroid), radiuses[index], fill=False))
        else:
            fig.canvas.mpl_connect("button_press_event", onClick)
        plt.show()

    # Save the data as csv file
    if args.csv:
        data.to_csv(args.csv+".csv", index=False, mode='w+')

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:,0:args.features], data.y, test_size=args.split, shuffle=True)
        
    # Test the data with a model
    clf = svm.SVC(gamma=args.gamma, C=args.c)
    clf.fit(X_train, y_train)

    predict = clf.predict(X_test)

    # Find the accuracy
    accuracy = accuracy_score(y_test, predict)

    print("Using SVM the model give accuracy = ", round(accuracy, 2)*100)

    # Display the trained model if only the number of features equals 2
    if args.display_model and args.features == 2:
        # Display the model 
        xx, yy = make_meshgrid(X_test.x0, X_test.x1)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        
        # Plot also the training points
        plt.scatter(X_test.x0, X_test.x1, c=y_test, cmap=plt.cm.coolwarm)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title("SVC_RBF")

        plt.show()