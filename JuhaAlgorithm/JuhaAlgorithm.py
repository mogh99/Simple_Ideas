import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from matplotlib.widgets import Cursor
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

global data, upnormal_cluster

def argument_parser():
    parser = argparse.ArgumentParser(prog="Juha Algorithm"
                                    ,description="Juah Algorithm assume linearly separable clusters that are labeled as normal data.\
                                                The algorithm allows the user to generate upnormal data \
                                                and train a supervised model with the generated dataset.")

    # Number of data clusters
    parser.add_argument("--clusters", nargs="?", default=2, type=int,
                        help="Number of data clusters", metavar="clusters", dest="clusters")

    #Dataset Path
    parser.add_argument("--dataset", nargs="?", required=False, type=str, 
                        help="The dataset path", metavar="path", dest="path")

    # Train/Test split ratio
    parser.add_argument("--split", nargs="?", default=0.33, type=float,
                        help="Train/Test split ratio", metavar="split", dest="split")                  

    # Number of samples
    parser.add_argument("--num-samples", nargs="?", default=1000, type=int,
                        help="Number of data samples", metavar="samples", dest="samples")

    # Center box ranges
    parser.add_argument("--center-box", nargs="+", default=(50,-50), metavar="ranges", dest="ranges",
                        type=int, help="The range that the cluster centers can be generated in.")

    #Save the generated data as csv file
    parser.add_argument("--save-data", default=False, help="Save the generated data as csv file",
                        dest="csv", action='store_true')

    args = parser.parse_args()

    return args

def onClick(event):
    if event.inaxes:
        new_point = [event.xdata, event.ydata, upnormal_cluster]
        data.loc[len(data)] = new_point
    
    plt.scatter(data.x1,data.x2,c=data.y)
    plt.show()


def make_meshgrid(x, y, h=.02):
    # h is the step size in the mesh
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h)
                        ,np.arange(y_min, y_max, h))
    return xx, yy

if __name__ == "__main__":
    args = argument_parser()

    # Generate data as clusters and give them the same label.
    if args.path:
        data = pd.read_csv("JuhaAlgorithm\JuhaDataset.csv")
    else:
        X, y = make_blobs(n_samples=args.samples, center_box=args.ranges, centers=args.clusters)
        data = pd.DataFrame({"x1":X[:, 0], "x2":X[:, 1], "y":y})
    

    # Plot the normal data and allow the user to generate upnormal data.
    upnormal_cluster = args.clusters
    fig = plt.figure()
    plt.scatter(data.x1, data.x2, c=data.y)
    fig.canvas.mpl_connect("button_press_event", onClick)
    plt.show()

    # Save the data as csv file
    if args.csv:
        data.to_csv("JuhaDataset.csv", index=False)

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:,0:2], data.y, test_size=args.split, shuffle=True)

    # Test the data with a model
    clf = svm.SVC(gamma=10, C=100.)
    clf.fit(X_train, y_train)

    predict = clf.predict(X_test)

    # Find the accuracy
    accuracy = accuracy_score(y_test, predict)

    print("Using SVM the model give accuracy = ", round(accuracy, 2)*100)

    # Display the model 
    xx, yy = make_meshgrid(X_test.x1, X_test.x2)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    # Plot also the training points
    plt.scatter(X_test.x1, X_test.x2, c=y_test, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("SVC_RBF")

    plt.show()