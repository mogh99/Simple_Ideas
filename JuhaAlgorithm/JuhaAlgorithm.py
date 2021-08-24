import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from matplotlib.widgets import Cursor
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

global data

def argument_parser():
    parser = argparse.ArgumentParser(prog="Juha Algorithm"
                                    ,description="Juah Algorithm assume linearly separable clusters that are labeled as normal data.\
                                                The algorithm allows the user to generate upnormal data \
                                                and train a supervised model with the generated dataset.")

    # Number of data clusters
    parser.add_argument("--clusters", nargs="?", default=2, type=int,
                        help="Number of data clusters", metavar="clusters", dest="clusters")

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
        new_point = [event.xdata, event.ydata, 0]
        data.loc[len(data)] = new_point
    
    plt.scatter(data.x1,data.x2,c=data.y)
    plt.show()

if __name__ == "__main__":
    args = argument_parser()

    # Generate data as clusters and give them the same label.
    X, _ = make_blobs(n_samples=args.samples, center_box=args.ranges, centers=args.clusters)
    data = pd.DataFrame({"x1":X[:, 0], "x2":X[:, 1], "y":1})

    # Plot the normal data and allow the user to generate upnormal data.
    fig = plt.figure()
    plt.scatter(data.x1, data.x2, c=data.y)
    fig.canvas.mpl_connect("button_press_event", onClick)
    plt.show()

    # Save the data as csv file
    if args.csv:
        data.to_csv("JuhaDataset.csv")

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:,0:2], data.y, test_size=args.split, shuffle=True)

    # Test the data with a model
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(X_train, y_train)

    predict = clf.predict(X_test)

    # Find the accuracy
    accuracy = accuracy_score(y_test, predict)

    print("Using SVM the model give accuracy = ", round(accuracy, 2)*100)


    