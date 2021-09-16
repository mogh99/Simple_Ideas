import argparse

def argument_parser():
    parser = argparse.ArgumentParser(prog="Juha Algorithm"
                                    ,description="Juah Algorithm assume linearly separable clusters that are labeled as normal data.\
                                                The algorithm allows the user to generate upnormal data \
                                                and train a supervised model with the generated dataset.")


    # Normal Dataset Variables:
    # 1. Number of data clusters
    parser.add_argument("--clusters", nargs="?", default=2, type=int,
                        help="Number of data clusters", metavar="clusters", dest="clusters")

    # 2.Dataset Path
    parser.add_argument("--dataset", nargs="?", required=False, type=str, 
                        help="The dataset path", metavar="path", dest="path")

    # 3.Number of samples
    parser.add_argument("--num-samples", nargs="?", default=100, type=int,
                        help="Number of data samples", metavar="samples", dest="samples")

    # 4.Center box ranges
    parser.add_argument("--center-box", nargs="+", default=(-20,20), metavar="ranges", dest="ranges",
                        type=int, help="The range that the cluster centers can be generated in.")

    # 5.Number of features
    parser.add_argument("--num-features", nargs="?", default=2, type=int,
                        help="Number of data features (columns, axis, dimensions)", metavar="features", dest="features")

    # Noise Dataset Variables: 
    # 1. Use gaussian noise
    parser.add_argument("--gaussain-noise", default=False, help="Add gaussain noise around each cluster",
                    dest="noise", action='store_true')

    # 2. Noise Circle Radius
    parser.add_argument("--radius-threshold", nargs="?", default=0.5, metavar="radius_threshold", dest="radius_threshold",
                        type=float, help="The value to increase or decrease the noise circle radius.")

    # 3. Noise Clusters Standard Deviation
    parser.add_argument("--noise-std", nargs="?", metavar="noise_std", dest="noise_std",
                        type=float, help="The standard deviation for the noise cluster")

    # 4. Noise Clusters Number of Samples
    parser.add_argument("--num-noise", nargs="?", metavar="num_noise", dest="num_noise",
                        type=int, help="The number of samples for each noise cluster")

    # Machine Learning Model Variables:
    # 1. Gamma value
    parser.add_argument("--gamma", nargs="?", default=0.001, metavar="gamma", dest="gamma",
                        type=float, help="The used gamma for the SVC with rbf kernel algorithm.")

    # 2. c value
    parser.add_argument("-c", nargs="?", default=100.0, metavar="c", dest="c",
                        type=float, help="The used c for the SVC with rbf kernel algorithm.")

    # 3. Train/Test split ratio
    parser.add_argument("--split", nargs="?", default=0.33, type=float,
                        help="Train/Test split ratio", metavar="split", dest="split")                  

    # 4. Use different labels
    parser.add_argument("--same-labels", default=True, help="Give the normal clusters the same label",
                    dest="same_labels", action='store_false')

    # 5. Save the generated data as csv file
    parser.add_argument("--save-data", nargs="?", required=False, type=str, 
                        help="Save the generated data as csv to the specified path", metavar="csv", dest="csv")

    # 6. Display the generated model
    parser.add_argument("--display-model", default=False, help="Display the generated model",
                        dest="display_model", action='store_true')

    args = parser.parse_args()

    if args.noise and not args.num_noise and not args.noise_std:
        parser.error("--gaussain-noise rquires --num-noise and --noise-std")

    return args