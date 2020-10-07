import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from collections import Counter
import argparse
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-inputFile',
        default='iris.csv',
        type=str,
        help='name of file that will be worked')
    parser.add_argument(
        '-sep',
        default=',',
        help='column identifier')
    parser.add_argument(
        '-dec',
        default='.',
        help='decimal identifier')

    parser.add_argument(
        '-target',
        default='variety',
        type=str,
        help='target')

    return parser.parse_args()


# Function to apply kmeans
def apply_model(x, k):
    Model = KMeans(k)
    return Model.fit(x)


# Calculate distances for each cluster
def distance_each_cluster_mean(X, labels_):

    instances_length, clusters_length = len(X), Counter(labels_)
    distances = np.zeros((instances_length, len(set(labels_))))

    for i in range(instances_length):

        for j in range(instances_length):
            base = labels_[i]
            target = labels_[j]
            
            same_cluster = False
            if base == target:
                same_cluster = True

            dist_ = euclidean(X[i], X[j])

            if same_cluster == True:

                distances[i, target] += dist_ / (clusters_length[target] - 1)
            
            else:
                distances[i, target] += dist_ / clusters_length[target]

    return distances
                

# Function to calculate silhouette
def silhouette_metric(X, labels_):

    instances_number, distances = len(X), distance_each_cluster_mean(X, labels_)

    cluster_intra, cluster_inter = np.empty(instances_number), np.empty(instances_number)

    for i in range(instances_number):

        label = labels_[i]
        cluster_intra[i] = distances [i, label]
        distances[i, label] = np.inf
        cluster_inter[i] = min(distances[i, :])

    return np.mean(((cluster_inter - cluster_intra) / np.maximum(cluster_inter, cluster_intra)))



if __name__ == "__main__":

    args = get_args()

    # Reading dataset
    df = pd.read_csv('dataset/' + args.inputFile, sep=args.sep, decimal=args.dec)
    
    # Selecting just numeric values
    x = df.select_dtypes(exclude=['object'])

    # Running k-means for each cluster number
    for n_cluster in [2,3,4]:
        # Apply model
        Model = apply_model(x, n_cluster)

        # Print Silhouette
        print("Silhouette value for {} group is: ".format(str(n_cluster)), silhouette_metric(x.to_numpy(), Model.labels_))
        print("Silhouette (sklearn) value  for {} group is: ".format(str(n_cluster)), silhouette_score(x, Model.labels_))
