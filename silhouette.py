import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from collections import Counter
import argparse
from scipy.spatial.distance import euclidean
np.seterr(divide='ignore', invalid='ignore', all='warn')

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
def distance_each_cluster_mean(X, labels_, centroids):

    instances_length, clusters_length = len(X), Counter(labels_)
    distances_centroids = np.zeros((instances_length, len(set(labels_))))
    distances_each_point = np.zeros((instances_length, len(set(labels_))))

    for i in range(instances_length):
        for k in range(instances_length):

            base = labels_[i]
            target = labels_[k]
            same_cluster = False
            if (base == target):
                same_cluster = True

            # Centroid distance
            dist_centroid_ = euclidean(X[i], centroids[k])

            # Distance of each point
            distances_each_point_ = euclidean(X[i], X[k])

            if same_cluster == True:

                distances_centroids[i, target] += dist_centroid_ / (clusters_length[target] - 1)
                distances_each_point[i, target] += distances_each_point_ / (clusters_length[target] - 1)
            
            else:
                distances_centroids[i, target] += dist_centroid_ / clusters_length[target]
                distances_each_point[i, target] += distances_each_point_ / clusters_length[target]

    return distances_centroids, distances_each_point
                

# Function to calculate silhouette
def silhouette_metric(X, labels_, centroids):

    instances_number = len(X)
    distances_centroid, distances_each_point = distance_each_cluster_mean(X, labels_, centroids)

    cluster_intra_centroids, cluster_inter_centroids = np.empty(instances_number), np.empty(instances_number)

    cluster_intra, cluster_inter = np.empty(instances_number), np.empty(instances_number)

    for i in range(instances_number):

            label = labels_[i]

            # Calculate simplify silhouette 
            cluster_intra_centroids[i] = distances_centroid[i, label]
            distances_centroid[i, label] = np.inf
            cluster_inter_centroids[i] = min(distances_centroid[i, :])

            silhouette_centroid = np.mean(((cluster_inter_centroids - cluster_intra_centroids) / np.maximum(cluster_inter_centroids, cluster_intra_centroids)))

            # Calculate silhouette 
            cluster_intra[i] = distances_each_point[i, label]
            distances_each_point[i, label] = np.inf
            cluster_inter[i] = min(distances_each_point[i, :])

            silhouette = np.mean(((cluster_inter - cluster_intra) / np.maximum(cluster_inter, cluster_intra)))

            
    return silhouette_centroid, silhouette



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

        # Getting centroids
        centroids = [Model.cluster_centers_ [i] for i in Model.labels_]

        # Calling function to calculate centroid
        silhouette_centroid, silhouette = silhouette_metric(x.to_numpy(), Model.labels_, centroids)

        # Print values
        print("Simplify Silhouette value for {} group is: ".format(str(n_cluster)), silhouette_centroid)
        print("Silhouette value for {} group is: ".format(str(n_cluster)), silhouette)
        print('-------------------------------------------------------------------------------')

