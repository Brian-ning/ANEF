'''
Created on Feb 12, 2015
Contact: pachlioptas@gmail.com
Copyright notice: 
Copyright (c) 2015, Panagiotis Achlioptas
You are free to use, change, or redistribute this code in any way you want for non-commercial purposes only.
'''
from sklearn import metrics
import numpy as np
from numpy import dtype
from scipy.misc import comb
from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd
from sklearn import decomposition
from sklearn import cluster as sklearn_cluster
from sklearn.preprocessing import Imputer


def cluster_distance_matrix(distanceMatrix, clusteringType, **kwargs):
    '''
    Performs the requested clustering algorithm given a distance matrix.  
    '''

    if clusteringType == 'hierarchical':
        linkageFunction = 'centroid'
        trueClusterNum = 2
        print("\nPerforming **Hierarchical Clustering with linkage = %s.**" % (linkageFunction,))
        distArray = ssd.squareform(distanceMatrix)
        Z = hierarchy.linkage(distArray, method=linkageFunction)
        labels = hierarchy.fcluster(Z, trueClusterNum, criterion="maxclust")
        if len(np.unique(labels)) != trueClusterNum:
            print("!! Clusters found: " + str(len(np.unique(labels))))
        return labels

    if clusteringType == 'affinity':
        print("\nPerforming **Affinity Propagation.**")
        affinities = np.exp(- (distanceMatrix ** 2) / (2 * (np.median(distanceMatrix) ** 2)))
        cluster_centers_indices, labels = sklearn_cluster.affinity_propagation(affinities, copy=False, verbose=True)
        print("%d clusters found." % (len(np.unique(labels)),))
        return labels

    if clusteringType == "dbscan":
        print("\nPerforming **DBScan Clustering.**")
        eps = np.percentile(distanceMatrix, 10)
        if eps < 0:
            eps = -eps
        labels = sklearn_cluster.DBSCAN(eps, min_samples=10, metric='precomputed').fit_predict(distanceMatrix)
        print("% Predicted as Noise: " + str(np.sum(labels == -1) / float(len(labels))))
        return labels

    if clusteringType == "spectral":
        trueClusterNum = 3
        print("\nPerforming **Spectral (with Normalized Laplacian) Clustering.**")
        affinities = np.exp(- (distanceMatrix ** 2) / (2 * (np.median(distanceMatrix) ** 2)))
        print(affinities.size)
        # arpack was chosen for stability reasons.
        classifier = sklearn_cluster.SpectralClustering(n_clusters=trueClusterNum, affinity='precomputed', assign_labels='kmeans', eigen_solver='arpack')
        # trainDataVecs = Imputer().fit_transform(affinities)
        affinities_clearnan = np.isnan(affinities)
        classifier.fit(affinities_clearnan)
        return classifier.labels_


def evaluate_unsup_clustering(labels_true, labels, n_clusters=None, verbose=True):
    homo = metrics.homogeneity_score(labels_true, labels)
    comp = metrics.completeness_score(labels_true, labels)
    vmea = metrics.v_measure_score(labels_true, labels)
    aran = metrics.adjusted_rand_score(labels_true, labels)
    amut = metrics.adjusted_mutual_info_score(labels_true, labels)

    if verbose:
        if n_clusters != None:
            print('Estimated number of clusters: %d' % n_clusters)
        print("Homogeneity: %0.3f" % homo)
        print("Completeness: %0.3f" % comp)
        print("V-measure: %0.3f" % vmea)
        print("Adjusted Rand Index: %0.3f" % aran)
        print("Adjusted Mutual Information: %0.3f" % amut)

    return homo, comp, vmea, aran, amut


def evaluate_distance_matrix(distanceMatrix, trueClusters, clusteringType, **kwargs):
    # TODO: 1. clear blackList dependency
    #       2. clustering type is an unlucky name for betaCV and the like.

    trueClusterNum = len(np.unique(trueClusters))
    #     distanceMatrixCopy = np.copy(distanceMatrix)

    if clusteringType == 'all' or 'betaCV' in clusteringType:
        res = beta_cv(distanceMatrix, trueClusters, blackList=None, ranks=False)
        print
        "Beta-CV = %f" % (res,)

    if clusteringType == 'all' or 'cIndex' in clusteringType:
        res = c_index(distanceMatrix, trueClusters, blackList=None)
        print
        "C-Index = %f" % (res,)

    if clusteringType == 'all' or 'silhouette' in clusteringType:
        print
        "Silhouette = %f" % (metrics.silhouette_score(distanceMatrix, trueClusters, metric='precomputed'),)

    if clusteringType == 'all' or 'hierarchical' in clusteringType:
        print
        "\nEvaluating **Hierarchical Clustering**"
        distArray = ssd.squareform(distanceMatrix)
        try:
            linkageFunction = kwargs['linkage']
        except:
            linkageFunction = "complete"

        print
        "Linkage = " + linkageFunction
        Z = hierarchy.linkage(distArray, method=linkageFunction)
        T = hierarchy.fcluster(Z, trueClusterNum, criterion="maxclust")
        if len(np.unique(T)) != trueClusterNum:
            print
            "!Clusters found: " + str(len(np.unique(T)))

        res = evaluate_unsup_clustering(trueClusters, T, None, verbose=True)

    if clusteringType == 'all' or 'affinity' in clusteringType:
        print
        "\nEvaluating **Affinity Propagation**"
        affinities = np.exp(- (distanceMatrix ** 2) / (2 * (np.median(distanceMatrix) ** 2)))
        cluster_centers_indices, labels = sklearn_cluster.affinity_propagation(affinities, copy=False, verbose=True)
        res = evaluate_unsup_clustering(trueClusters, labels, len(cluster_centers_indices), verbose=True)

    if clusteringType == 'all' or "dbscan" in clusteringType:
        print
        "\nEvaluating **DBScan Clustering**"
        # TODO maybe adapt eps
        eps = np.percentile(distanceMatrix, 5)
        predictedLabels = sklearn_cluster.DBSCAN(eps, metric='precomputed').fit_predict(distanceMatrix)
        print("Predicted as Noise: " + str(np.sum(predictedLabels == -1)))
        res = evaluate_unsup_clustering(trueClusters, predictedLabels, len(np.unique(predictedLabels)), verbose=True)

    if clusteringType == 'all' or "spectral" in clusteringType:
        print("\nEvaluating **Spectral (with Normalized Laplacian) Clustering**")
        affinities = np.exp(- (distanceMatrix ** 2) / (2 * (np.median(distanceMatrix) ** 2)))
        # arpack was chosen for stability reasons.
        classifier = sklearn_cluster.SpectralClustering(n_clusters=trueClusterNum, affinity='precomputed',
                                                        assign_labels='kmeans', eigen_solver='arpack')
        classifier.fit(affinities)
        res = evaluate_unsup_clustering(trueClusters, classifier.labels_, None, verbose=True)

    # assert(np.all(distanceMatrixCopy == distanceMatrix))
    return res


# def evaluate_embedding():
#     if clusteringType == 'nmf' or clusteringType =='all':
#         print "\nEvaluating **NMF Clustering**"
#         trueClusterNum = len(np.unique(trueClusters))
#         nmf            = decomposition.NMF(n_components=trueClusterNum, nls_max_iter=3000)
#         nmfModel       = nmf.fit_transform(distMatrix)
#         predictedLabels = np.argmax(nmfModel, axis=1)
#         evaluate_unsup_clustering(trueClusters, predictedLabels, len(np.unique(predictedLabels)), verbose=True)


def boolean_intersection(vector1, vector2):
    '''
    Calculates the logical AND operator between two boolean vectors.  

    :param : np.array(boolean) : vector1       : n x 1 boolean vector.    
    :param : np.array(boolean) : vector2       : n x 1 boolean vector.
    :return: np.array(boolean) : result        : n x 1 boolean vector where result[i] is True iff vector1[i] == vector2[i] == 1.
    '''

    assert (vector1.shape == vector2.shape)
    return np.array([vector1[i] and vector2[i] for i in range(len(vector1))], dtype=np.bool)


def number_of_intra_pairs(clusters):
    '''
    :param : np.array(boolean) : clusters    : n x 1 cluster indicator vector, clusters[i]=j iff i-th point in member of the j-th cluster.
    :return: double            : total:      : number of all distinct pairs that can be formed between points belonging to the same cluster.
    '''

    total = 0
    for i in range(min(clusters), max(clusters) + 1):
        n = np.sum(clusters == i)
        total += comb(n, 2)
    return total


def number_of_inter_pairs(clusters):
    '''
    :param : np.array(boolean) : clusters    : n x 1 cluster indicator vector, clusters[i]=j iff i-th point in member of the j-th cluster.
    :return: double            : total:      : number of all distinct pairs that can be formed between points belonging to different clusters.
    '''
    total = 0
    cluster_sizes = []
    for i in range(min(clusters), max(clusters) + 1):
        cluster_sizes.append(np.sum(clusters == i))

    for i in range(len(cluster_sizes) - 1):
        for j in range(i + 1, len(cluster_sizes)):
            total += cluster_sizes[i] * cluster_sizes[j]

    return total


def inner_cluster_distances(distance_matrix, cluster1, cluster2, symmetric=True):
    '''
    Calculates the sum of pairwise distances between all members of 2 clusters.

    :param: np.array          : distanceMatrix : square matrix of pairwise distances between n elements.
    :param: np.array(boolean) : cluster1       : n x 1 vector where cluster[i]=True iff element i belongs is in cluster1. 
    :param: np.array(boolean) : cluster2       : n x 1 vector where cluster[i]=True iff element i belongs is in cluster2.    
    :param: boolean           : symmetric      : True iff distance_matrix(i,j) = distance_matrix(j,i) for every i and j.
    '''

    assert (len(cluster1) == len(cluster2) == distance_matrix.shape[0])
    assert (np.any(boolean_intersection(cluster1, cluster2)) == False)

    sum = np.sum

    if symmetric:
        return sum(np.dot(cluster1, np.dot(distance_matrix, cluster2))) / 2.  # float( sum(cluster1) * sum(cluster2))
    else:
        return sum(np.dot(cluster1, np.dot(distance_matrix, cluster2)))


def mask_intra_pairs(clusters, blackList=None):
    '''
    Creates a binary n x n mask (np.array) whose entry (i,j) is 1 iff the elements i and j are on the same cluster.  I.e., clusters[i] = clusters[j].

    :param: np.array  : clusters, n x 1 vector denoting at position -i- what is the class of element i.
    :param: list(int) : blackList, a list carrying integers corresponding to classes IDs. Any element -i- for which clusters[i] is in blackList will be
    ignored.

    :returns: The square array masking the pairs of indices that correspond to elements that belong on same clusters.
    '''

    sum = np.sum
    n = clusters.shape[0]
    cluster_sizes = []  # Used only for asserting purposes

    intraMask = np.zeros((n, n), dtype=np.bool)
    for i in range(min(clusters), max(clusters) + 1):
        if blackList != None and i in blackList: continue
        cluster_i = np.array(clusters == i).reshape(n, 1)
        intraMask += cluster_i.dot(cluster_i.T)
        cluster_sizes.append(sum(cluster_i))

        # assert that we are looking over the correct number of elements
    # take into account that we keep both (i,j) and (j,i) and the diagonal elements in our mask (i,i)
    assert ((sum(intraMask) - sum(cluster_sizes)) / 2 == sum(
        map(lambda n: 0 if n == 1 else n * (n - 1) / float(2), cluster_sizes)))
    return intraMask


def c_index(distance_matrix, clusters, blackList=None):
    # Currently implemented only for symmetric distance metrics.
    sum = np.sum

    intra_mask = mask_intra_pairs(clusters, blackList)
    intra_mask[np.tril_indices_from(
        intra_mask)] = False  # Discard the diagonal and the lower-triangular corresponding to trivial or symmetric pairs.

    intra_pairs = sum(intra_mask)
    intra_dists = sum(distance_matrix[intra_mask])

    all_distances = distance_matrix[np.triu_indices_from(distance_matrix,
                                                         k=1)]  # Again, we only consider the upper triangular without the diagonal elements.
    all_distances.sort()

    globally_minimum_dists = sum(all_distances[:intra_pairs])
    globally_maximum_dists = sum(all_distances[-intra_pairs:])

    res = (intra_dists - globally_minimum_dists) / float(globally_maximum_dists - globally_minimum_dists)
    assert (res >= 0 and res <= 1)
    return res


def beta_cv(distance_matrix, clusters, blackList=None, ranks=False):
    inner_dists, intra_dists = inner_intra_distances(distance_matrix, clusters, blackList=None, ranks=False)
    return np.average(intra_dists) / np.average(inner_dists)


def inner_intra_distances(distMatrix, clusters, blackList=None, ranks=False, verbose=False):
    '''
    :param: np.array :disMatrix,  n x n matrix capturing the distance between element i and j in disMatrix(i,j)
    :param: np.array :clusters,   n x 1 matrix denoting at position -i- what is the class of element i.
    :param: list(int) :blackList, a list carrying integers corresponding to classes, that should be ignored in the calculations.       
    '''
    assert (distMatrix.shape[0] == distMatrix.shape[1] == len(clusters))

    if ranks == False:  # Then the distMatrix must be symmetric
        assert (np.allclose(distMatrix, distMatrix.T))

    sum = np.sum
    n = distMatrix.shape[0]
    intraMask = mask_intra_pairs(clusters, blackList)

    # Make a mask that is one iff (i,j) belong at different clusters.
    innerMask = np.ones(distMatrix.shape, dtype=np.bool)
    innerMask = innerMask - intraMask

    if blackList != None:
        for blackClass in blackList:
            for elem in np.where(clusters == blackClass)[0]:
                innerMask[elem, :] = False
                innerMask[:, elem] = False

    if ranks:  # if ranks were used the matrix is not neccesarily symmetric
        intraMask[np.diag_indices(n)] = False  # The distance between an element and itself is trivially minimal
        innerMask[np.diag_indices(n)] = False
    else:
        intraMask[np.tril_indices_from(intraMask)] = False  # we can throw the lower triangular and the diagonal.
        innerMask[np.tril_indices_from(innerMask)] = False

    intraDistances = distMatrix[intraMask]
    assert (len(intraDistances) - number_of_intra_pairs(clusters) == 0)

    innerDistances = distMatrix[innerMask]
    assert (len(innerDistances) - number_of_inter_pairs(clusters) == 0)

    return innerDistances, intraDistances