import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs

import imageio
import os


class Point:
    def __init__(self, x, y, cluster=-1):
        self.x = x
        self.y = y
        self.cluster = cluster
        self.ps = None

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def dist(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def get_r_points(n, k):
    points = []
    X, y_true = make_blobs(n_samples=n, centers=k,
                           cluster_std=0.6, random_state=0)

    for xy in X:
        points.append(Point(xy[0], xy[1]))

    return points, y_true


def centroids(points, k):
    x_center = np.mean(list(map(lambda p: p.x, points)))
    y_center = np.mean(list(map(lambda p: p.y, points)))
    center = Point(x_center, y_center)
    R = max(map(lambda r: dist(r, center), points))
    centers = []
    for i in range(k):
        x_c = x_center + R * np.cos(2 * np.pi * i / k)
        y_c = y_center + R * np.sin(2 * np.pi * i / k)
        centers.append(Point(x_c, y_c))
    return centers


def new_center(points):
    if points:
        x_center = np.mean(list(map(lambda p: p.x, points)))
        y_center = np.mean(list(map(lambda p: p.y, points)))
        center = Point(x_center, y_center)
    else:
        center = Point(0, 0)

    return center


def nearest_centroids(points, centers):
    for point in points:
        min_dist = dist(point, centers[0])
        point.cluster = 0
        for i in range(len(centers)):
            temp = dist(point, centers[i])
            if temp < min_dist:
                min_dist = temp
                point.cluster = i


def dist_to_clusters(points, centers, m=2):

    for point in points:
        dists_to_clusters = []
        ps = []
        for center in centers:
            dists_to_clusters.append(dist(point, center))

        sum_of_distances = sum(dists_to_clusters)

        for distance in dists_to_clusters:
            ps.append((distance / sum_of_distances) ** (2 / (1 - m)))

        point.ps = ps

        point.cluster = np.argmax(ps)


def create_matrix(points, k):
    matrix = np.zeros(shape=(len(points), k))

    for i, point in enumerate(points):
        for j, p in enumerate(point.ps):
            matrix[i][j] = p

    return matrix


def get_max_elem(matrix1, matrix2):

    new_matrix = np.zeros(shape=(len(matrix1), len(matrix1[0])))

    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            new_matrix[i][j] = abs(matrix1[i][j] - matrix2[i][j])

    return np.max(new_matrix)


if __name__ == "__main__":
    k = 3
    m = 2
    e = 0.001

    random_points = get_r_points(500, 3)[0]

    centers = centroids(random_points, k)

    dist_to_clusters(random_points, centers)

    p_matrix = create_matrix(random_points, k)

    past_matrix = np.zeros(shape=(len(random_points), k))

    clusters_points = [[] for _ in range(k)]

    iter = 0

    while True:

        print(iter)

        if get_max_elem(past_matrix, p_matrix) <= e:
            break

        clusters_points = [[] for _ in range(k)]

        for point in random_points:
            clusters_points[point.cluster].append(point)

        past_matrix = p_matrix
        centers = list(map(new_center, clusters_points))

        dist_to_clusters(random_points, centers, k)
        p_matrix = create_matrix(random_points, k)

        iter += 1

    for cluster in clusters_points:
        for point in cluster:
            print(point.ps)
        print("------------------")

    colors = []
    for cluster in clusters_points:
        p = plt.scatter(list(map(lambda l: l.x, cluster)), list(map(lambda l: l.y, cluster)), linewidths=3)
        colors.append(p.get_facecolor())

    for i, center in enumerate(centers):
        plt.scatter(center.x, center.y, linewidths=6, marker='v', color='black')

    plt.savefig("res.png")
    plt.show()
