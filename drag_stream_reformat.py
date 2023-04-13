
import math
from datetime import datetime
from typing import Union

import numpy as np


def z_norm_dist(x: np.array, y: np.array) -> float:
    # x /= np.linalg.norm(x)
    # y /= np.linalg.norm(y)
    print(f'inside z-nom {math.sqrt(2*len(x)*(1-np.corrcoef(x, y)[0][1]))}')
    # ******songer à tester sans la distance normale ajoutée  pour voir l plus de la distance proposée
    return np.linalg.norm(x - y)*math.sqrt(2*len(x)*(1-np.corrcoef(x, y)[0][1]))


def distance(a: np.array, b: np.array) -> float:
    """
    Args:
            a (np.array): vector
            b (np.array): vector

    Returns:
            float : z_normalized distance between a and b
    """
    # ******songer à tester sans la distance normale ajoutée  pour voir l plus de la distance proposée
    a = a/np.norm(a)
    b = b/np.norm(b)
    return z_norm_dist(a, b)


class Cluster:
    def __init__(self, subsequences: np.array, max_size: int, distance=z_norm_dist) -> None:
        self.activity = datetime.now()
        if len(subsequences.shape) == 1:
            self.items = [subsequences]
        else:
            self.items = list(subsequences)
        self.size = max_size
        self.distance = distance

    def __len__(self):
        return len(self.items)

    def __str__(self) -> str:
        return str(self.items)

    def __repr__(self) -> str:
        return f"Cluster({self.items})"

    def add_item(self, sub_sequence: np.array, r: float) -> None:
        self.activity = datetime.now()
        already_exist = any(np.array_equal(sub_sequence, i)
                            for i in self.items)

        if len(self) < self.size and not already_exist:
            self.items.append(sub_sequence)

        elif not already_exist:
            # if there is a subsequence which is too near another subsequence, we can remove it . we can do it because l'inégalité triangulaire est vérifiée
            dist_matrice = np.array([
                [self.distance(i, j) for i in self.items] for j in self.items
            ])
            min_dist = dist_matrice[dist_matrice != 0].min()
            ij_min = np.where(dist_matrice == min_dist)[0]
            ij_min = tuple([i.item() for i in ij_min])
            if r > min_dist:
                self.items[ij_min[0]] = sub_sequence


class ClusterManager:
    def __init__(self, subsequence: Union[np.array, None], max_nb_clusters: int, r, p: int = 4, distance=z_norm_dist):
        self.p = 4
        self.cluster_number = max_nb_clusters
        self.r = r
        if subsequence:
            self.clusters = [Cluster(subsequence, p)]
        else:
            self.clusters = []
        self.max_clusters = max_nb_clusters
        self.distance = distance

    def __str__(self) -> str:
        return f'{self.clusters}'

    @property
    def clusters_activity(self) -> list[datetime]:
        return [c.activity for c in self.clusters]

    def add_new_cluster(self, subsequence):
        if len(self.clusters) > self.max_clusters:
            min_index = self.clusters_activity.index(
                min(self.clusters_activity))
            # print(
            #     f"{'*'*10}.This is the cluster that had the lowest activity", min_index, self.clusters_activity
            # )
            self.clusters.pop(min_index)
        self.clusters.append(Cluster(subsequence, self.p))

    def clustering(self, subsequence):
        min_dist = float('inf')
        cluster_id = -1
        # try to identify its cluster
        for i, cluster in enumerate(self.clusters):
            for clustroid in cluster.items:
                d = self.distance(clustroid, subsequence)
                min_dist = min(min_dist, d)
                if d < self.r:
                    # print(f"{'*'*10}, it entered a cluster")
                    if d < min_dist:
                        min_dist = d
                        cluster_id = i
        # try to know if it can be the centroid
        if cluster_id != -1:
            self.clusters[cluster_id].add_item(subsequence, self.r)
            return True, min_dist
        else:
            return False, min_dist


class DragStream:
    def __init__(self, r, nbr_cluster, training_period, p=4, distance=z_norm_dist) -> None:
        self.r = r
        self.nbr_cluster = nbr_cluster
        self.training_period = training_period + 1
        self.p = p
        self.cluster_manager = ClusterManager(None, nbr_cluster, r, p)
        self.discords = []
        self._first_learn = True
        self.distance = distance

    def _ajust_threshold(self):
        if len(self.discords) > 1:
            dist_ = []
            for i in range(len(self.discords)):
                for j in range(i+1, len(self.discords)):
                    dist_.append(self.distance(
                        self.discords[i], self.discords[j]))
            print(self.r)
            self.r = 48
            print(self.r)

    def score_one(self, x: np.array):
        # print(len(self.discords))
        if self._first_learn:
            self._first_learn = False
            self.cluster_manager.add_new_cluster(x)
            self.discords.append(x)
            return -1

        isCandidate = True
        min_dist_if_discord = float('inf')
        if self.training_period > 0:
            self.training_period -= 1
        l = []
        for c in self.discords:
            d = self.distance(x, c)
            min_dist_if_discord = min(min_dist_if_discord, d)
            if d < self.r:
                isCandidate = False
            else:
                l.append(c)

        self.discords = l

        isOutlier, dist_to_cluster = self.cluster_manager.clustering(x)
        isOutlier = not isOutlier

        if isCandidate and isOutlier:
            # print('discord')
            self.discords.append(x)
            self._ajust_threshold()
            if self.training_period > 0:
                return -1
            return min_dist_if_discord if min_dist_if_discord != float('inf') else 1
        elif isOutlier and not isCandidate:
            self.cluster_manager.add_new_cluster(x)

        if self.training_period > 0:
            return -1
        return 0

    def test(self, T, w):
        """_summary_

        Args:
                T (_type_): _description_
                w (_type_): _description_
        """
        S = [*range(0, len(T), int(w/2))]
        to_remove = []
        for idx, s in enumerate(S):
            if (len(T) < S[idx]+w):
                to_remove.append(s)
                # S.remove(s)  # to correct later
        for e in to_remove:
            S.remove(e)
        C_score = []
        for s in S:
            C_score.append(self.score_one(T[s:s+w]))

        return self.discords, S, C_score, self.cluster_manager.clusters


class NaiveIdea:
    def __init__(self, training, window_size) -> None:
        self.training = training + 1
        self.window_size = window_size
        self.window = []

    def train(self, X):
        scores = []
        for x in X:
            scores.append(self.train_one(x))
        return scores
    
    def train_one(self, x):
        last_std = np.std(self.window[self.window_size//2:])
        if len(self.window) > self.window_size:
            self.window.pop(0)
        self.window.append(x)

        if self.training > 0:
            self.training -= 1
        if self.training > 0:
            return 0
        return np.std(self.window[self.window_size//2:]) - last_std
            


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.express as px
    # a = np.random.randint(0,10, 2)
    # b = np.random.randint(0,10, 2)
    # m = a.shape[0]
    # c = np.sqrt(2*m*(1-(np.sum(a*b)-m*np.mean(a)*np.mean(b))/(m*np.std(a)*np.std(b))))
    # e = np.array((a-np.mean(a))/np.std(a))
    # f = (b-np.mean(b))/np.std(b)
    # d = np.linalg.norm((a-np.mean(a))/np.std(a) - (b-np.mean(b))/np.std(b), )
    # print(c)
    df = pd.read_csv(
        'dataset/nab-data/realKnownCause/ambient_temperature_system_failure.csv')

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(df['value'].index[:3721], df['value'][:3721], c='black')
    ax[0].plot(df['value'].index[3721:3721+364//4],
             df['value'][3721:3721+364//4], c='red')
    ax[0].plot(df['value'].index[3721+362//4: 6180],
             df['value'][3721+362//4: 6180], c='black')
    ax[0].plot(df['value'].index[6180:6180+362//4],
             df['value'][6180:6180+362//4], c='red')
    ax[0].plot(df['value'].index[6180+362//4:],
             df['value'][6180+362//4:], c='black')
    
    n = NaiveIdea(1000, 100)
    ax[1].plot(n.train(df['value']))
    
    plt.show()
    # plt.quiver((0,0), )
    # print(d)
    # print(z_norm_dist(a, b))
    # df = pd.read_csv(
    #     "dataset/nab-data/realKnownCause/ambient_temperature_system_failure.csv", usecols=['value'])
    # dragClass = DragStream(5, 22, 20)
    # _, S, scores, _ = dragClass.test(df['value'].values, 300)
    # dist_ = []
    # for i in range(len(dragClass.discords)):
    #     for j in range(i+1, len(dragClass.discords)):
    #         dist_.append(
    #             dragClass.distance(dragClass.discords[i], dragClass.discords[j]))
    # print(len(dragClass.discords))
    # print(dist_, len(S))
    # if dist_:
    #     print(np.mean(dist_))
    #     print(np.std(dist_))

    # # print(scores, S)
    # # print(df.shape)
    # plt.plot(scores)
    # plt.show()
