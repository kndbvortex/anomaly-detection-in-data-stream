# Import modules.
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from score_nab import evaluating_change_point
import matrixprofile as mp
import plotly.graph_objects as go
from river import drift
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, f1_score
# from pysad.evaluation import AUROCMetric
import numpy as np
from typing import Union, List, Tuple


import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import math
from datetime import datetime



def shape_base_distance(x, y):
    pass 

def distance(a, b):
    x = a
    y = b
    # ******songer à tester sans la distance normale ajoutée  pour voir l plus de la distance proposée
    return np.linalg.norm(a - b)*math.sqrt(2*len(x)*(1-np.corrcoef(x, y)[0][1]))


def z_norm_dist(x, y):
    # ******songer à tester sans la distance normale ajoutée  pour voir l plus de la distance proposée
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    # Meilleur gestion de la division par 0 
    t = pd.DataFrame({
		0: x,
		1: y
	}, dtype=np.float64)
    
    return math.sqrt(2*len(x)*(1-t.corr()[0][1]))

class Cluster:
    def __init__(self, subsequences: np.array, max_size: int, distance=z_norm_dist) -> None:
        self.activity = datetime.now()
        if len(subsequences.shape) == 1:
            self.items = [subsequences]
        else:
            self.items = list(subsequences)
        self.p = max_size
        self.distance = distance

    def __len__(self):
        return len(self.items)

    def __str__(self) -> str:
        return str(self.items)

    def __repr__(self) -> str:
        return f"Cluster({self.items})"

    def add_item(self, sub_sequence: np.array, r: float) -> None:
        self.activity = datetime.now()
        already_exist = any(np.array_equal(sub_sequence, i) for i in self.items)

        if len(self) < self.p and not already_exist:
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
                

class ClusterSet:
    def __init__(self, subsequence: Union[np.array, None], max_nb_clusters: int, r, p: int = 4, distance=z_norm_dist):
        self.p = p
        self.cluster_number = max_nb_clusters
        self.r = r
        if type(subsequence) == type(np.zeros(1)):
            self.clusters = [Cluster(subsequence, p)]
        else:
            self.clusters = []
        self.max_clusters = max_nb_clusters
        self.distance = distance

    def __str__(self) -> str:
        return f'{self.clusters}'

    @property
    def clusters_activity(self) -> List[datetime]:
        return [c.activity for c in self.clusters]

    def add_new_cluster(self, subsequence):
        print("New cluster !!!")
        if len(self.clusters) > self.max_clusters:
            min_index = self.clusters_activity.index(
                min(self.clusters_activity))
            self.clusters.pop(min_index)
        self.clusters.append(Cluster(subsequence, self.p))

    def clustering(self, subsequence) -> Tuple[bool, float]:
        min_dist = float('inf')
        cluster_id = -1
        dist = self.r
        # try to identify its cluster
        for i, cluster in enumerate(self.clusters):
            for clustroid in cluster.items:
                if d := self.distance(clustroid, subsequence) < dist:
                    dist = d
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
        
    @classmethod
    def stream_discord(cls, T, w, r, training, max_clusters):
        """

        Args:
                T (_type_): Dataset in this case the column
                w (_type_): windows size
                r (_type_): Threshold value
                training (_type_):
                max_clusters (_type_): Maximum size of a cluster

        Returns:
                _type_: _description_
        """

        S = [*range(0, len(T), int(w/2))]
        to_remove = []
        for idx, s in enumerate(S):
            if (len(T) < S[idx]+w):
                to_remove.append(s)
                # S.remove(s)  # to correct later
        for e in to_remove:
            S.remove(e)
        C = [S[0]]
        clusters = ClusterSet(T[S[0]:S[0]+w], max_clusters, r)
        C_score = np.zeros(len(T))
        C_score[S[0]] = 0
        # print(C)
        for s in [i for i in S if i not in C]:
            isCandidate = True
            min_dist_if_discord = float('inf')
            for c in C:
                # print(s,"*",s+w,"**",len(T))
                min_dist_if_discord = min(
                    min_dist_if_discord, distance(T[s:s+w], T[c:c+w]))
                # if c <= training:  # ********because we can't update at every time
                #     C_score[c] = min(C_score[c], distance(T[s:s+w], T[c:c+w]))
                if distance(T[s:s+w], T[c:c+w]) < r:
                    C.remove(c)
                    isBeingClustered, _ = clusters.clustering(T[c:c+w])
                    if not isBeingClustered:
                        clusters.add_new_cluster(T[s:s+w])
                    isCandidate = False
                    # Normalement ici aussi on devrait l'ajouter au cluster mais le clustering n'est pas encore très bon et lui donner trop de responsabilité peut être difficile
            
            isBeingClustered, min_dist_to_cluster = clusters.clustering(T[s:s+w])

            if isCandidate and not isBeingClustered:
                C.append(s)
                if s >= training:
                    if(min_dist_if_discord == float('inf')):
                        if min_dist_to_cluster == float('inf'):
                            C_score[s] = r - 1e-8
                        else:
                            C_score[s] = min_dist_to_cluster                
                    else:
                        C_score[s] = min_dist_if_discord
            if not isCandidate and not isBeingClustered:
                print("***************************it's not entering any cluster")
                clusters.add_new_cluster(T[s:s+w])

        # S=[i for i in S if i not in C]
        #print(f'Fucking Scores: {np.unique(C_score)}, len: {T.shape} training: {training}, len(s): {len(S)}')
        return C, S, C_score, clusters.clusters
    
    @classmethod
    def test(cls, dataset, X, right, nbr_anomalies, gap, scoring_metric="merlin"):

        def our(X: np.array, w:int, r, training:int, max_nb_cluster):
            # X should be a one dimensional vector
            _, _, scores, clust = cls.stream_discord(X, w, r, training, max_nb_cluster)
            print("*********nbr of clusters", len(clust))
            return scores

        def scoring(scores):
            identified = np.where(scores.squeeze() == 1)[0]
            anomalies_detected = np.array([])
            ground_truth_anomalies = np.array([])
            # gap = np.floor(len(scores)/100)
    
            for identify in identified:
                anomalies_detected = np.concatenate([anomalies_detected, np.arange(identify, identify+gap)])
            for identify in right:
                identify = int(identify)
                ground_truth_anomalies = np.concatenate(
                    [ground_truth_anomalies, np.arange(identify, identify+gap)])
            anomalies_detected = np.unique(anomalies_detected)
            ground_truth_anomalies = np.unique(ground_truth_anomalies)
            recall = len(np.intersect1d(anomalies_detected, ground_truth_anomalies))/len(ground_truth_anomalies)
            precision = len(np.intersect1d(anomalies_detected, ground_truth_anomalies))/len(anomalies_detected)
            try:
                score = 2*(recall*precision)/(recall+precision)
            except:
                score = 0.0
            print('gap', gap)
            # print('Scores compute', score)
            return score

        def score_to_label(nbr_anomalies, scores, gap):

            thresholds = np.unique(scores)
            f1_scores = []
            for threshold in thresholds:
                labels = np.where(scores < threshold, 0, 1)
                f1_scores.append(scoring(labels))
            q = list(zip(f1_scores, thresholds))
            thres = sorted(q, reverse=True, key=lambda x: x[0])[0][1]
            threshold = thres
            # arg = np.where(thresholds == thres)
            # print(f'thresholds : {thresholds} thres: {thres}'.center(100, '$'))

            # i will throw only real_indices here. [0 if i<threshold else 1 for i in scores ]
            return np.where(scores < threshold, 0, 1)

        def objective(args):
            scores = our(X, w=args["window"], r=args["threshold"], training=args["training"], max_nb_cluster=args["cluster"])
            scores = score_to_label(nbr_anomalies, scores, gap)
            return -1*scoring(scores)

        possible_window = np.array([gap, gap])  # arange(100,gap+200)
        possible_threshold = np.arange(1, 2*np.sqrt(gap), 0.1)
        right_discord = [int(discord) for discord in right]
        possible_training = np.arange(100, min(min(right_discord), int(len(X)/4)))
        possible_cluster = np.arange(10, 30)
        space2 = {
            "training": hp.choice("training_index", possible_training),
            "window": hp.choice("window_index", possible_window),
            "threshold": hp.choice("threshold_index", possible_threshold),
            "cluster": hp.choice("cluster_index", possible_cluster)
        }
        trials = Trials()

        start = time.monotonic()

        best = fmin(fn=objective, space=space2,
                    algo=tpe.suggest, max_evals=10, trials=trials)
        end = time.monotonic()
        best_param = {
            "cluster": possible_cluster[best["cluster_index"]],
            "training": possible_training[best["training_index"]],
            "window": possible_window[best["window_index"]],
            'threshold': possible_threshold[best["threshold_index"]]
        }
        scores = our(X, w=best_param["window"], r=best_param["threshold"],training=best_param["training"], max_nb_cluster=best_param["cluster"])

        scores_label = score_to_label(nbr_anomalies, scores, gap)
        identified =[key for key, val in enumerate(scores_label) if val in [1]] 
        return np.zeros(len(X)), scores_label, identified, scoring(scores_label), best_param, end-start