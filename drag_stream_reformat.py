
import math
from datetime import datetime

import numpy as np

def z_norm_dist(x:np.array, y:np.array) -> float:
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
    return z_norm_dist(a, b)


class Cluster:
    def __init__(self, subsequence:np.array, max_size:int) -> None:
        self.activity = datetime.now()
        self.items = [subsequence]
        self.size = max_size
        
    def __len__(self):
        return len(self.items)
    
    def add_item(self, sub_sequence: np.array, r:float) -> None:
        self.activity = datetime.now()
        already_exist = any(np.array_equal(sub_sequence, i) for i in self.items)
        
        if len(self) < self.size and not already_exist:
            self.items.append(sub_sequence)

        elif not already_exist:
            # if there is a subsequence which is too near another subsequence, we can remove it . we can do it because l'inégalité triangulaire est vérifiée
            dist_matrice = np.array([
                [z_norm_dist(i, j) for i in self.items] for j in self.items
            ])
            min_dist = dist_matrice[dist_matrice != 0].min()
            ij_min = np.where(dist_matrice == min_dist)[0]
            ij_min = tuple([i.item() for i in ij_min])
            if r > min_dist:
                self.items[ij_min[0]] = sub_sequence

class ClusterManager:
    def __init__(self, subsequence:np.array|None, max_nb_clusters:int,r, p:int=4):
        self.nb_clustroid = 4
        self.cluster_number = max_nb_clusters
        self.r = r
        if subsequence:
            self.clusters = [Cluster(subsequence, p)]
        else:
            self.clusters = []
        self.clusters_activity = [datetime.now()]
        self.max_clusters = max_clusters
    
    @property
    def clusters_activity(self) -> list[datetime]:
        return [c.activity for c in self.items]

    def add_new_cluster(self, subsequence): 
        if len(self.clusters) > self.max_clusters:
            min_index = self.clusters_activity.index(
                min(self.clusters_activity))
            print(
                f"{'*'*10}.This is the cluster that had the lowest activity",min_index, self.clusters_activity
            )
            self.clusters.pop(min_index)
        self.clusters.append(Cluster(subsequence))


    def clustering(self, subsequence):
        min_dist = float('inf')
        cluster_id = -1
        # try to identify its cluster
        for i, cluster in enumerate(self.clusters):
            for clustroid in cluster.items():
                if d := z_norm_dist(clustroid, subsequence) < self.r:
                    print(f"{'*'*10}, it entered a cluster")
                    if d < min_dist:
                        min_dist = d
                        cluster_id = i
        # try to know if it can be the centroid
        if cluster_id != -1:
            self.clusters[cluster_id].add_item(subsequence)
        else:
            self.add_new_cluster(subsequence)


class DragStream:
    def __init__(self, r, nbr_cluster, training_period, p=4) -> None:
        self.r = r
        self.nbr_cluster = nbr_cluster
        self.training_period = training_period
        self.p = p
        self.cluster_manager = ClusterManager(None, nbr_cluster, r, p)
        self.discords = []
        self._first_learn = True

    def score_one(self, x:np.array) -> float:
        if self._first_learn:
            return 0
        
    

    def learn_one(self, x:np.array):
        if self._first_learn:
            self._first_learn = False
            self.cluster_manager.clustering(x)
            self.discords.append(x)
            return self

        isCandidate = True
		min_dist_if_discord = float('inf')
        if self.training_period > 0:
            self.training_period -= 1

        for c in self.discords:
            d = distance(x, c)
			min_dist_if_discord = min(min_dist_if_discord, d)
            if d < self.r:
                self.cluster_manager.clustering(c)
				self.discords.remove(c)

				# if not clustering(cluster, r, c):
				# 	cluster.add_cluster(T[s:s+w])
				if c <= training:  # *********because we can't update at every time
					# ******** voir comment ajouter un temps d'attente
					C_score[c] = 0
				isCandidate = False
				# Normalement ici aussi on devrait l'ajouter au cluster mais le clustering n'est pas encore très bon et lui donner trop de responsabilité peut être difficile

		if isCandidate and not clustering(cluster, r, T[s:s+w]):
			C.append(s)
			C_score[s] = min_dist_if_discord
		if not isCandidate and not clustering(cluster, r, T[s:s+w]):
			print("***************************it's not entering any cluster")
			cluster.add_cluster(T[s:s+w])

def stream_discord(T, w, r, training, max_clusters):
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
	cluster = Cluster(T[S[0]:S[0]+w], r, max_clusters)
	C_score = np.zeros(len(T))
	C_score[S[0]] = float('inf')
	# print(C)
	for s in [i for i in S if i not in C]:
		isCandidate = True
		min_dist_if_discord = float('inf')
		for c in C:
			# print(s,"*",s+w,"**",len(T))
			min_dist_if_discord = min(min_dist_if_discord, distance(T[s:s+w], T[c:c+w]))
			if c <= training:  # ********because we can't update at every time
				C_score[c] = min(C_score[c], distance(T[s:s+w], T[c:c+w]))
			if distance(T[s:s+w], T[c:c+w]) < r:
				C.remove(c)
				if not clustering(cluster, r, T[c:c+w]):
					cluster.add_cluster(T[s:s+w])
				if c <= training:  # *********because we can't update at every time
					# ******** voir comment ajouter un temps d'attente
					C_score[c] = 0
				isCandidate = False
				# Normalement ici aussi on devrait l'ajouter au cluster mais le clustering n'est pas encore très bon et lui donner trop de responsabilité peut être difficile

		if isCandidate and not clustering(cluster, r, T[s:s+w]):
			C.append(s)
			C_score[s] = min_dist_if_discord
		if not isCandidate and not clustering(cluster, r, T[s:s+w]):
			print("***************************it's not entering any cluster")
			cluster.add_cluster(T[s:s+w])

	# S=[i for i in S if i not in C]
	return C, S, C_score, cluster
