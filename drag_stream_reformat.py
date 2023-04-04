
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
    
    def add_item(self, sub_sequence: np.array, dist:float) -> None:
        self.activity = datetime.now()

        #print("********************Ajout à un cluster", cluster_id, np.array(Cluster.clusters).shape)

        already_exist = any(np.array_equal(sub_sequence, i) for i in self.items)
        
        if len(self) < self.size and not already_exist:
            self.items.append(sub_sequence)

        elif not already_exist:
            # print("******************** il vient remplacer un clustroid")
            # if there is a subsequence which is too near another subsequence, we can remove it . we can do it because l'inégalité triangulaire est vérifiée
            dist_matrice = np.array([
                [z_norm_dist(i, j) for i in self.items] for j in self.items
            ])
            min_dist = dist_matrice[dist_matrice != 0].min()
            ij_min = np.where(dist_matrice == min_dist)[0]
            ij_min = tuple([i.item() for i in ij_min])
            if dist > min_dist:
                self.items[ij_min[0]] = sub_sequence

class ClusterManager:
    def __init__(self, subsequence, radius, max_clusters):
        self.radius = radius
        self.nb_clustroid = 4
        self.outliers = []
        self.clusters = [[subsequence]]
        self.clusters_activity = [datetime.now()]
        self.max_clusters = max_clusters

    def add_cluster(self, subsequence):
        if len(self.clusters) > self.max_clusters:
            min_index = self.clusters_activity.index(
                min(self.clusters_activity))
            print(f"{'*'*10}This is the cluster that had the lowest activity",
                  min_index, self.clusters_activity)
            self.clusters_activity.pop(min_index)
            self.clusters.pop(min_index)

        self.clusters_activity.append(datetime.now())
        self.clusters.append([subsequence])


def clustering(Cluster, r, subsequence):
    dist = r
    min_dist = float('inf')
    cluster_id = False
    there_is_a_cluster = False
    # try to identify its cluster
    for id_cluster, cluster in enumerate(Cluster.clusters):
        for clustroid in cluster:
            if d := z_norm_dist(clustroid, subsequence) < dist:
                print("********************************, it entered a cluster")
                if d < min_dist:
                    min_dist = d
                    cluster_id = id_cluster
                # try to know if it can be the centroid
    """if min_dist >r and cluster_id!=False:
		print("Rien fait: Cette partie est délicate car on essaie d'optimiser le rayon du cluster")
		# on fait un clustering hierarchique pour garder un certain rayon dans notre algorithme de clustering
	"""
    # try to know if it can be the centroid
    if cluster_id != False:
        Cluster.clusters_activity[cluster_id] = datetime.now()

        print("********************Ajout à un cluster",
              cluster_id, np.array(Cluster.clusters).shape)

        if len(Cluster.clusters[cluster_id]) < Cluster.nb_clustroid and not any(np.array_equal(subsequence, i) for i in Cluster.clusters[cluster_id]):
            Cluster.clusters[cluster_id].append(subsequence)

        elif any(np.array_equal(subsequence, i) for i in Cluster.clusters[cluster_id]):
            # print("**** il avait deja son jumeau")
            return True
        else:
            # print("******************** il vient remplacer un clustroid")
            # if there is a subsequence which is too near another subsequence, we can remove it . we can do it because l'inégalité triangulaire est vérifiée
            dist_matrice = np.array([
                [z_norm_dist(i, j) for i in Cluster.clusters[cluster_id]] for j in Cluster.clusters[cluster_id]
            ])
            min_dist = dist_matrice[dist_matrice != 0].min()
            ij_min = np.where(dist_matrice == min_dist)[0]
            ij_min = tuple([i.item() for i in ij_min])
            if dist > min_dist:
                Cluster.clusters[cluster_id][ij_min[0]] = subsequence
        return True
    else:
        return False
