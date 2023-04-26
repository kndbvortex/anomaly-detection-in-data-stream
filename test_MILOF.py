# Import modules.
from multiprocessing import Manager
import multiprocessing as mp
import multiprocessing
from test_KitNet import class_KitNet
from test_ARIMAFD import class_ARIMAFD
from score_nab import evaluating_change_point
from test_iforestASD import class_iforestASD
from test_hs_tree import class_hstree
from test_LAMP import class_LAMP
from drag_stream import class_our
from test_Milof import class_MILOF
from datetime import datetime
import sys
import math
import numpy as np
# from pysad.models.integrations.reference_window_model import ReferenceWindowModel
# from pysad.utils import Data
# import scipy.io
# import plotly.graph_objects as go
import pandas as pd
# import matplotlib.pyplot as plt
# import time
import os

# from hyperopt import fmin, tpe,hp, STATUS_OK, Trials

# from numba import jit, cuda
import code
# code.interact(local=locals)
import time
import os
# import numba  # We added these two lines for a 500x speedup
# from numba import njit, types
# from numba.extending import overload, register_jitable
# from numba.core.errors import TypingError
actual_dataset = [0]

sys.path.append('MILOF/lib')
# from MiLOF import MILOF


# Test pipeline   les threshold des methodes coe iforest seront récupérés dans NAB parce qu'NAB à une fonction de score automatisé.
# *****************************************************************************************************************************
mutex = multiprocessing.Lock()

base_file = 'real_known_point_datasets.xlsx'
base = pd.read_excel(base_file)
print(base.head(2))

merlin_score = np.zeros(len(base))
time_taken = np.zeros(len(base))
best_params = ["params" for i in time_taken]
all_identified = ["no" for i in time_taken]


def dataset_test(merlin_score, best_params, time_taken, all_identified, key, idx, dataset, scoring_metric="merlin"):

    try:
        base2 = pd.read_csv("f1score_"+scoring_metric +
                            "_abnormal_point_results.xlsx")
        ligne = base2[key+"best_param"][idx]
        flag = False
        print('ok')
    except Exception as e:
        flag = True
        print(e)
        print("erreur de fichier ")
        ligne = "erreur"

    # try :
    if True:  # "ambient_temperature_system_failure" in dataset: #ligne =="params" or flag:
        
        if dataset.split('.')[-1] == 'txt':
            df = pd.read_csv("dataset/"+dataset, sep='\s+', header=None)
            df.columns = ['value']
        else:
            df = pd.read_csv("dataset/"+dataset)
        print('Lecture', df.columns, df['value'][0])
        df = pd.DataFrame(df["value"])
        print(df.columns, '......')

        if os.path.exists("real_nab_data/"+dataset):
            df = pd.read_csv("real_nab_data/"+dataset, names=['value'])
            print(df.columns)

        column = "value"

        # reading the dataset
        X = [[i] for i in df[column].values]
        right = np.array(str(base["Position discord"][idx]).split(';'))
        nbr_anomalies = len(str(base["Position discord"][idx]).split(';'))
        gap = int(len(X)/100)
        if scoring_metric == "merlin":
            # gap =int(int(base["Dataset length"][idx])/100)
            # discord length
            gap = int(len(X)/100)
        if scoring_metric == "nab":
            gap = int(len(X)/(20*nbr_anomalies))
            print(f'gap : {gap}'.center(50, '%'))

        if key == "HS-tree":
            # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            real_scores, scores_label, identified, score, best_param, time_taken_1 = class_hstree(
            ).test(dataset, X, right, nbr_anomalies, gap, scoring_metric=scoring_metric)

        if key == "MILOF":
            # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            real_scores, scores_label, identified, score, best_param, time_taken_1 = class_MILOF(
            ).test(dataset, X, right, nbr_anomalies, gap)
        if key == "iforestASD":
            # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            real_scores, scores_label, identified, score, best_param, time_taken_1 = class_iforestASD(
            ).test(X, right, nbr_anomalies, gap,)
        if key == "ARIMAFD":
            # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            real_scores, scores_label, identified, score, best_param, time_taken_1 = class_ARIMAFD(
            ).test(df[[column]], X, right, nbr_anomalies, gap, scoring_metric="merlin")
        if key == "KitNet":
            # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            real_scores, scores_label, identified, score, best_param, time_taken_1 = class_KitNet(
            ).test(X, right, nbr_anomalies, gap, scoring_metric="merlin")

        if key == "LAMP":
            base2 = pd.read_excel("point_methods_result_milof.xlsx")
            if base2[key+"best_param"][idx] == 'params':
                return idx, 0, 0, 0, 0
            # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            real_scores, scores_label, identified, score, best_param, time_taken_1 = class_LAMP(
            ).test(dataset, df[column].values, right, nbr_anomalies, gap)

        if key == "our":
            print(right, nbr_anomalies)
            # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            real_scores, scores_label, identified, score, best_param, time_taken_1 = class_our.test(
                dataset, df[column].values, right, nbr_anomalies, int(base["discord length"][idx]))

        df["anomaly_score"] = real_scores
        df["label"] = scores_label  # [0 if i<threshold else 1 for i in scores ]

        # print("anomalies at:" , identified, "while real are at:", str(base["Position discord"][idx]) )

        directory = os.path.dirname(('streaming_results/'+key+'/'+dataset))
        if not os.path.exists(directory):
            os.makedirs(directory)
        data_file_name = dataset.split('/')[-1]
        data_file_name = key+'_'+data_file_name
        dataset = directory+'/'+data_file_name
        df.to_csv(dataset, index=False)

        # thresholds.append(threshold)
        print("terminé")

        def insertion(file):
            best_params[idx] = best_param
            time_taken[idx] = time_taken_1
            merlin_score[idx] = score
            all_identified[idx] = identified
            try:
                
                base2 = pd.read_excel(file)
                print(all_identified[idx])
                base2.loc[idx, key+"_identified"] = str(all_identified[idx])
                base2.loc[idx, key+"_Overlap_merlin"] = score
                base2.loc[idx, key+"best_param"] = str(best_params [idx])
                base2.loc[idx, key+"time_taken"] = time_taken[idx]
            except Exception as e:
                print(f'Except,  {e} \n\n\n\n')
                base2 = {}
                base2[key+"_identified"] = all_identified
                base2[key+"_Overlap_merlin"] = merlin_score
                base2[key+"best_param"] = best_params 
                base2[key+"time_taken"]= time_taken
                
                if key in file:

                    print(best_params[idx], best_param)
                    for key2,value in best_params[idx].items():
                        base2["best_param"+key2] ="RAS"
            finally:
                print("**********************************************************")
                print("**********************************************************")
                print(
                    f'\tdataset: {dataset} \n\tScore: {score}\n\tBest params: {best_param}\n\ttime {time_taken_1}')
                print("**********************************************************")
                
            base2 = pd.DataFrame(base2)
            if key in file:
                """for key2,value in best_params[idx].items():
                    base2["best_param"+key2][idx] =best_params[idx][key2]"""
                base2.to_excel(file)
            else:
                base2.to_excel(file)

        # with mutex:
        # with open('real_known_point_datasets.xlsx') as csv_file:
        insertion("f1score_"+scoring_metric+"_abnormal_point_results.xlsx")
        insertion("result/f1score_"+scoring_metric+"_" +
                  key+"_abnormal_point_univariate.xlsx")
        # csv_file.flush()
        return idx, best_param, time_taken_1, score, identified


pool = mp.Pool(mp.cpu_count())


def test(meth):

    methods = {meth: 0}  # , "HS-tree":0,"MILOF":0,"HS-tree":0, "iforestASD":0}#"MILOF":0}# "MILOF":class_MILOF.test, "iforestASD_SUB":iforestASD_SUB,"subSequenceiforestASD":iforestASD } #"iforestASD":iforestASD, "HStree":HStree "MILOF":MILOF
    scoring_metric = ["merlin", 'nab']  # ,"merlin"
    print(methods)
    for key, method in methods.items():
        thresholds = []

        for scoring in scoring_metric:
            # dataset_test(merlin_score,best_params,time_taken,all_identified,key,1,base["Dataset"][1],scoring_metric=scoring)

            """for i, d in enumerate(base["Dataset"]):
                dataset_test(merlin_score,best_params,time_taken,all_identified,key,i,base["Dataset"][i],scoring_metric=scoring)
            """
            with Manager() as mgr:
                merlin_score = mgr.list([]) + list(np.zeros(len(base)))
                time_taken = mgr.list([]) + list(np.zeros(len(base)))
                best_params = mgr.list([]) + ["params" for i in time_taken]
                all_identified = mgr.list([]) + ["no" for i in time_taken]
                output = pool.starmap(dataset_test, [(merlin_score, best_params, time_taken, all_identified,
                                      key, idx, dataset, scoring) for idx, dataset in enumerate(base["Dataset"])])
    print("**** merlin score", merlin_score)


if __name__ == '__main__':
    import sys

    print("***@@@@@@@@@@@@@@", sys.argv)

    # test(0)
    # test("iforestASD")
    # test("HS-tree")
