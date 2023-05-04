# Import modules.
import numpy as np
#from pysad.models.integrations.reference_window_model import ReferenceWindowModel
from pysad.utils import Data
import scipy.io
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import time
import os 

from hyperopt import fmin, tpe,hp, STATUS_OK, Trials

from numba import jit, cuda
import code
#code.interact(local=locals)
import time
import os
import numba  # We added these two lines for a 500x speedup
from numba import njit, types
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError
actual_dataset=[0]

import math
import sys
from datetime import datetime
sys.path.append('MILOF/lib')
from MiLOF import MILOF

from drag_stream import class_our
from test_LAMP import class_LAMP
from test_hs_tree import class_hstree
from test_iforestASD import class_iforestASD
from score_nab import evaluating_change_point
from test_ARIMAFD import class_ARIMAFD
from test_KitNet import class_KitNet
from drag_stream_2 import DragStream


# Test pipeline   les threshold des methodes coe iforest seront récupérés dans NAB parce qu'NAB à une fonction de score automatisé. 
#*****************************************************************************************************************************
import multiprocessing
mutex =multiprocessing.Lock()

base_file ='discord.xlsx'
base = pd.read_excel(base_file)


def dataset_test(merlin_score,best_params,time_taken,all_identified,key,idx,dataset,scoring_metric="merlin"):

    # try: 
    #     base2 = pd.read_excel(scoring_metric+"_discord_results.xlsx") 
    #     ligne = base2[key+"best_param"][idx]
    #     flag=False 
    # except :
    #     flag=True
    #     print("erreur de fichier ...", scoring_metric+"_discord_results.xlsx")
    #     ligne="erreur"
        
    #try :
    if True : #True :#ligne =="params" or flag:

        df = pd.read_csv("dataset/"+dataset, names=["value"])
        print(dataset)
        if os.path.exists("real_nab_data/"+dataset) :
            df = pd.read_csv("real_nab_data/"+dataset)
        column="value"

        X =[[i] for i in df[column].values]
        print(df.head())
        right=np.array(str(base["Position discord"][idx]).split(';'))
        nbr_anomalies=len(str(base["Position discord"][idx]).split(';'))

        if scoring_metric=="merlin":
            gap =int(int(base["discord length"][idx]))
        if scoring_metric=="nab":
            gap = int(len(X)/(20*nbr_anomalies))
       
        if key=="LAMP":
            """base2 = pd.read_excel("point_methods_result_milof.xlsx")
            if base2[key+"best_param"][idx]=='params':
                return idx, 0,0, 0, 0"""
            real_scores, scores_label, identified,score,best_param, time_taken_1= class_LAMP.test(dataset,df[column].values,right,nbr_anomalies,int(base["discord length"][idx]))  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme

        if key=="our":
            real_scores, scores_label, identified,score,best_param, time_taken_1= DragStream.test(dataset,df[column].values,right,nbr_anomalies,int(base["discord length"][idx]))  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            #real_scores, scores_label, identified,score,best_param, time_taken_1= class_our.test(dataset,df[column].values,right,nbr_anomalies,int(base["discord length"][idx]))  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme

        if key=="matrix_profile":
            real_scores, scores_label, identified,score,best_param, time_taken_1= class_LAMP.test_mp(dataset,df[column].values,right,nbr_anomalies,int(base["discord length"][idx]))  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            print("*****")
        if key=="hotsax":
            real_scores, scores_label, identified,score,best_param, time_taken_1= class_LAMP.test_hotsax(dataset,df[column].values,right,nbr_anomalies,int(base["discord length"][idx]))  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme




        df["anomaly_score"]=real_scores
        df["label"]=scores_label#[0 if i<threshold else 1 for i in scores ]
        
        #print("anomalies at:" , identified, "while real are at:", str(base["Position discord"][idx]) )
        
        directory = os.path.dirname(('streaming_results/'+key+'/'+dataset))
        if not os.path.exists(directory):
            os.makedirs(directory)
        data_file_name=dataset.split('/')[-1]
        data_file_name =key+'_'+data_file_name
        dataset =directory+'/'+data_file_name
        df.to_csv(dataset, index=False)
        
        #thresholds.append(threshold)
        print("terminé")
        def insertion(file):
            best_params[idx]=best_param
            time_taken[idx]=time_taken_1
            merlin_score[idx] = score
            all_identified[idx] =identified
            try:
                
                base2 = pd.read_excel(file)
                print(base.iloc[idx, 0],all_identified[idx])
                base2.loc[idx, key+"_identified"] = str(all_identified[idx])
                base2.loc[idx, key+"_Overlap_merlin"] = str(score)
                base2.loc[idx, key+"best_param"] = str(best_params[idx])
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

            """if key in file:
                for key2,value in best_params[idx].items():
                    base2["best_param"+key2][idx] =best_params[idx][key2]
                base2.to_excel(file)
            else:"""
            base2 = pd.DataFrame(base2)
            base2.to_excel(file,index=False)




        with mutex:
            with open('abnormal_point_datasets.xlsx') as csv_file:
                insertion(scoring_metric+"_discord_results.xlsx")
                insertion("result/"+scoring_metric+"_"+key+"discord.xlsx")
                csv_file.flush()
        return idx, best_param,time_taken_1, score, identified
    

import multiprocessing as mp
from multiprocessing import Manager
pool =mp.Pool(3)
merlin_score=np.zeros(len(base))
time_taken = np.zeros(len(base))
best_params= ["params" for i in time_taken]
all_identified= ["no" for i in time_taken]

def test (meth) :                                                         
    
    methods= {meth:0}#, "HS-tree":0,"MILOF":0,"HS-tree":0, "iforestASD":0}#"MILOF":0}# "MILOF":class_MILOF.test, "iforestASD_SUB":iforestASD_SUB,"subSequenceiforestASD":iforestASD } #"iforestASD":iforestASD, "HStree":HStree "MILOF":MILOF
    scoring_metric=["merlin"] # ,"merlin"
    for key, method in methods.items():
        thresholds=[]
        
        for scoring  in scoring_metric:
            #dataset_test(merlin_score,best_params,time_taken,all_identified,key,1,base["Dataset"][1],scoring_metric=scoring)
            merlin_score=np.zeros(len(base))
            time_taken = np.zeros(len(base))
            best_params= ["params" for i in time_taken]
            all_identified= ["no" for i in time_taken]

            """for i, d in enumerate(base["Dataset"]):
                dataset_test(merlin_score,best_params,time_taken,all_identified,key,i,base["Dataset"][i],scoring_metric=scoring)
                #break"""
            with Manager() as mgr:
                merlin_score=mgr.list([]) + list(np.zeros(len(base)))
                time_taken = mgr.list([]) + list(np.zeros(len(base)))
                best_params= mgr.list([]) +  ["params" for i in time_taken]
                all_identified= mgr.list([]) + ["no" for i in time_taken]
                output =pool.starmap(dataset_test, [(merlin_score,best_params,time_taken,all_identified,key,i,base["Dataset"][i],scoring) for i ,dataset in enumerate(base["Dataset"])  ] )
                # print(f'Output: {output}')
                print ("**** merlin score", [element[3] for element in output])

test("our")
# Scores : 0.7777777777777779  {'cluster': 29, 'training': 747, 'window': 200, 'threshold': 4.5} [0.18181818181818182, 0.3076923076923077, 0.4444444444444445]
# [0.18181818181818182, 0.25, 0.4444444444444445] dragstream
# [0.06897740989825832, 0.47, 0.81] mp