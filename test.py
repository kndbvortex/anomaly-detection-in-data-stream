from drag_stream import stream_discord
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px

def scoring(scores, right, gap):
    identified = np.where(scores.squeeze() == 1)[0]
    sub_identified = np.array([])
    sub_right = np.array([])
    for identify in identified:
        sub_identified = np.concatenate(
            [sub_identified, np.arange(identify, identify+gap)])
    for identify in right:
        identify = int(identify)
        sub_right = np.concatenate(
            [sub_right, np.arange(identify, identify+gap)])
    sub_identified = np.unique(sub_identified)
    sub_right = np.unique(sub_right)
    recall = len(np.intersect1d(
        sub_identified, sub_right))/len(sub_right)
    precision = len(np.intersect1d(
        sub_identified, sub_right))/len(sub_identified)
    try:
        score = 2*(recall*precision)/(recall+precision)
    except:
        score = 0.0
    return score

def score_to_label(scores,right,  gap):

    thresholds = np.unique(scores)
    f1_scores = []
    for threshold in thresholds:
        labels = np.where(scores < threshold, 0, 1)
        print('Label'.center(10, "="))
        print(labels)
        print('Threshold'.center(10, "="))
        print()
        f1_scores.append(scoring(labels, right, gap))
    q = list(zip(f1_scores, thresholds))
    thres = sorted(q, reverse=True, key=lambda x: x[0])[0][1]
    threshold = thres
    arg = np.where(thresholds == thres)

    # i will throw only real_indices here. [0 if i<threshold else 1 for i in scores ]
    return np.where(scores < threshold, 0, 1)

def main():
    path = "~/Téléchargements/UCR_TimeSeriesAnomalyDatasets2021/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/"
    # df = pd.read_csv("dataset/nab-data/realKnownCause/ambient_temperature_system_failure.csv", usecols=['value'])
    df = pd.read_csv(path+"182_UCR_Anomaly_qtdbSel1005V_4000_12400_12800.txt", header=None, names=['value'], sep="\s+")
    
    # right = np.array([3721,6180])
    # nbr_anomalies = 2
    # gap = int(df.shape[0]/100)


    # C, S, C_score, cluster = stream_discord(df['value'].values, 362, 10, 1375, 10)

    # color = np.zeros(len(C_score))
    # color[3721] = 1
    # color[6180] = 1

    plt.plot(df['value'])
    plt.show()
    px.line(df, y='value').show()
    # px.scatter(x=df.index, y=df['value'], color=color).show()


if __name__=='__main__':
    main()