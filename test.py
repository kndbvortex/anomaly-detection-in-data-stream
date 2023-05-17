from drag_stream import stream_discord, class_our
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import plotly.express as px

# {'cluster': 15, 'training': 389, 'window': 100, 'threshold': 4.5}


def drag_reproduce(df, param):
    _, _, scores, _ = stream_discord(
        df, param['window'], param['threshold'], param['training'], param['cluster'])

    # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
    real_scores, scores_label, identified, score, best_param, time_taken_1 = class_our.test(
        dataset, df[column].values, right, nbr_anomalies, int(base["discord length"][idx]))


def main():
    path = "~/Téléchargements/UCR_TimeSeriesAnomalyDatasets2021/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/"
    # df = pd.read_csv("dataset/nab-data/realKnownCause/ambient_temperature_system_failure.csv", usecols=['value'])
    df = pd.read_csv(path+"182_UCR_Anomaly_qtdbSel1005V_4000_12400_12800.txt",
                     header=None, names=['value'], sep="\s+")

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


def make_dataset():
    dir = '/home/dbkamgangu/Téléchargements/UCR_TimeSeriesAnomalyDatasets2021/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/'
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    dataset = pd.DataFrame(columns=['Dataset', 'Dataset length', 'Type', '#discords', 'Position discord', 'discord length', 'Max train'])
    for file in onlyfiles:
        df = pd.read_csv(dir+file, sep='\s+', header=None)
        *_ , max_training, start, end = file.split('.txt')[0].split('_')
        max_training, start, end = int(max_training), int(start), int(end)
        dataset.loc[len(dataset),:] = [file, len(df), 'UCR', 1, start, end-start+1, max_training]
        df.drop(df.index, inplace=True)
    dataset.to_excel('discord_ucr.xlsx')


if __name__ == '__main__':
    df = pd.read_excel('discord_ucr.xlsx')
    px.line(df, x=df.index, y=['Max train']).show()
    print(df['Max train'].describe())
