from drag_stream import stream_discord
import numpy as np
import pandas as pd

import plotly.express as px


df = pd.read_csv("dataset/nab-data/realKnownCause/ambient_temperature_system_failure.csv", usecols=['value'])
right = np.array([3721,6180])
nbr_anomalies = 2
gap = int(df.shape[0]/100)


C, S, C_score, cluster = stream_discord(df['value'].values, 362, 10, 1375, 10)

color = np.zeros(len(C_score))
color[3721] = 1
color[6180] = 1


px.scatter(x=df.index, y=C_score, color=color).show()
px.scatter(x=df.index, y=df['value'], color=color).show()