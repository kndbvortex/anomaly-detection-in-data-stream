import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots


urls = {
    'ecg': [
        'http://www.cs.ucr.edu/~eamonn/discords/ECG_data.zip',
        'http://www.cs.ucr.edu/~eamonn/discords/mitdbx_mitdbx_108.txt', #Colone 1
        'http://www.cs.ucr.edu/~eamonn/discords/qtdbsele0606.txt',
        'http://www.cs.ucr.edu/~eamonn/discords/chfdbchf15.txt',
        'http://www.cs.ucr.edu/~eamonn/discords/qtdbsel102.txt'
    ],
    'gesture': [
        'http://www.cs.ucr.edu/~eamonn/discords/ann_gun_CentroidA'
    ],
    'space_shuttle': [
        'http://www.cs.ucr.edu/~eamonn/discords/TEK16.txt',
        'http://www.cs.ucr.edu/~eamonn/discords/TEK17.txt',
        'http://www.cs.ucr.edu/~eamonn/discords/TEK14.txt'
    ]
}


def get_data():
    urls = dict()
    urls['ecg'] = [
        'http://www.cs.ucr.edu/~eamonn/discords/ECG_data.zip',
        'http://www.cs.ucr.edu/~eamonn/discords/mitdbx_mitdbx_108.txt',
        'http://www.cs.ucr.edu/~eamonn/discords/qtdbsele0606.txt',
        'http://www.cs.ucr.edu/~eamonn/discords/chfdbchf15.txt',
        'http://www.cs.ucr.edu/~eamonn/discords/qtdbsel102.txt'
    ]
    urls['gesture'] = ['http://www.cs.ucr.edu/~eamonn/discords/ann_gun_CentroidA']
    urls['space_shuttle'] = [
        'http://www.cs.ucr.edu/~eamonn/discords/TEK16.txt',
        'http://www.cs.ucr.edu/~eamonn/discords/TEK17.txt',
        'http://www.cs.ucr.edu/~eamonn/discords/TEK14.txt'
    ]
    # urls['respiration'] = [
    #     'http://www.cs.ucr.edu/~eamonn/discords/nprs44.txt',
    #     'http://www.cs.ucr.edu/~eamonn/discords/nprs43.txt'
    # ]
    urls['power_demand'] = [
        'http://www.cs.ucr.edu/~eamonn/discords/power_data.txt'
    ]


if __name__ == '__main__':
    # d = pd.read_csv('~/vis.txt', header=None, sep='\s+')
    # plt.plot(d[0])
    # plt.plot(d[1])
    # plt.plot(d[2])
    
    # df = pd.read_csv(urls['ecg'][2], sep='\t', header=None)
    # df2 = pd.read_csv(urls['ecg'][1], sep='\s+', header=None)
    # df[3] = df[2] - df[1]
    # fig = px.line(df, y=[1,2,3], x=0)
    # fig2 = px.line(df2[:16000], y=[1,2,0])
    
    # fig.line(df, y=1)
    # fig.line(df, y=2)
    # fig.show()
    # fig2.show()
    # plt.plot(df[0])
    # plt.plot(df[1])
    # plt.plot(df[2])
    # print(df.shape)
    # print(df2)
    
    a = pd.read_csv('streaming_results/our/our_231_UCR_Anomaly_mit14134longtermecg_8763_47530_47790.txt', sep=',')
    a['new'] = a['value']
    a['new'][:47530] = None
    a['new'][47790:] = None

    fig = px.line(a)
    
    
    b = pd.read_csv('streaming_results/our/our_227_UCR_Anomaly_mit14134longtermecg_11231_29000_29100.txt', sep=',')
    b['new'] = b['value']
    b['new'][:29000] = None
    b['new'][29100:] = None
    fig1 = px.line(b)
    fig1.show()
    
    # fig = px.line(a)
    
    # fig.show()
    plt.show()
    

