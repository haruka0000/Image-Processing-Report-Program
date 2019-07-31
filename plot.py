import plotly.offline as offline
import plotly.graph_objs as go

def plot(sigma=[], lsm=[], mle=[], kcr=[], lsm_B=[], mle_B=[]):
    data_LSM = go.Scatter(
        x = sigma,
        y = lsm,
        name = 'LSM'
    )
    data_MLE = go.Scatter(
        x = sigma,
        y = mle,
        name = 'MLE'
    )

    data_KCR = go.Scatter(
        x = sigma,
        y = kcr,
        name = 'KCR'
    )
    data_LSM_B = go.Scatter(
        x = sigma,
        y = lsm_B,
        name = 'lsm'
    )    
    data_MLE_B = go.Scatter(
        x = sigma,
        y = mle_B,
        name = 'mle'
    )

    data = go.Data([data_LSM, data_MLE, data_KCR])
    data2 = go.Data([data_LSM_B, data_MLE_B])
    layout = go.Layout(
        title = 'RMS誤差',
        xaxis = dict(
            title = '標準偏差',
        ),
        yaxis = dict(
            title = 'RMS誤差',
        )
    )

    layout2 = go.Layout(
        title = '偏差',
        xaxis = dict(
            title = '標準偏差',
        ),
        yaxis = dict(
            title = '偏差',
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig2 = go.Figure(data=data2, layout=layout2)
    offline.plot(fig, filename='Final_report')
    offline.plot(fig2, filename='Final_report2')