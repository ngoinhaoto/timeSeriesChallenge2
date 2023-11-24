from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import plotly.express as px
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import math

tf.random.set_seed(10)
meta_model = tf.keras.models.load_model('meta.h5')
microsoft_model = tf.keras.models.load_model('microsoft.h5')

app = Flask(__name__)


meta_df = yf.download(tickers=['META'], period='4y')


meta_data = meta_df['Close'].fillna(method='ffill')
meta_dataset = meta_data.values.reshape(-1, 1)
meta_training_data_len = math.ceil(len(meta_dataset) * .8)

scaler = MinMaxScaler(feature_range=(0,1))

scaler = scaler.fit(meta_dataset)

meta_dataset = scaler.transform(meta_dataset)
#now we generate
n_lookback = 120 #len of input sequences
n_forecast = 60 #len of prediction


# meta_X = []
# meta_Y = []

# for i in range(n_lookback, len(meta_dataset) - n_forecast + 1):
#     meta_X.append(meta_dataset[i - n_lookback: i])
#     meta_Y.append(meta_dataset[i: i + n_forecast])

# meta_X = np.array(meta_X)
# meta_Y = np.array(meta_Y)

meta_lookback = meta_dataset[-n_lookback:]

meta_lookback = meta_lookback.reshape(1, n_lookback, 1)
meta_forecast = meta_model.predict(meta_lookback)
meta_forecast = scaler.inverse_transform(meta_forecast)
# print(meta_forecast)



meta_past = meta_df[['Close']][-180:].reset_index()
meta_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
meta_past['Date'] = pd.to_datetime(meta_past['Date'])
meta_past['Forecast'] = np.nan

meta_past.loc[meta_past.index[-1], 'Forecast'] = meta_past.loc[meta_past.index[-1], 'Actual']

# meta_past['Forecast'].iloc[-1] = meta_past['Actual'].iloc[-1]


meta_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
meta_future['Date'] = pd.date_range(start=meta_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
meta_future['Forecast'] = meta_forecast.flatten()
meta_future['Actual'] = np.nan

# results = meta_past.append(meta_future).set_index('Date')
results = pd.concat([meta_past, meta_future]).set_index('Date')


meta_volume = meta_df['Volume'][-1]
print(meta_volume)

meta_price_24h = results.Forecast[-n_forecast]
meta_price_24h =  "{:.2f}".format(meta_price_24h)
print(meta_price_24h)

meta_price_7d = results.Forecast[-n_forecast + 7]
meta_price_7d = "{:.2f}".format(meta_price_7d)


meta_price_today = meta_df['Close'][-1]
meta_price_today =  "{:.2f}".format(meta_price_today)

print(meta_price_today)

percentage_difference = ((float(meta_price_24h) - float(meta_price_today)) / float(meta_price_today)) * 100
percentage_difference = "{:.2f}".format(percentage_difference)

percentage_difference = float(percentage_difference)

meta_percentage_difference_7d = ((float(meta_price_7d) - float(meta_price_today)) / float(meta_price_today)) * 100
meta_percentage_difference_7d = "{:.2f}".format(meta_percentage_difference_7d)
meta_percentage_difference_7d = float(meta_percentage_difference_7d)


#microsoft

microsoft_df = yf.download(tickers=['MSFT'], period='4y')
microsoft_data = microsoft_df['Close'].fillna(method='ffill')
microsoft_dataset = microsoft_data.values.reshape(-1, 1)
microsoft_training_data_len = math.ceil(len(microsoft_dataset) * .8)
microsoft_scaler = MinMaxScaler(feature_range=(0,1))

microsoft_scaler = microsoft_scaler.fit(microsoft_dataset)

microsoft_dataset = microsoft_scaler.transform(microsoft_dataset)

n_lookback = 120 #len of input sequences
n_forecast = 60 #len of prediction


# microsoft_X = []
# microsoft_Y = []

# for i in range(n_lookback, len(microsoft_dataset) - n_forecast + 1):
#     microsoft_X.append(microsoft_dataset[i - n_lookback: i])
#     microsoft_Y.append(microsoft_dataset[i: i + n_forecast])
# microsoft_X = np.array(microsoft_X)
# microsoft_Y = np.array(microsoft_Y)


microsoft_lookback = microsoft_dataset[-n_lookback:]

microsoft_lookback = microsoft_lookback.reshape(1, n_lookback, 1)
microsoft_forecast = microsoft_model.predict(microsoft_lookback)
microsoft_forecast = microsoft_scaler.inverse_transform(microsoft_forecast)

# print(microsoft_forecast)

microsoft_past = microsoft_df[['Close']][-180:].reset_index()
microsoft_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
microsoft_past['Date'] = pd.to_datetime(microsoft_past['Date'])
microsoft_past['Forecast'] = np.nan

microsoft_past.loc[microsoft_past.index[-1], 'Forecast'] = microsoft_past.loc[microsoft_past.index[-1], 'Actual']

# microsoft_past['Forecast'].iloc[-1] = microsoft_past['Actual'].iloc[-1]


microsoft_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
microsoft_future['Date'] = pd.date_range(start=microsoft_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
microsoft_future['Forecast'] = microsoft_forecast.flatten()
microsoft_future['Actual'] = np.nan



microsoft_results = pd.concat([microsoft_past, microsoft_future]).set_index('Date')
microsoft_volume = microsoft_df['Volume'][-1]

microsoft_price_24h = microsoft_results.Forecast[-n_forecast]
microsoft_price_24h =  "{:.2f}".format(microsoft_price_24h)
print(microsoft_price_24h)


microsoft_price_today = microsoft_df['Close'][-1]
microsoft_price_today =  "{:.2f}".format(microsoft_price_today)


microsoft_percentage_difference = ((float(microsoft_price_24h) - float(microsoft_price_today)) / float(microsoft_price_today)) * 100
microsoft_percentage_difference = "{:.2f}".format(microsoft_percentage_difference)

microsoft_percentage_difference = float(microsoft_percentage_difference)


microsoft_price_7d = microsoft_results.Forecast[-n_forecast + 7]
microsoft_price_7d = "{:.2f}".format(microsoft_price_7d)

microsoft_percentage_difference_7d = ((float(microsoft_price_7d) - float(microsoft_price_today)) / float(microsoft_price_today)) * 100
microsoft_percentage_difference_7d = "{:.2f}".format(microsoft_percentage_difference_7d)
microsoft_percentage_difference_7d = float(microsoft_percentage_difference_7d)




@app.route('/')
def index():
    fig_meta = px.line(results, x=results.index, y=['Actual', 'Forecast'], title='Meta Forecasting in 2 months')
    fig_meta.add_shape(
        go.layout.Shape(
            type="line",
            x0=results.index[-n_forecast], y0=results['Actual'].min(),
            x1=results.index[-n_forecast], y1=results['Actual'].max(),
            line=dict(color="red", width=1, dash="dash")
        )
    )
    
    fig_microsoft = px.line(microsoft_results, x=microsoft_results.index, y=['Actual', 'Forecast'], title='Microsoft Forecasting in 2 months')
    fig_microsoft.add_shape(
        go.layout.Shape(
            type="line",
            x0=microsoft_results.index[-n_forecast], y0=microsoft_results['Actual'].min(),
            x1=microsoft_results.index[-n_forecast], y1=microsoft_results['Actual'].max(),
            line=dict(color="red", width=1, dash="dash")
        )
    )



    
    # Convert the Plotly figure to HTML
    div_meta = fig_meta.to_html(full_html=False)
    div_microsoft = fig_microsoft.to_html(full_html=False)

    # Render the HTML template with the meta graph in the appropriate section
    return render_template('/index.html', microsoft_percentage_difference=microsoft_percentage_difference, microsoft_volume=microsoft_volume,
    microsoft_price_24h=microsoft_price_24h,microsoft_price_today=microsoft_price_today,
    microsoft_price_7d=microsoft_price_7d, microsoft_percentage_difference_7d=microsoft_percentage_difference_7d,
    meta_price_7d=meta_price_7d, meta_percentage_difference_7d=meta_percentage_difference_7d,
    div_meta=div_meta, div_microsoft=div_microsoft,meta_volume=meta_volume, meta_price_24h=meta_price_24h, meta_price_today = meta_price_today, percentage_difference=percentage_difference)


@app.route('/meta')
def meta():
    # Code to fetch data and render the Meta page
    fig_meta_df = px.line(meta_df, x=meta_df.index, y='Close', title='Meta Data')
    div_meta_df = fig_meta_df.to_html(full_html=False)

    fig_meta = px.line(results, x=results.index, y=['Actual', 'Forecast'], title='Meta Forecasting in 2 months')
    fig_meta.add_shape(
        go.layout.Shape(
            type="line",
            x0=results.index[-n_forecast], y0=results['Actual'].min(),
            x1=results.index[-n_forecast], y1=results['Actual'].max(),
            line=dict(color="red", width=1, dash="dash")
        )
    )
    div_meta = fig_meta.to_html(full_html=False)


    return render_template('meta.html', 
        meta_df=meta_df,
        results=results,
        meta_price_today=meta_price_today,
        meta_price_24h=meta_price_24h,
        meta_price_7d=meta_price_7d,
        meta_percentage_difference_7d=meta_percentage_difference_7d,
        meta_volume=meta_volume,
        div_meta_df=div_meta_df ,
        div_meta=div_meta, n_forecast=n_forecast
    )

@app.route('/microsoft')
def microsoft():
    fig_microsoft_df = px.line(microsoft_df, x=microsoft_df.index, y='Close', title='Microsoft Data')
    div_microsoft_df = fig_microsoft_df.to_html(full_html=False)

    fig_microsoft = px.line(microsoft_results, x=microsoft_results.index, y=['Actual', 'Forecast'], title='Microsoft Forecasting in 2 months')
    fig_microsoft.add_shape(
        go.layout.Shape(
            type="line",
            x0=microsoft_results.index[-n_forecast], y0=microsoft_results['Actual'].min(),
            x1=microsoft_results.index[-n_forecast], y1=microsoft_results['Actual'].max(),
            line=dict(color="red", width=1, dash="dash")
        )
    )
    div_microsoft = fig_microsoft.to_html(full_html=False)

    return render_template('microsoft.html', n_forecast=n_forecast,
        microsoft_df=microsoft_df,
        microsoft_results=microsoft_results,
        microsoft_price_today=microsoft_price_today,
        microsoft_price_24h=microsoft_price_24h,
        microsoft_price_7d=microsoft_price_7d,
        microsoft_percentage_difference_7d=microsoft_percentage_difference_7d,
        microsoft_volume=microsoft_volume,
        div_microsoft_df=div_microsoft_df,
        div_microsoft=div_microsoft
    )


if __name__ == '__main__':
    app.run(debug=True) 