import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import pandas_datareader as web
import plotly.graph_objects as go

from streamlit_efficient_frontier import *
from streamlit_var import *

import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot_plotly
import plotly.offline as py
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot_components_plotly
st.title('پیش‌بینی اتوماتیک با استفاده از پکیج پیامبر فیس‌بوک')
st.sidebar.title('پیش‌بینی اتوماتیک با استفاده از پکیج پیامبر فیس‌بوک')


"""
این برنامه به طور خاص برای پیش‌بینی اتوماتیک با استفاده از داده‌ی دانلود شده از تریدینگ‌ویوو طراحی شده است 

ساخت: احمد و امین مصطفوی

"""

"""
 مرحله‌ی اول: وارد کردن داده‌ها از تریدینگ‌ویوو
"""
df = st.file_uploader(
    'داده‌های دانلود شده از تریدینگ‌ویوو را بارگذاری کنید',
    type='csv')

if df is not None:
    data = pd.read_csv(df)
    data.rename(columns={'time': 'ds', 'close': 'y'}, inplace=True)
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce', utc=True)
    data['ds'] = data['ds'].dt.strftime('%Y-%m-%d %H:%M')
    data.rename(columns={'Date': 'ds', 'Value': 'y'}, inplace=True)
    st.write(data)

    max_date = data['ds'].max()
    # st.write(max_date)

"""
###  مرحله‌ی دوم:انتخاب تعداد روزهای مدنظر برای پیش‌بینی

Keep in mind that forecasts become less accurate with larger forecast horizons.
"""

periods_input = st.number_input('How many periods would you like to forecast into the future?',
                                min_value=1, max_value=365)

if df is not None:
    m = Prophet(seasonality_mode='multiplicative', seasonality_prior_scale=5)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_country_holidays(country_name='US')
    m.fit(data)

"""
### Step 3: Visualize Forecast Data

The below visual shows future predicted values. "yhat" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)

    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered = fcst[fcst['ds'] > max_date]
    st.write(fcst_filtered)

    """
    The next visual shows the actual (black dots) and predicted (blue line) values over time.
    """
    fig = plot_plotly(m, forecast, trend=True)  # This returns a plotly Figure
    st.write(fig)

    """
    The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
    """
    fig2 = plot_components_plotly(m, forecast,
                                 figsize=(800, 175))
    st.write(fig2)

"""
### Step 4: Download the Forecast Data

The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)
 
st.write('ساخته شده توسط احمد مصطفوی')
st.write("[GitHub](https://github.com/verethraghnah) |", 
         "[LinkedIn](https://www.linkedin.com/in/ahmad-mostafavi/) |",
         "[site](https://www.dadehkav.tech/)")
