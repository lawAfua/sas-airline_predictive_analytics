import itertools
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMAResults

# from statsmodels.tsa.arima_model import SARIMAResults

warnings.filterwarnings('ignore')


def adfuller_test(sales):
    """Method will test weather data is stationary or not"""
    rolmean = pd.Series(sales).rolling(window=12).mean()
    rolstd = pd.Series(sales).rolling(window=12).std()

    # plt.plot(sales, color='blue', label='Original')
    # plt.plot(rolmean, color='red', label='Rolling Mean')
    # plt.plot(rolstd, color='black', label='Rolling Std')
    # plt.legend(loc='best')
    # plt.title('Rolling Mean &amp; Standard Deviation')
    # plt.xlabel("Date")
    # plt.ylabel('No Of Booking Requests')
    # plt.show()

    result = adfuller(sales)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. "
              "Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


def predict_sas(df):
    df.dropna(inplace=True)
    df["no_of_requests"] = 1
    print(df["no_of_requests"].describe())
    df.rename(columns={'index': 'filedate'}, inplace=True)
    df['filedate'] = pd.to_datetime(df['filedate'])
    df["no_of_requests"] = df["no_of_requests"].apply(pd.to_numeric, errors='coerce')
    df = df.resample('d', on='filedate').sum().dropna(how='all')
    print(df["no_of_requests"].describe())
    # sns.set_style('darkgrid')
    # df["no_of_requests"].plot(kind='line', legend='reverse', title='Visualizing average number of request plot')
    # plt.xlabel("Month'S")
    # plt.ylabel('No Of Booking Requests')
    # plt.legend(loc='upper right', shadow=True)
    # plt.show()

    # result = seasonal_decompose(df["no_of_requests"])
    # result.plot()
    # pyplot.show()

    adfuller_test(df["no_of_requests"])

    df['Booking First Difference'] = df['no_of_requests'] - df['no_of_requests'].shift(1)

    # print(df["Booking First Difference"].describe())
    adfuller_test(df["Booking First Difference"].dropna())

    # fig = plt.figure(figsize=(12, 8))
    # ax1 = fig.add_subplot(211)
    # fig = sm.graphics.tsa.plot_acf(df["Booking First Difference"].iloc[6:], lags=13, ax=ax1)
    # ax2 = fig.add_subplot(212)
    # fig = sm.graphics.tsa.plot_pacf(df["Booking First Difference"].iloc[6:], lags=13, ax=ax2)
    # plt.show()

    train = df["no_of_requests"].dropna()[:100]
    test = df["no_of_requests"].dropna()[100:]
    print(train.head())

    param_dict = {}
    p = d = q = range(0, 5)
    pdq = list(itertools.product(p, d, q))

    for param in pdq:
        try:
            model = ARIMA(train, order=param)
            results = model.fit(disp=0)
            print(param, results.aic)
            param_dict[abs(results.aic)] = param
        except:
            continue

    arima_param = param_dict.get(min(list(param_dict.keys())))
    print("Training on order", arima_param)
    model = ARIMA(train, order=arima_param)
    results = model.fit(disp=0)
    prediction = results.forecast(steps=36)[0]
    plt.plot(test.values)
    plt.plot(prediction, color='red')
    plt.xlabel("Day'S")
    plt.ylabel('No Of Booking Requests')
    plt.show()
    print(prediction)

    results.save('model_arima.pkl')
    # load model
    loaded = ARIMAResults.load('model_arima.pkl')
    future = loaded.forecast(steps=20)[0]
    print(future)

    # seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    # print('Seasonal ARIMA result...')
    # for param in pdq:
    #     for param_seasonal in seasonal_pdq:
    #         try:
    #             mod = sm.tsa.statespace.SARIMAX(train,
    #                                             order=param,
    #                                             seasonal_order=param_seasonal,
    #                                             enforce_stationarity=True,
    #                                             enforce_invertibility=False)
    #             results = mod.fit(disp=0)
    #             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
    #         except:
    #             continue

    mod = sm.tsa.statespace.SARIMAX(train,
                                    order=arima_param,
                                    seasonal_order=(0, 1, 0, 52),
                                    enforce_stationarity=True,
                                    enforce_invertibility=False)
    results = mod.fit(disp=0)
    prediction = results.forecast(steps=36)
    print("Prediction", prediction)
    results.save('model_sarima.pkl')
    # load model
    loaded = ARIMAResults.load('model_sarima.pkl')
    # plt.plot(test.values)
    # plt.plot(prediction.values, color='red')
    # plt.xlabel("Day'S")
    # plt.ylabel('No Of Booking Requests')
    # plt.show()

# if __name__ == '__main__':
#
#     df = pd.read_csv('../Models/feature_file.csv')
#     predict_sas(df)
