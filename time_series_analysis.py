import warnings
import itertools
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

import seaborn as sns

if __name__ == "__main__":

    df = pd.read_csv("/sas_cargo_data/2_SHIPPING_REQUEST_OUT.csv")

    prd_code = df.loc[df['PRD_CODE'] == 'PAA']
    print(prd_code.shape)

    print("prd_min", prd_code['filedate'].min())
    print("prd_max", prd_code['filedate'].max())

    cols = ['filetime', 'RN', 'label', 'Function_Type', 'Function_name', 'num_segs', 'ORIGIN_STATION', 'DEST_STATION',
            'FIRST_ORIGIN', 'LAST_DEST', 'CURRENCY', 'SEQ_ID', 'PRD_ID', 'PRD_CODE', 'PRD_LAT_DATE', 'PRD_LAT_TIME',
            'TOA_DATE', 'TOA_TIME', 'WGT_CHRG_ACTUAL', 'TOT_BKFE', 'NEXT_DATE', 'NEXT_TIME', 'PI_PART_DBKEY',
            'PI_PART_CUSTSEG', 'PI_PART_NAME', 'PI_PART_NAME', 'PI_PART_CONT_STATION', 'PI_PART_CONT_IATA_CHARG',
            'PI_PART_BILL_TO_ACCNO1', 'PI_PART_BILL_TO_ACCID1', 'PI_PART_BILL_TO_ACCNO2', 'PI_PART_BILL_TO_ACCID2',
            'PI_PART_IATA_ACCNO', 'PI_PART_IATA_ACCID', 'Record_Load_date', 'SHIPRQ_COMPLETED', 'AGENT', 'LAT_DATE',
            'LAT_TIME', 'WUNI', 'VUNI', 'SVCERROR_CD', 'SVCERROR_TX',
            'PI_PART_BILL_TO_ACCNO3', 'PI_PART_BILL_TO_ACCID3', 'linkid']

    prd_code.drop(cols, axis=1, inplace=True)
    prd_code = prd_code.sort_values('filedate')
    print(prd_code.isnull().sum())

    # con = prd_code['filedate']
    # df1 = prd_code.loc[prd_code['filedate'] == '2019-09-01']
    df1 = prd_code

    df1['filedate'] = pd.to_datetime(df1['filedate'])
    df1["PRICE"] = df1["PRICE"].apply(pd.to_numeric, errors='coerce')
    print(df1['PRICE'].describe())
    df1 = df1.resample('d', on='filedate').mean().dropna(how='all')
    # print(df1.head())

    result = adfuller(df1)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    Q1 = df1.quantile(0.25)
    Q3 = df1.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)

    df = df1[~((df < (Q1 - 1.5 * IQR)) | (df1 > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(df.shape)

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Seasonal ARIMA result...')
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(df1,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=True,
                                                enforce_invertibility=False)
                results = mod.fit(disp=0)
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue

    mod = sm.tsa.statespace.SARIMAX(df1,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=True,
                                    enforce_invertibility=False)
    results = mod.fit(disp=0)
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()

    pred = results.get_prediction(start=pd.to_datetime('2020-01-14'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = df1['2019-09-01':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Furniture Sales')
    plt.legend()
    plt.show()
    print(results.summary().tables[1])
