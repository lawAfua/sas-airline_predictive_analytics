import numpy as np
import pandas as pd
import statsmodels.api as sm
from currency_converter import CurrencyConverter
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


def currency_exchange(df1):
    """this function will convert currencies"""
    c = CurrencyConverter()
    list_price = []
    print(df1.shape)
    for d in df1.iteritems():
        try:
            list_price.append(c.convert(d[1], d[0], 'DKK'))
        except ValueError:
            list_price.append(d[1])
    return list_price


# if __name__ == '__main__':
    # df = pd.read_csv("features_with_merging.csv", low_memory=False)
def train_regression(df):
    # df = pd.read_csv("merged_features11.csv", low_memory=False)

    df.drop(['linkid', '|linkid|_x', '|linkid|_y', 'PRD_CODE'], inplace=True, axis=1)

    price = currency_exchange(df.set_index('CURRENCY')['PRICE'])
    df['Exchanged_Price'] = price
    wgt_price = currency_exchange(df.set_index('CURRENCY')['WGT_CHRG_ACTUAL'])
    df['Exchanged_Wgt_Chrg_Actual'] = wgt_price
    tot_price = currency_exchange(df.set_index('CURRENCY')['TOT_BKFE'])
    df['Exchanged_Tot_Bkfe'] = tot_price

    df['no_of_request'] = 1

    df.dropna(inplace=True)
    df.columns = df.columns.str.strip('|')
    df1 = df.groupby(['filedate', 'PRD', 'ORIGIN_STATION', 'DEST_STATION', 'WUNI',  'VUNI'])[
        "Exchanged_Price", "Exchanged_Wgt_Chrg_Actual",
        "Exchanged_Tot_Bkfe", "no_of_request", 'WGT', 'VOL', 'PIECES', 'ACTUAL_RATE', 'TOT_CHARGE',
        'SUGGESTED_WGT', 'SUGGESTED_VOL', 'ORICAP_WGT', 'ORICAP_VOL', 'SEG_SPACE_WGT', 'SEG_SPACE_PCS',
        'SEG_SPACE_VOL'].apply(
        lambda x: x.astype(int).sum()).reset_index()

    mask = (np.abs(stats.zscore(df1['Exchanged_Price'])) > 1.5)
    df1.Exchanged_Price = df1.Exchanged_Price.mask(mask).interpolate()

    mask = (np.abs(stats.zscore(df1['Exchanged_Wgt_Chrg_Actual'])) > 1.5)
    df1.Exchanged_Wgt_Chrg_Actual = df1.Exchanged_Wgt_Chrg_Actual.mask(mask).interpolate()

    mask = (np.abs(stats.zscore(df1['Exchanged_Tot_Bkfe'])) > 1.5)
    df1.Exchanged_Tot_Bkfe = df1.Exchanged_Tot_Bkfe.mask(mask).interpolate()

    corr = df1.corr()["no_of_request"]
    print("Correlations")
    print(corr)
    # sns.countplot(x='ORIGIN_STATION', data=df1[:100000], facecolor=(0, 0, 0, 0), linewidth=5,
    #               edgecolor=sns.color_palette("dark", 3))
    # plt.xticks(rotation=90)
    # plt.show()
    #
    # sns.countplot(x='DEST_STATION', data=df1[:8000], facecolor=(0, 0, 0, 0), linewidth=5,
    #               edgecolor=sns.color_palette("dark", 3))
    # plt.xticks(rotation=90)
    # plt.show()
    #
    # sns.countplot(x='PRD', data=df1, facecolor=(0, 0, 0, 0), linewidth=5,
    #               edgecolor=sns.color_palette("dark", 3))
    # plt.xticks(rotation=90)
    # plt.show()
    #
    # df1['Exchanged_Price'].plot()
    # plt.show()
    #
    # df1['no_of_request'].plot()
    # plt.show()
    print("++++++++++++++++++++Annova test result+++++++++++++++++++")
    moore_lm = ols(
        'no_of_request ~ C(ORIGIN_STATION)+C(DEST_STATION)+C('
        'PRD)',
        data=df1).fit()

    table = sm.stats.anova_lm(moore_lm, typ=2)
    print(table)

    moore_lm = ols(
        'no_of_request ~ C(ORIGIN_STATION)+C(DEST_STATION)+C('
        'PRD)+Exchanged_Price+Exchanged_Wgt_Chrg_Actual+Exchanged_Tot_Bkfe',
        data=df1).fit()
    print(moore_lm.summary())

    # print(df1.columns.tolist())
    df1 = pd.get_dummies(df1, columns=['PRD', 'DEST_STATION', 'ORIGIN_STATION', 'WUNI', 'VUNI'])
    no_of_request_y = df1['no_of_request']
    del df1['no_of_request']
    del df1['filedate']
    X_train, X_test, y_train, y_test = train_test_split(df1, no_of_request_y, random_state=0)
    reg = KNeighborsRegressor(n_neighbors=20)

    reg.fit(X_train, y_train)
    print("Test accuracy", reg.score(X_test, y_test))
    print("Train accuracy", reg.score(X_train, y_train))
    print(reg.predict(X_test))

    rngr = RandomForestRegressor(max_depth=2, random_state=0)
    rngr.fit(X_train, y_train)

    print("Test accuracy random forest", rngr.score(X_test, y_test))

    grid = {'n_estimators': [100, 200]}

    # only one core is used (GOOD)
    rfr = RandomForestRegressor(n_estimators=100)
    rfr.fit(X_train, y_train)
    print("Test accuracy random forest", rfr.score(X_test, y_test))
    print(rfr.predict(X_test))

    filename = 'finalized_model.sav'
    pickle.dump(rfr, open(filename, 'wb'))

    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, y_test)
    # print(loaded_model.predict(X_test))

    print(result)