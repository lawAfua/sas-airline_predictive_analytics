import pickle

import pandas as pd
from currency_converter import CurrencyConverter
from sklearn.model_selection import train_test_split


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


if __name__ == '__main__':
    # df = pd.read_csv("features_with_merging.csv", low_memory=False)
    df = pd.read_csv("merged_features11.csv", low_memory=False)

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

    df1 = pd.get_dummies(df1, columns=['PRD', 'DEST_STATION', 'ORIGIN_STATION', 'WUNI', 'VUNI'])
    no_of_request_y = df1['no_of_request']
    del df1['no_of_request']
    del df1['filedate']
    X_train, X_test, y_train, y_test = train_test_split(df1, no_of_request_y, random_state=0)

    filename = 'finalized_model.sav'

    x_pred = X_test[:30].shift(1)
    X_test = x_pred.iloc[1:]

    # x_pred = x_pred.dropna(inplace=True)
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(loaded_model.predict(X_test))
