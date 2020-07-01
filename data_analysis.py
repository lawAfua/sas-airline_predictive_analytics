import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)


class DataAnalysis:

    def explore_data(self):
        df = pd.read_csv("/home/neosof/zenx_project/sas_cargo_data/2_SHIPPING_REQUEST_OUT.csv")
        print("=================Show first lines of the csv files==============")
        print(df.head())
        print("=================Describe the files stats==================")
        print(df.describe())
        print("=================Shows the datatype of every column=================")
        print(df.dtypes)
        print("=================Show the shape of the data")
        print(df.shape)

        print("===============Show number of null values in the columns=============")
        for i in df.columns:
            ab = df[i].isnull().sum()
            if ab != 0:
                print(i + " has {} null values.".format(ab))
                print()

        # "======================Droping the unnesesory columns=============="
        cols = ['filetime', 'RN', 'label', 'Function_Type', 'Function_name', 'num_segs', 'ORIGIN_STATION', 'DEST_STATION',
                'FIRST_ORIGIN', 'LAST_DEST', 'SEQ_ID', 'PRD_ID', 'PRD_LAT_DATE', 'PRD_LAT_TIME',
                'TOA_DATE', 'TOA_TIME', 'NEXT_DATE', 'NEXT_TIME', 'PI_PART_DBKEY',
                'PI_PART_CUSTSEG', 'PI_PART_NAME', 'PI_PART_NAME', 'PI_PART_CONT_STATION', 'PI_PART_CONT_IATA_CHARG',
                'PI_PART_BILL_TO_ACCNO1', 'PI_PART_BILL_TO_ACCID1', 'PI_PART_BILL_TO_ACCNO2', 'PI_PART_BILL_TO_ACCID2',
                'PI_PART_IATA_ACCNO', 'PI_PART_IATA_ACCID', 'Record_Load_date', 'SHIPRQ_COMPLETED', 'AGENT', 'LAT_DATE',
                'LAT_TIME', 'WUNI', 'VUNI', 'SVCERROR_CD', 'SVCERROR_TX',
                'PI_PART_BILL_TO_ACCNO3', 'PI_PART_BILL_TO_ACCID3', 'linkid', 'WGT_CHRG_ACTUAL', 'TOT_BKFE']
        df.drop(cols, axis=1, inplace=True)

        print(df.describe(df['PRICE']))

        print("============= Check if null values present in any columns ==============")
        print(df.isnull().sum())

        df['Date_of_Journey'] = pd.to_datetime(df['filedate'])
        df['day_of_week'] = df['Date_of_Journey'].dt.day_name()
        df['Journey_Month'] = pd.to_datetime(df.Date_of_Journey, format='%d/%m/%Y').dt.month_name()

        sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
        sns.distplot(
            df['PRICE'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
        ).set(xlabel='Sale Price', ylabel='Count')
        plt.show()

        # skewness and kurtosis
        print("Skewness: %f" % df['PRICE'].skew())
        print("Kurtosis: %f" % df['PRICE'].kurt())
        print("=================Shows the datatype of every column=================")
        print(df.dtypes)

        print("===============Show number of null values in the columns=============")
        for i in df.columns:
            ab = df[i].isnull().sum()
            if ab != 0:
                print(i + " has {} null values.".format(ab))
                print()

        # Show product code count
        sns.countplot(x='PRD_CODE', data=df)
        plt.xticks(rotation=90)
        plt.show()

        # print("==============Convert object to datetime datatype========")
        # con = df['filedate']
        df['filedate'] = pd.to_datetime(df['filedate'])
        df.set_index('filedate', inplace=True)
        # print(df.index)
        df.plot()
        plt.show()
        #
        # df['PRICE'].resample('MS').mean()
        #
        # print(df['2019':])
        # df.plot(figsize=(15, 6))
        # plt.show()
        #
        # # Detecting outlier with quantile function
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        print(IQR)

        df = df[~((df < (Q1-1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        print(df.shape)
        #
        sns.boxplot(x=df['PRICE'])
        plt.show()

        sns.boxplot(x=df['WGT_CHRG_ACTUAL'])
        plt.show()

        sns.boxplot(x=df['TOT_BKFE'])
        plt.show()

        # Finding correlation
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
        plt.show()

        sns.scatterplot(df['PRICE'], df['WGT_CHRG_ACTUAL'])
        plt.show()

        sns.scatterplot(df['PRICE'], df['TOT_BKFE'])
        plt.show()
                                                                                                                        

if __name__ == "__main__":

    DataAnalysis().explore_data()

    # df = pd.DataFrame().assign(a=np.random.randint(-10, 11, 100))
