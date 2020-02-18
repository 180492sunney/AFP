import pandas as pd
import numpy as np
from pdb import set_trace
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class Factors:
    def __init__(self):
        self.fundamentals = pd.read_csv('Fundamentals.csv')
        self.price_data = pd.read_csv('price_data.csv')
        self.vol = pd.read_csv('vol_data.csv')
        self.betas = pd.read_csv('betas.csv')
        self.industry_class = pd.read_csv('Industry.csv')

    def combine_data(self,load=1):
        if load==0:
            self.fundamentals['public_date'] = pd.to_datetime(self.fundamentals['public_date'])
            self.fundamentals['month'] = self.fundamentals['public_date'].dt.month
            self.fundamentals['year'] = self.fundamentals['public_date'].dt.year
            self.vol['date'] = pd.to_datetime(self.vol['date'])
            self.vol['month'] = self.vol['date'].dt.month
            self.vol['year'] = self.vol['date'].dt.year
            self.betas['DATE'] = pd.to_datetime(self.betas['DATE'], format='%Y%m%d')
            self.betas['month'] = self.betas['DATE'].dt.month
            self.betas['year'] = self.betas['DATE'].dt.year

            #fundamentals and vol
            self.fundamentals = pd.merge(self.fundamentals,self.vol[['TICKER','year','month','1M_vol','3M_vol']],left_on=['Ticker','year','month'],right_on=['TICKER','year','month'],how='left')
            #fundamentals and betas
            self.fundamentals = pd.merge(self.fundamentals ,self.betas[['TICKER','year','month','b_mkt','b_smb','b_hml','b_umd']],left_on=['Ticker','year','month'],right_on=['TICKER','year','month'],how='left')
            #fundamentals and industry
            self.fundamentals = pd.merge(self.fundamentals, self.industry_class[['Symbol','GICS Sector']], left_on=['Ticker'], right_on=['Symbol'],
                                         how='inner')
            self.fundamentals.rename(columns={"GICS Sector": "Industry"}, inplace=True)
            self.fundamentals.to_csv('processed_fundamentals.csv')
        else:
            self.fundamentals = pd.read_csv('processed_fundamentals.csv')

        #self.fundamentals = self.fundamentals.groupby('Ticker', as_index=False).fillna(method='backfill')
        #self.fundamentals = self.fundamentals.groupby('Ticker', as_index=False).fillna(method='ffill')



    def get_factors_df(self):
        self.price_data['PRC'] = self.price_data['PRC'].apply(lambda x: -x if x<0 else x)#handling negative values in the column, negativ values show bid/ask average
        self.price_data['ADJPRC'] = self.price_data['PRC']/self.price_data['CFACPR']# see https://wrds-www.wharton.upenn.edu/pages/support/data-overview/wrds-overview-crsp-us-stock-database/
        self.price_data['ADJSHRS'] = self.price_data['SHROUT']*self.price_data['CFACSHR'] # see https://wrds-www.wharton.upenn.edu/pages/support/data-overview/wrds-overview-crsp-us-stock-database/

        self.price_data['liquidity'] = self.price_data['VOL']/self.price_data['ADJSHRS']
        self.fundamentals['debt_cov'] = 1/self.fundamentals['debt_ebitda']

        self.price_data['date'] = pd.to_datetime(self.price_data['date'], format='%Y%m%d')
        self.price_data['month'] = self.price_data['date'].dt.month
        self.price_data['year'] = self.price_data['date'].dt.year

        self.fundamentals = pd.merge(self.fundamentals,self.price_data[['TICKER','year','month','liquidity']],left_on=['Ticker','year','month'],right_on=['TICKER','year','month'],how='left')
        return self.fundamentals


class PriceData:
    """
    Class to download and transform stock and factor data
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def download_data(self, filepath):
        return pd.read_csv(self.filepath)

    def calc_monthly_price(self, filepath):
        price_df = self.download_data(filepath)
        price_df = price_df[['TICKER', 'date', 'PRC']]
        price_df['date'] = pd.to_datetime(price_df['date'], format='%Y%m%d')
        price_df['ret'] = price_df.groupby(['TICKER'], as_index=False).PRC.pct_change()
        price_df.dropna(how='any', axis=0, inplace=True)

        # check for later if median is negative in a month
        first_quantile = price_df.groupby(['date'], as_index=True)['ret'].quantile(0.2)
        first_quantile.rename(columns={"ret": 'first_quantile'}, inplace=True)
        # first_quantile.set_index(['date'], inplace = True)

        second_quantile = price_df.groupby(['date'], as_index=True)['ret'].quantile(0.4)
        second_quantile.rename(columns={"ret": 'second_quantile'}, inplace=True)
        # second_quantile.set_index(['date'], inplace = True)

        third_quantile = price_df.groupby(['date'], as_index=True)['ret'].quantile(0.6)
        third_quantile.rename(columns={"ret": 'third_quantile'}, inplace=True)
        # third_quantile.set_index(['date'], inplace = True)

        fourth_quantile = price_df.groupby(['date'], as_index=True)['ret'].quantile(0.8)
        fourth_quantile.rename(columns={"ret": 'fourth_quantile'}, inplace=True)
        # fourth_quantile.set_index(['date'], inplace = True)

        new_df = pd.concat([first_quantile, second_quantile, third_quantile, fourth_quantile], join='inner', axis=1)
        new_df.columns = ['first_quantile', 'second_quantile', 'third_quantile', 'fourth_quantile']
        new_df.reset_index(inplace=True)

        price_df = pd.merge(price_df, new_df, on='date', how='inner')

        price_df['five_bucket'] = 0
        price_df.loc[price_df.ret <= price_df.first_quantile, 'five_bucket'] = -2
        price_df.loc[((price_df.ret > price_df.first_quantile) & (price_df.ret <= price_df.second_quantile)), 'five_bucket'] = -1
        price_df.loc[((price_df.ret > price_df.second_quantile) & (price_df.ret <= price_df.third_quantile)), 'five_bucket'] = 0
        price_df.loc[((price_df.ret > price_df.third_quantile) & (price_df.ret <= price_df.fourth_quantile)), 'five_bucket'] = 1
        price_df.loc[price_df.ret > price_df.fourth_quantile, 'five_bucket'] = 2

        med_ret_df = price_df.groupby(['date'], as_index=False).median()
        med_ret_df.rename(columns={"ret": 'med_ret'}, inplace=True)
        price_df = pd.merge(price_df, med_ret_df[['date', 'med_ret']], on='date', how='inner')
        price_df['two_bucket'] = 0
        price_df.loc[price_df.ret >= price_df.med_ret, 'two_bucket'] = 1
        price_df.loc[price_df.ret < price_df.med_ret, 'two_bucket'] = -1
        price_df.reset_index(inplace=True, drop=True)
        price_df['3M_mom'] = price_df.groupby(['TICKER'], as_index=False).ret.rolling(3, min_periods=3).sum().reset_index(0, drop=True)
        price_df['12M_mom'] = price_df.groupby(['TICKER'], as_index=False).ret.rolling(12, min_periods=12).sum().reset_index(0, drop=True)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df['month'] = price_df['date'].dt.month
        price_df['year'] = price_df['date'].dt.year

        return price_df


class Training:

    def __init__(self, data):
        self.data = data

    def get_cleaned_date(self, startDate, trainWindow, testWindow, bucket='two_bucket'):
        data_processed = self.data[(self.data['public_date'] >= startDate) & (self.data['public_date'] < (startDate + pd.DateOffset(months=(trainWindow + testWindow))))]
        data_processed = data_processed.groupby('Ticker', as_index=False).fillna(method='backfill')
        data_processed = data_processed.groupby('Ticker', as_index=False).fillna(method='ffill')
        # regress_cols = ['Ticker', 'public_date', 'month', 'year', 'bm', 'pe_exi', 'pe_op_dil', 'evm', 'debt_at', 'de_ratio', 'liquidity', 'roe', 'roa', 'roce', 'DIVYIELD', 'dpr', 'intcov_ratio', 'debt_ebitda', 'rect_turn', 'pay_turn', 'at_turn', 'inv_turn', 'cash_ratio', 'quick_ratio', 'curr_ratio', 'cash_conversion', '1M_vol', '3M_vol', 'debt_cov', 'Industry', '3M_mom', '12M_mom', 'b_mkt', 'b_smb', 'b_hml', 'b_umd', 'quantile']
        # regress_cols = ['Ticker', 'public_date', 'month', 'year', 'bm', 'pe_exi', 'pe_op_dil', 'evm', 'debt_at', 'de_ratio', 'liquidity', 'roe', 'roa', 'roce', 'dpr', 'intcov_ratio', 'debt_ebitda', 'rect_turn', 'pay_turn', 'at_turn', 'inv_turn', 'cash_ratio', 'quick_ratio', 'curr_ratio', 'cash_conversion', '1M_vol', '3M_vol', 'debt_cov', 'Industry', '3M_mom', '12M_mom', 'b_mkt', 'b_smb', 'b_hml', 'b_umd', 'quantile']
        data_processed.rename(columns={bucket: 'quantile'}, inplace=True)
        regress_cols = ['Ticker', 'public_date', 'month', 'year', 'bm', 'pe_exi', 'pe_op_dil', 'evm', 'debt_at',
                        'de_ratio', 'liquidity', 'roe', 'roa', 'roce', 'dpr', 'intcov_ratio', 'debt_ebitda',
                        'rect_turn', 'pay_turn', 'at_turn', 'inv_turn', 'cash_ratio', 'quick_ratio', 'curr_ratio',
                        'cash_conversion', '1M_vol', '3M_vol', 'Industry', '3M_mom', '12M_mom', 'b_mkt', 'b_smb',
                        'b_hml', 'b_umd', 'quantile']
        data_processed = data_processed[regress_cols]
        null_aggr = data_processed.isnull().sum()
        null_aggr_list = null_aggr[null_aggr != 0].index.tolist()
        for col in null_aggr_list:
            a = data_processed.groupby('Ticker').apply(lambda x: x[col].isnull().sum())
            empty_tickers = a[a != 0].index.tolist()
            for ticker in empty_tickers:
                # print(col, ticker)
                ind = data_processed[data_processed['Ticker'] == ticker]['Industry'].head(1).values[0]
                data_processed.loc[data_processed[data_processed['Ticker'] == ticker].index.tolist(), col] = data_processed[data_processed['Industry'] == ind][col].mean()
        train_data = data_processed[(data_processed['public_date'] >= startDate) & (data_processed['public_date'] < (startDate + pd.DateOffset(months=trainWindow)))]
        test_data = data_processed[(data_processed['public_date'] >= (startDate + pd.DateOffset(months=trainWindow))) & (data_processed['public_date'] < (startDate + pd.DateOffset(months=(trainWindow + testWindow))))]
        return train_data, test_data

    def adaBoost_train(self, train_data, test_data):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split

        # train_data = train_data[train_data['debt_cov'] != float("inf")]
        # test_data = test_data[test_data['debt_cov'] != float("inf")]
        # X = train_data.drop(columns=['Ticker', 'public_date', 'Industry', 'quantile'])
        # y = train_data['quantile']
        # train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
        train_X = train_data.drop(columns=['Ticker', 'public_date', 'month', 'year', 'Industry', 'quantile'])
        train_y = train_data['quantile']
        test_X = test_data.drop(columns=['Ticker', 'public_date', 'month', 'year', 'Industry', 'quantile'])
        test_y = test_data['quantile']
        #print(train_X.shape)
        #print(test_X.shape)
        # print(train_y.shape)
        # print(test_y.shape)
        classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
        classifier.fit(train_X, train_y)
        predictions = classifier.predict(test_X)
        test_data['prediction'] = predictions
        #print(confusion_matrix(test_y, predictions))
        return test_data


class Portfolio:

    def __init__(self, price_data):
        self.price_df = price_data

    def construction(self, test_data, quantiles):
        stocks_long = list(test_data[test_data['prediction'].isin([a for a in quantiles if a > 0])]['Ticker'].unique())
        stocks_short = list(test_data[test_data['prediction'].isin([a for a in quantiles if a < 0])]['Ticker'].unique())
        month, year = test_data['month'].unique()[0], test_data['year'].unique()[0]
        ret_long_only = price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (
            self.price_df['TICKER'].isin(stocks_long))]['ret'].mean()
        ret_short_only = -1 * price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (
            self.price_df['TICKER'].isin(stocks_short))]['ret'].mean()
        return ret_long_only, ret_short_only, (0.5 * ret_long_only + 0.5 * ret_short_only), stocks_long, stocks_short


    def returns(self, trainObj, startDate, EndDate, trainWindow, testWindow, bucket='five_bucket', quantiles=[-2, 2], Algo='AdaBoost'):
        returns_dict = {}
        date = startDate
        while (date <= EndDate):
            print(date)
            train_data, test_data = trainObj.get_cleaned_date(date, trainWindow, testWindow, bucket)
            if Algo == 'AdaBoost':
                test_with_prediction = trainObj.adaBoost_train(train_data, test_data)
                long_only_return, short_only_return, long_short_return, _, _ = self.construction(test_with_prediction, quantiles)
                dt = test_data['public_date'].unique()[0]
                print(long_only_return, short_only_return, long_short_return)
                returns_dict[dt] = [long_only_return, short_only_return, long_short_return]
            date = date + pd.DateOffset(months=1)
        return pd.DataFrame.from_dict(returns_dict, orient='index', columns=['Long_Only', 'Short_Only', 'Long_Short'])


class Utils:
    def get_cumulative_returns_aqr(self, aqr_fp, rf_fp):
        df = pd.read_csv(aqr_fp)
        rf = pd.read_csv(rf_fp)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        rf['Date'] = pd.to_datetime(rf['Date'], dayfirst=True)
        rf['Month'] = rf['Date'].dt.month
        rf['Year'] = rf['Date'].dt.year
        df = pd.merge(df, rf, on=['Year', 'Month'], how='left')
        df['Cum_Val'] = (
                    1 + df['VALLS_VME_US90'] + df['Rate'] / 1200).cumprod()  # dividing risk free by 12 to get monthly
        df['Cum_Mom'] = (1 + df['MOMLS_VME_US90'] + df['Rate'] / 1200).cumprod()
        # returns are in decimals, need to multiply with 100 for percent returns
        return df

    def get_cumulative_returns_ours(self, returns):
        returns['Cum_L'] = (1 + returns['Long_Only']).cumprod()
        returns['Cum_S'] = (1 + returns['Short_Only']).cumprod()
        returns['Cum_LS'] = (1 + returns['Long_Short']).cumprod()
        return returns


class Plot_results:
    def __init__(self):
        self.u = Utils()

    def plot_benchmark_aqr(self):
        cum_ret = self.u.get_cumulative_returns_aqr('AQR_Val_Mom.csv', 'Treasury_1M.csv')
        plt.figure(figsize=(20, 10))
        plt.plot(cum_ret['Date_x'], cum_ret['Cum_Val'], color='blue', label='Value')
        plt.plot(cum_ret['Date_x'], cum_ret['Cum_Mom'], color='red', label='Mom')
        plt.legend()
        plt.title('Aqr Mom Factor Returns', fontsize=18)
        plt.show()

    def plot_our_results(self, returns):
        returns = self.u.get_cumulative_returns_ours(returns)
        plt.figure(figsize=(20, 10))
        plt.plot(list(returns.index), returns['Cum_L'], color='green', label='Long Only')
        plt.plot(list(returns.index), returns['Cum_S'], color='cyan', label='Short Only')
        plt.plot(list(returns.index), returns['Cum_LS'], color='magenta', label='Long_Short')
        plt.legend()
        plt.title('Aqr Mom Factor Returns', fontsize=18)
        plt.show()
        print(returns)


price_filepath = 'price_data.csv'
data = PriceData(price_filepath)
price_df = data.calc_monthly_price(price_filepath)

factors = Factors()
factors.combine_data(0)
f = factors.get_factors_df()

reg_df = pd.merge(f,price_df,left_on =['Ticker','year','month'],right_on=['TICKER','year','month'],how='inner')
print(reg_df.shape)

train = Training(reg_df)
port = Portfolio(price_df)
#train_data, test_data = train.get_cleaned_date(pd.to_datetime('28-02-2014'), 12, 1, 'five_bucket')
#test_with_prediction = train.adaBoost_train(train_data, test_data)
#port = Portfolio(price_df)
#long_only_return, short_only_return, long_short_return,_,_ = port.construction(test_with_prediction, [-2,2])
#print(long_only_return, short_only_return, long_short_return)
#hello
returns_df = port.returns(train, pd.to_datetime('28-02-2014'), pd.to_datetime('28-05-2014'), 12, 1, 'five_bucket', [-2,2], Algo='AdaBoost')
returns_df

p = Plot_results()
p.plot_benchmark_aqr()

p.plot_our_results(returns_df)