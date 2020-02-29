import pandas as pd
import numpy as np
from pdb import set_trace
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from collections import defaultdict


class Factors:
    def __init__(self):
        self.fundamentals = pd.read_csv('Fundamentals.csv')
        self.price_data = pd.read_csv('price_data.csv')
        self.vol = pd.read_csv('vol_data.csv')
        self.betas = pd.read_csv('betas.csv')
        self.industry_class = pd.read_csv('Industry.csv')

    def combine_data(self, load=1):
        if load == 0:
            self.fundamentals['public_date'] = pd.to_datetime(self.fundamentals['public_date'])
            self.fundamentals['month'] = self.fundamentals['public_date'].dt.month
            self.fundamentals['year'] = self.fundamentals['public_date'].dt.year
            self.vol['date'] = pd.to_datetime(self.vol['date'])
            self.vol['month'] = self.vol['date'].dt.month
            self.vol['year'] = self.vol['date'].dt.year
            self.betas['DATE'] = pd.to_datetime(self.betas['DATE'], format='%Y%m%d')
            self.betas['month'] = self.betas['DATE'].dt.month
            self.betas['year'] = self.betas['DATE'].dt.year

            # fundamentals and vol
            self.fundamentals = pd.merge(self.fundamentals, self.vol[['TICKER', 'year', 'month', '1M_vol', '3M_vol']],
                                         left_on=['Ticker', 'year', 'month'], right_on=['TICKER', 'year', 'month'],
                                         how='left')
            # fundamentals and betas
            self.fundamentals = pd.merge(self.fundamentals,
                                         self.betas[['TICKER', 'year', 'month', 'b_mkt', 'b_smb', 'b_hml', 'b_umd']],
                                         left_on=['Ticker', 'year', 'month'], right_on=['TICKER', 'year', 'month'],
                                         how='left')
            # fundamentals and industry
            self.fundamentals = pd.merge(self.fundamentals, self.industry_class[['Symbol', 'GICS Sector']],
                                         left_on=['Ticker'], right_on=['Symbol'],
                                         how='inner')
            self.fundamentals.rename(columns={"GICS Sector": "Industry"}, inplace=True)
            self.fundamentals.to_csv('processed_fundamentals.csv')
        else:
            self.fundamentals = pd.read_csv('processed_fundamentals.csv')

        # self.fundamentals = self.fundamentals.groupby('Ticker', as_index=False).fillna(method='backfill')
        # self.fundamentals = self.fundamentals.groupby('Ticker', as_index=False).fillna(method='ffill')

    def get_factors_df(self):
        self.price_data['PRC'] = self.price_data['PRC'].apply(
            lambda x: -x if x < 0 else x)  # handling negative values in the column, negativ values show bid/ask average
        self.price_data['ADJPRC'] = self.price_data['PRC'] / self.price_data[
            'CFACPR']  # see https://wrds-www.wharton.upenn.edu/pages/support/data-overview/wrds-overview-crsp-us-stock-database/
        self.price_data['ADJSHRS'] = self.price_data['SHROUT'] * self.price_data[
            'CFACSHR']  # see https://wrds-www.wharton.upenn.edu/pages/support/data-overview/wrds-overview-crsp-us-stock-database/

        self.price_data['liquidity'] = self.price_data['VOL'] / self.price_data['ADJSHRS']
        self.fundamentals['debt_cov'] = 1 / self.fundamentals['debt_ebitda']

        self.price_data['date'] = pd.to_datetime(self.price_data['date'], format='%Y%m%d')
        self.price_data['month'] = self.price_data['date'].dt.month
        self.price_data['year'] = self.price_data['date'].dt.year

        self.fundamentals = pd.merge(self.fundamentals, self.price_data[['TICKER', 'year', 'month', 'liquidity']],
                                     left_on=['Ticker', 'year', 'month'], right_on=['TICKER', 'year', 'month'],
                                     how='left')
        return self.fundamentals


class PriceData:
    global ticker_list

    """
    Class to download and transform stock and factor data
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def download_data(self, filepath):
        return pd.read_csv(self.filepath)

    def calc_monthly_price(self, filepath,shift=-1):
        global ticker_list
        price_df = self.download_data(filepath)
        price_df = price_df[['TICKER', 'date', 'PRC']]
        ticker_list = price_df['TICKER'].unique()
        price_df['date'] = pd.to_datetime(price_df['date'], format='%Y-%m-%d')

        mktcap_df = pd.read_csv('price_data.csv')
        mktcap_df['ADJSHRS'] = (mktcap_df['SHROUT'] * mktcap_df['CFACSHR'])
        mktcap_df['date'] = pd.to_datetime(mktcap_df['date'], format='%Y%m%d')
        for tick in ticker_list:
            mktcap_df.loc[mktcap_df['TICKER'] == str(tick), 'ADJSHRS'] = mktcap_df[
                mktcap_df['TICKER'] == str(tick)].ADJSHRS.ffill(axis=0)
            mktcap_df.loc[mktcap_df['TICKER'] == str(tick), 'ADJSHRS'] = mktcap_df[
                mktcap_df['TICKER'] == str(tick)].ADJSHRS.bfill(axis=0)
        price_df = pd.merge(price_df, mktcap_df[['date', 'TICKER', 'ADJSHRS']], how='left', on=['date', 'TICKER'])
        price_df = price_df[['TICKER', 'date', 'PRC', 'ADJSHRS']]
        price_df['ADJSHRS'] = price_df.groupby('TICKER')['ADJSHRS'].transform(lambda v: v.ffill())
        price_df['ADJSHRS'] = price_df.groupby('TICKER')['ADJSHRS'].transform(lambda v: v.bfill())

        price_df['ret'] = price_df.groupby(['TICKER'], as_index=False).PRC.pct_change()
        for tick in ticker_list:
            price_df.loc[price_df['TICKER'] == str(tick), 'ret'] = price_df[price_df['TICKER'] == str(tick)].ret.shift(
                periods=shift)
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
        price_df.loc[
            ((price_df.ret > price_df.first_quantile) & (price_df.ret <= price_df.second_quantile)), 'five_bucket'] = -1
        price_df.loc[
            ((price_df.ret > price_df.second_quantile) & (price_df.ret <= price_df.third_quantile)), 'five_bucket'] = 0
        price_df.loc[
            ((price_df.ret > price_df.third_quantile) & (price_df.ret <= price_df.fourth_quantile)), 'five_bucket'] = 1
        price_df.loc[price_df.ret > price_df.fourth_quantile, 'five_bucket'] = 2

        med_ret_df = price_df.groupby(['date'], as_index=False).median()
        med_ret_df.rename(columns={"ret": 'med_ret'}, inplace=True)
        price_df = pd.merge(price_df, med_ret_df[['date', 'med_ret']], on='date', how='inner')
        price_df['two_bucket'] = 0
        price_df.loc[price_df.ret >= price_df.med_ret, 'two_bucket'] = 1
        price_df.loc[price_df.ret < price_df.med_ret, 'two_bucket'] = -1
        price_df.reset_index(inplace=True, drop=True)
        price_df['3M_mom'] = price_df.groupby(['TICKER'], as_index=False).ret.rolling(3,
                                                                                      min_periods=3).sum().reset_index(
            0, drop=True)
        price_df['12M_mom'] = price_df.groupby(['TICKER'], as_index=False).ret.rolling(12,
                                                                                       min_periods=12).sum().reset_index(
            0, drop=True)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df['month'] = price_df['date'].dt.month
        price_df['year'] = price_df['date'].dt.year

        price_df['lagged_returns1'] = price_df.groupby(['TICKER'], as_index=False).PRC.pct_change().shift(periods=1)
        price_df['lagged_returns2'] = price_df.groupby(['TICKER'], as_index=False).PRC.pct_change().shift(periods=2)
        print(price_df.head())

        return price_df


class Training:

    def __init__(self, data):
        self.data = data

    def get_cleaned_date(self, startDate, trainWindow, testWindow, bucket='two_bucket', interpolation='linear'):
        path_train = "./TrainData/trainData_" + str(startDate.date()) + "_" + str(trainWindow) + "_" + str(
            testWindow) + "_" + interpolation + ".csv"
        path_test = "./TestData/testData_" + str(startDate.date()) + "_" + str(trainWindow) + "_" + str(
            testWindow) + "_" + interpolation + ".csv"

        if not os.path.exists("./TrainData"):
            os.mkdir("./TrainData")

        if not os.path.exists("./TestData"):
            os.mkdir("./TestData")

        if os.path.exists(path_train) & os.path.exists(path_test):
            train_data = pd.read_csv(path_train).drop('Unnamed: 0', axis=1)
            test_data = pd.read_csv(path_test).drop('Unnamed: 0', axis=1)
            return train_data, test_data
        else:

            data_processed = self.data[(self.data['public_date'] >= startDate) & (
                    self.data['public_date'] < (startDate + pd.DateOffset(months=(trainWindow + testWindow))))]

            # linear interpolation
            if interpolation == 'linear':
                data_processed = data_processed.groupby('Ticker', as_index=False).apply(
                    lambda group: group.interpolate(method='linear'))

            # updating NA by moving in line with industry average
            if interpolation == 'trend':
                cols_update = ['bm', 'pe_exi', 'pe_op_dil', 'evm', 'debt_at',
                               'de_ratio', 'liquidity', 'roe', 'roa', 'roce', 'DIVYIELD', 'dpr', 'intcov_ratio',
                               'debt_ebitda',
                               'rect_turn', 'pay_turn', 'at_turn', 'inv_turn', 'cash_ratio', 'quick_ratio',
                               'curr_ratio',
                               'cash_conversion', '1M_vol', '3M_vol', '3M_mom', '12M_mom', 'b_mkt', 'b_smb',
                               'b_hml', 'b_umd']
                for col in cols_update:
                    # print(col)
                    df2 = pd.DataFrame()
                    df2['Ticker'] = data_processed['Ticker']
                    df2['avg'] = data_processed.groupby(['Industry', 'public_date'])[col].transform(
                        lambda x: x.median())
                    df2['ratio'] = data_processed.groupby(['Industry', 'public_date'])[col].transform(
                        lambda x: x / x.median())
                    df2 = df2.groupby('Ticker', as_index=False).fillna(method='ffill')
                    df2 = df2.groupby('Ticker', as_index=False).fillna(method='backfill')
                    data_processed[col] = df2['avg'] * df2['ratio']

            # regress_cols = ['Ticker', 'public_date', 'month', 'year', 'bm', 'pe_exi', 'pe_op_dil', 'evm', 'debt_at', 'de_ratio', 'liquidity', 'roe', 'roa', 'roce', 'DIVYIELD', 'dpr', 'intcov_ratio', 'debt_ebitda', 'rect_turn', 'pay_turn', 'at_turn', 'inv_turn', 'cash_ratio', 'quick_ratio', 'curr_ratio', 'cash_conversion', '1M_vol', '3M_vol', 'debt_cov', 'Industry', '3M_mom', '12M_mom', 'b_mkt', 'b_smb', 'b_hml', 'b_umd', 'quantile']
            # regress_cols = ['Ticker', 'public_date', 'month', 'year', 'bm', 'pe_exi', 'pe_op_dil', 'evm', 'debt_at', 'de_ratio', 'liquidity', 'roe', 'roa', 'roce', 'dpr', 'intcov_ratio', 'debt_ebitda', 'rect_turn', 'pay_turn', 'at_turn', 'inv_turn', 'cash_ratio', 'quick_ratio', 'curr_ratio', 'cash_conversion', '1M_vol', '3M_vol', 'debt_cov', 'Industry', '3M_mom', '12M_mom', 'b_mkt', 'b_smb', 'b_hml', 'b_umd', 'quantile']
            data_processed.rename(columns={bucket: 'quantile'}, inplace=True)
            regress_cols = ['Ticker', 'public_date', 'month', 'year', 'bm', 'pe_exi', 'pe_op_dil', 'evm', 'debt_at',
                            'de_ratio', 'liquidity', 'roe', 'roa', 'roce', 'DIVYIELD', 'dpr', 'intcov_ratio',
                            'debt_ebitda',
                            'rect_turn', 'pay_turn', 'at_turn', 'inv_turn', 'cash_ratio', 'quick_ratio', 'curr_ratio',
                            'cash_conversion', '1M_vol', '3M_vol', 'Industry', '3M_mom', '12M_mom', 'b_mkt', 'b_smb',
                            'b_hml', 'b_umd', 'quantile']
            data_processed = data_processed[regress_cols]
            data_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
            null_aggr = data_processed.isnull().sum()
            null_aggr_list = null_aggr[null_aggr != 0].index.tolist()
            for col in null_aggr_list:
                a = data_processed.groupby('Ticker').apply(lambda x: x[col].isnull().sum())
                empty_tickers = a[a != 0].index.tolist()
                for ticker in empty_tickers:
                    # print(col, ticker)
                    ind = data_processed[data_processed['Ticker'] == ticker]['Industry'].head(1).values[0]
                    data_processed.loc[data_processed[data_processed['Ticker'] == ticker].index.tolist(), col] = \
                        data_processed[data_processed['Industry'] == ind][col].median()
            train_data = data_processed[(data_processed['public_date'] >= startDate) & (
                    data_processed['public_date'] < (startDate + pd.DateOffset(months=trainWindow)))]
            test_data = data_processed[
                (data_processed['public_date'] >= (startDate + pd.DateOffset(months=trainWindow))) & (
                        data_processed['public_date'] < (
                        startDate + pd.DateOffset(months=(trainWindow + testWindow))))]

            train_data.drop_duplicates(inplace=True)
            test_data.drop_duplicates(inplace=True)
            len_before = train_data.shape[0]

            train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            train_data.dropna(inplace=True)
            test_data.dropna(inplace=True)
            len_after = train_data.shape[0]

            if float((len_before - len_after) / len_before) > 0.25:
                print(
                    'Data dropped significantly. Intial data=' + str(len_before) + ' and data fater=' + str(len_after))

            train_data.to_csv(path_train)
            test_data.to_csv(path_test)

            return train_data, test_data

    def generateTrainTestFiles(self, startDate, endDate, trainWindow, testWindow, bucket, interpolation):
        date = startDate
        while (date <= endDate):
            print(date)
            train_data, test_data = self.get_cleaned_date(date, trainWindow, testWindow, bucket, interpolation)
            # train_data.to_csv("./TrainData/trainData_"+str(date.date())+"_"+str(trainWindow)+"_"+str(testWindow)+"_"+interpolation+".csv")
            # test_data.to_csv("./TestData/testData_"+str(date.date())+"_"+str(trainWindow)+"_"+str(testWindow)+"_"+interpolation+".csv")
            date = date + pd.DateOffset(months=1)

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
        # print(train_X.shape)
        # print(test_X.shape)
        # print(train_y.shape)
        # print(test_y.shape)
        predict_df = test_data.copy()
        classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
        classifier.fit(train_X, train_y)
        predictions = classifier.predict(test_X)
        predict_df['prediction'] = predictions
        probabilities = classifier.predict_proba(test_X)
        prob = []
        for idx, val in enumerate(predictions):
            prob.append(probabilities[idx][val + 2])
        predict_df['predict_prob'] = prob
        cf = confusion_matrix(test_y, predictions)
        op_accuracy = cf[0][0] / sum(cf[0])
        up_accuracy = cf[-1][-1] / sum(cf[-1])
        return predict_df, classifier.feature_importances_, [op_accuracy, up_accuracy]

    def gradientBoost_train(self, train_data, test_data):
        from sklearn.ensemble import GradientBoostingClassifier
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
        # print(train_X.shape)
        # print(test_X.shape)
        # print(train_y.shape)
        # print(test_y.shape)
        predict_df = test_data.copy()
        classifier = GradientBoostingClassifier(max_depth=1, n_estimators=200)
        classifier.fit(train_X, train_y)
        predictions = classifier.predict(test_X)
        predict_df['prediction'] = predictions
        probabilities = classifier.predict_proba(test_X)
        prob = []
        for idx, val in enumerate(predictions):
            prob.append(probabilities[idx][val + 2])
        predict_df['predict_prob'] = prob
        cf = confusion_matrix(test_y, predictions)
        op_accuracy = cf[0][0] / sum(cf[0])
        up_accuracy = cf[-1][-1] / sum(cf[-1])
        return predict_df, classifier.feature_importances_, [op_accuracy, up_accuracy]

    def randomforest_train(self, train_data, test_data):
        from sklearn.ensemble import RandomForestClassifier
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
        # print(train_X.shape)
        # print(test_X.shape)
        # print(train_y.shape)
        # print(test_y.shape)
        predict_df = test_data.copy()
        classifier = RandomForestClassifier(criterion='gini', max_depth=1, n_estimators=200)
        classifier.fit(train_X, train_y)
        predictions = classifier.predict(test_X)
        predict_df['prediction'] = predictions
        probabilities = classifier.predict_proba(test_X)
        prob = []
        for idx, val in enumerate(predictions):
            prob.append(probabilities[idx][val + 2])
        predict_df['predict_prob'] = prob
        cf = confusion_matrix(test_y, predictions)
        op_accuracy = cf[0][0] / sum(cf[0])
        up_accuracy = cf[-1][-1] / sum(cf[-1])
        return predict_df, classifier.feature_importances_, [op_accuracy, up_accuracy]

    def logisticregression_train(self, train_data, test_data):
        from sklearn.linear_model import LogisticRegression
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
        # print(train_X.shape)
        # print(test_X.shape)
        # print(train_y.shape)
        # print(test_y.shape)
        classifier = LogisticRegression()
        classifier.fit(train_X, train_y)
        predictions = classifier.predict(test_X)
        test_data['prediction'] = predictions
        cf = confusion_matrix(test_y, predictions)
        op_accuracy = cf[0][0] / sum(cf[0])
        up_accuracy = cf[-1][-1] / sum(cf[-1])
        return test_data, [op_accuracy, up_accuracy]


class Portfolio:

    def __init__(self, price_data):
        self.price_df = price_data

    def get_transaction_costs(self, prev_stocks, curr_stocks):
        tr_cost_l_rate = 0.0015  # assumed 0.1% transaction cost for either buy or sell
        tr_cost_s_rate = 0.0025
        tr_cost_l = 0
        tr_cost_s = 0
        for x in curr_stocks.index:
            tr_cost_l += abs(curr_stocks.loc[x, 'Long'] - prev_stocks.loc[x, 'Long']) * tr_cost_l_rate
            tr_cost_s += abs(curr_stocks.loc[x, 'Short'] - prev_stocks.loc[x, 'Short']) * tr_cost_s_rate
        return tr_cost_l, tr_cost_s

    def construction(self, test_data, quantiles, prev_stocks, valuation='mean', filterStocks='no_rule', tr_cost=False):
        global ticker_list
        if filterStocks == 'no_rule':
            stocks_long = list(
                test_data[test_data['prediction'].isin([a for a in quantiles if a > 0])]['Ticker'].unique())
            stocks_short = list(
                test_data[test_data['prediction'].isin([a for a in quantiles if a < 0])]['Ticker'].unique())
        elif filterStocks == 'probability':
            all_long = test_data[test_data['prediction'].isin([a for a in quantiles if a > 0])]
            all_long = all_long[all_long['predict_prob'] > all_long['predict_prob'].quantile(0.8)]
            stocks_long = list(all_long['Ticker'].unique())
            all_short = test_data[test_data['prediction'].isin([a for a in quantiles if a < 0])]
            all_short = all_short[all_short['predict_prob'] > all_short['predict_prob'].quantile(0.8)]
            stocks_short = list(all_short['Ticker'].unique())

        curr_stocks = pd.DataFrame(np.zeros(shape=(len(ticker_list), 3)), columns=['Long', 'Short', 'LS'],
                                   index=ticker_list)
        curr_stocks.loc[curr_stocks.index.isin(stocks_long), 'Long'] = 1 / (len(stocks_long))
        curr_stocks.loc[curr_stocks.index.isin(stocks_short), 'Short'] = 1 / (len(stocks_short))
        curr_stocks.loc[curr_stocks.index.isin(stocks_long + stocks_short), 'LS'] = 1 / (
                    len(stocks_long) + len(stocks_short))
        tr_cost_l, tr_cost_s = self.get_transaction_costs(prev_stocks, curr_stocks)

        month, year = test_data['month'].unique()[0], test_data['year'].unique()[0]
        if valuation == 'mean':
            if tr_cost:
                ret_long_only = self.price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (
                    self.price_df['TICKER'].isin(stocks_long))]['ret'].mean() - tr_cost_l
                ret_short_only = -1 * \
                                 self.price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (
                                     self.price_df['TICKER'].isin(stocks_short))]['ret'].mean() - tr_cost_s
            else:
                ret_long_only = self.price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (
                    self.price_df['TICKER'].isin(stocks_long))]['ret'].mean()
                ret_short_only = -1 * \
                                 self.price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (
                                     self.price_df['TICKER'].isin(stocks_short))]['ret'].mean()
            return ret_long_only, ret_short_only, (
                        len(stocks_long) * ret_long_only + len(stocks_short) * ret_short_only) / (
                               len(stocks_short) + len(stocks_long)), stocks_long, stocks_short, curr_stocks
        elif valuation == 'market_cap':
            if tr_cost:
                long_filtered = self.price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (
                    self.price_df['TICKER'].isin(stocks_long))]
                short_filtered = self.price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (
                    self.price_df['TICKER'].isin(stocks_short))]
                ret_long_only = sum(long_filtered['PRC'] * long_filtered['ADJSHRS'] * long_filtered['ret'] / sum(
                    long_filtered['PRC'] * long_filtered['ADJSHRS']))
                ret_long_only -= tr_cost_l
                ret_short_only = -1 * sum(
                    short_filtered['PRC'] * short_filtered['ADJSHRS'] * short_filtered['ret'] / sum(
                        short_filtered['PRC'] * short_filtered['ADJSHRS']))
                ret_short_only -= tr_cost_s
                ret_long_short = (ret_long_only * sum(
                    long_filtered['PRC'] * long_filtered['ADJSHRS']) + ret_short_only * sum(
                    short_filtered['PRC'] * short_filtered['ADJSHRS'])) / (
                                             sum(long_filtered['PRC'] * long_filtered['ADJSHRS']) + sum(
                                         short_filtered['PRC'] * short_filtered['ADJSHRS']))
            else:
                long_filtered = self.price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (
                    self.price_df['TICKER'].isin(stocks_long))]
                short_filtered = self.price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (
                    self.price_df['TICKER'].isin(stocks_short))]
                ret_long_only = sum(long_filtered['PRC'] * long_filtered['ADJSHRS'] * long_filtered['ret'] / sum(
                    long_filtered['PRC'] * long_filtered['ADJSHRS']))
                ret_short_only = -1 * sum(
                    short_filtered['PRC'] * short_filtered['ADJSHRS'] * short_filtered['ret'] / sum(
                        short_filtered['PRC'] * short_filtered['ADJSHRS']))
                ret_long_short = (ret_long_only * sum(
                    long_filtered['PRC'] * long_filtered['ADJSHRS']) + ret_short_only * sum(
                    short_filtered['PRC'] * short_filtered['ADJSHRS'])) / (
                                             sum(long_filtered['PRC'] * long_filtered['ADJSHRS']) + sum(
                                         short_filtered['PRC'] * short_filtered['ADJSHRS']))

            return ret_long_only, ret_short_only, ret_long_short, stocks_long, stocks_short, curr_stocks
        elif valuation == 'dollar_neutral_refreshed':
            # long_filtered = self.price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (self.price_df['TICKER'].isin(stocks_long))]
            # short_filtered = self.price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (self.price_df['TICKER'].isin(stocks_short))]
            # long_value_end = sum((1 + long_filtered['ret']) / long_filtered.shape[0])
            # short_value_end = sum((1 + short_filtered['ret']) / short_filtered.shape[0])
            # long_short_return = long_value_end - short_value_end
            # return long_filtered['ret'].mean(), -1 * short_filtered['ret'].mean(), long_short_return, stocks_long, stocks_short
            if tr_cost:
                ret_long_only = self.price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (
                    self.price_df['TICKER'].isin(stocks_long))]['ret'].mean()
                ret_long_only -= tr_cost_l
                ret_short_only = -1 * self.price_df[
                    (self.price_df['month'] == month) & (self.price_df['year'] == year) & (
                        self.price_df['TICKER'].isin(stocks_short))]['ret'].mean()
                ret_short_only -= tr_cost_s
            else:
                ret_long_only = self.price_df[(self.price_df['month'] == month) & (self.price_df['year'] == year) & (
                    self.price_df['TICKER'].isin(stocks_long))]['ret'].mean()
                ret_short_only = -1 * self.price_df[
                    (self.price_df['month'] == month) & (self.price_df['year'] == year) & (
                        self.price_df['TICKER'].isin(stocks_short))]['ret'].mean()
            return ret_long_only, ret_short_only, 0.5 * ret_long_only + 0.5 * ret_short_only, stocks_long, stocks_short, curr_stocks

    def returns(self, trainObj, startDate, EndDate, trainWindow, testWindow, bucket='five_bucket', quantiles=[-2, 2],
                Algo='AdaBoost', interpolation='linear', valuation='mean', filsterStocks='no_rule', tr_cost=False,
                all_combined=True):
        global ticker_list
        returns_dict = {}
        feature_imp_dict = {}
        op_up_acc_dict = {}
        date = startDate
        prev_stocks = pd.DataFrame(np.zeros(shape=(len(ticker_list), 3)), columns=['Long', 'Short', 'LS'],
                                   index=ticker_list)
        while (date <= EndDate):
            print(date)
            train_data, test_data = trainObj.get_cleaned_date(date, trainWindow, testWindow, bucket, interpolation)
            if Algo == 'Combination':
                test_a, _, _ = trainObj.adaBoost_train(train_data, test_data)
                test_g, _, _ = trainObj.gradientBoost_train(train_data, test_data)
                test_r, _, _ = trainObj.randomforest_train(train_data, test_data)
                tmp = pd.merge(test_a[['Ticker', 'prediction']], test_g[['Ticker', 'prediction']], on=['Ticker'],
                               how='inner')
                tmp = pd.merge(tmp, test_r[['Ticker', 'prediction']], on=['Ticker'], how='inner')
                tmp['All3'] = -10
                tmp['2same'] = -10
                for i in range(0, tmp.shape[0]):
                    if ((tmp.loc[i, 'prediction_x'] == tmp.loc[i, 'prediction_y']) & (
                            tmp.loc[i, 'prediction_x'] == tmp.loc[i, 'prediction'])):
                        tmp.loc[i, 'All3'] = tmp.loc[i, 'prediction_x']
                    if ((tmp.loc[i, 'prediction_x'] == tmp.loc[i, 'prediction_y']) | (
                            tmp.loc[i, 'prediction_x'] == tmp.loc[i, 'prediction']) | (
                            tmp.loc[i, 'prediction'] == tmp.loc[i, 'prediction_y'])):
                        if (tmp.loc[i, 'prediction_x'] == tmp.loc[i, 'prediction_y']):
                            tmp.loc[i, '2same'] = tmp.loc[i, 'prediction_x']
                        else:
                            tmp.loc[i, '2same'] = tmp.loc[i, 'prediction']
                tmp.drop(columns=['prediction_y', 'prediction_x', 'prediction'], inplace=True)
                tmp.rename(columns={'All3': 'prediction'} if all_combined else {'2same': 'prediction'}, inplace=True)
                tmp['month'] = test_a['month']
                tmp['year'] = test_a['year']
                long_only_return, short_only_return, long_short_return, long, short = self.construction(tmp,
                                                                                                        quantiles,
                                                                                                        prev_stocks,
                                                                                                        valuation,
                                                                                                        filsterStocks,
                                                                                                        tr_cost
                                                                                                        )
                print(long)
                dt = test_data['public_date'].unique()[0]
                # print(long_only_return, short_only_return, long_short_return)
                for x in long:
                    curr_stocks[x] = 1
                for x in short:
                    curr_stocks[x] = -1
                # tr_cost = self.get_transaction_costs(prev_stocks, curr_stocks)
                returns_dict[dt] = [long_only_return, short_only_return, long_short_return, len(long),
                                    len(short)]
                prev_stocks = curr_stocks
                # set_trace()
                date = date + pd.DateOffset(months=1)

            if Algo == 'AdaBoost':
                test_with_prediction, imp_features, accu = trainObj.adaBoost_train(train_data, test_data)
                long_only_return, short_only_return, long_short_return, long, short, curr_stocks = self.construction(
                    test_with_prediction,
                    quantiles, prev_stocks, valuation, filsterStocks, tr_cost)
                dt = test_data['public_date'].unique()[0]
                # print(long_only_return, short_only_return, long_short_return)
                returns_dict[dt] = [long_only_return, short_only_return, long_short_return, len(long),
                                    len(short)]
                feature_imp_dict[dt] = imp_features
                op_up_acc_dict[dt] = accu
                prev_stocks = curr_stocks.copy()
                # set_trace()
                date = date + pd.DateOffset(months=1)

            if Algo == 'GradientBoost':
                test_with_prediction, imp_features, accu = trainObj.gradientBoost_train(train_data, test_data)
                long_only_return, short_only_return, long_short_return, long, short, curr_stocks = self.construction(
                    test_with_prediction,
                    quantiles, prev_stocks, valuation, filsterStocks, tr_cost)
                dt = test_data['public_date'].unique()[0]
                # print(long_only_return, short_only_return, long_short_return)
                returns_dict[dt] = [long_only_return, short_only_return, long_short_return, len(long), len(short)]
                feature_imp_dict[dt] = imp_features
                op_up_acc_dict[dt] = accu
                date = date + pd.DateOffset(months=1)

            if Algo == 'RandomForest':
                test_with_prediction, imp_features, accu = trainObj.randomforest_train(train_data, test_data)
                long_only_return, short_only_return, long_short_return, long, short, curr_stocks = self.construction(
                    test_with_prediction,
                    quantiles, prev_stocks, valuation, filsterStocks, tr_cost)
                dt = test_data['public_date'].unique()[0]
                # print(long_only_return, short_only_return, long_short_return)
                returns_dict[dt] = [long_only_return, short_only_return, long_short_return, len(long), len(short)]
                feature_imp_dict[dt] = imp_features
                op_up_acc_dict[dt] = accu
                date = date + pd.DateOffset(months=1)

            if Algo == 'LogisticRegression':
                test_with_prediction, accu = trainObj.logisticregression_train(train_data, test_data)
                long_only_return, short_only_return, long_short_return, _, _ = self.construction(test_with_prediction,
                                                                                                 quantiles, prev_stocks,
                                                                                                 valuation,
                                                                                                 filsterStocks, tr_cost)
                dt = test_data['public_date'].unique()[0]
                # print(long_only_return, short_only_return, long_short_return)
                returns_dict[dt] = [long_only_return, short_only_return, long_short_return]
                op_up_acc_dict[dt] = accu
                date = date + pd.DateOffset(months=1)

        return pd.DataFrame.from_dict(returns_dict, orient='index',
                                      columns=['Long_Only', 'Short_Only', 'Long_Short', 'Num Long', 'Num Short', ]), \
               pd.DataFrame.from_dict(feature_imp_dict, orient='index'), \
               pd.DataFrame.from_dict(op_up_acc_dict, orient='index')


class Utils:
    def get_cumulative_returns_aqr(self, aqr_fp, rf_fp, start_dt=[], end_dt=[]):
        df = pd.read_csv(aqr_fp)
        rf = pd.read_csv(rf_fp)
        df['Date'] = pd.to_datetime(df['Date'])
        rf['Date'] = pd.to_datetime(rf['Date'], dayfirst=True)
        if start_dt:  # specify either both start and end or none
            df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]
            rf = rf[(rf['Date'] >= start_dt) & (rf['Date'] <= end_dt)]

        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year

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

    def benchmark_portfolio(self, me_ind=1, ia_ind=1, roe_ind=1, year=1967, month=1):
        bench_df = pd.read_csv('/Users/vikrantdhall/Documents/MFE/AFP/AFP/benportf_me_ia_roe_monthly_2019.csv')
        bench_df = bench_df[(bench_df.year >= year) | (bench_df.month >= month)]
        bench_df = bench_df[
            (bench_df.rank_ME == me_ind) & (bench_df.rank_IA == ia_ind) & (bench_df.rank_ROE == roe_ind)]
        bench_df['cum_ret'] = (1 + bench_df.ret_vw / 100).cumprod()
        bench_df['day'] = 1
        bench_df['date'] = pd.to_datetime(bench_df[['year', 'month', 'day']])

        return bench_df


class Plot_results:
    def __init__(self):
        self.u = Utils()

    def plot_benchmark_aqr(self):
        cum_ret = self.u.get_cumulative_returns_aqr('AQR_Val_Mom.csv', 'Treasury_1M.csv')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        xticks = pd.to_datetime(cum_ret['Date_x'])
        ax.plot(xticks, cum_ret['Cum_Val'], color='blue', label='Value')
        ax.plot(xticks, cum_ret['Cum_Mom'], color='red', label='Mom')
        ax.legend()
        ax.set_title('Aqr Mom Factor Returns', fontsize=18)
        fig.tight_layout()

    def plot_our_results(self, returns):
        returns = self.u.get_cumulative_returns_ours(returns)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        xticks = pd.to_datetime(returns.index.values)
        ax.plot(xticks, (returns['Cum_L']), color='green', label='Long Only')
        ax.plot(xticks, (returns['Cum_S']), color='cyan', label='Short Only')
        ax.plot(xticks, (returns['Cum_LS']), color='magenta', label='Long_Short')
        ax.legend()
        ax.set_title('Our returns', fontsize=18)
        fig.tight_layout()

    def plot_bench_results(self):
        result = self.u.benchmark_portfolio(me_ind=1, ia_ind=1, roe_ind=1, year=1967, month=1)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        xticks = pd.to_datetime(result.date)
        ax.plot(xticks, result.cum_ret, color='green', label='benchmark_portfolio')

    def plot_combined(self, returns, start_dt, end_dt):
        cum_ret_benchmark = self.u.get_cumulative_returns_aqr('AQR_Val_Mom.csv', 'Treasury_1M.csv', start_dt, end_dt)
        cum_ret_benchmark['date'] = pd.to_datetime(cum_ret_benchmark['Date_x'])
        cum_ret_benchmark['month'] = cum_ret_benchmark['date'].dt.month
        cum_ret_benchmark['year'] = cum_ret_benchmark['date'].dt.year
        returns = self.u.get_cumulative_returns_ours(returns)
        returns_t = returns.reset_index()
        returns_t['month'] = returns_t['index'].dt.month
        returns_t['year'] = returns_t['index'].dt.year
        df = pd.merge(cum_ret_benchmark, returns_t, on=['year', 'month'], how='inner')
        set_trace()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        xticks = pd.to_datetime(cum_ret_benchmark['Date_x'])
        ax.plot(xticks, df['Cum_Val'], marker='o', label='AQR Value')
        ax.plot(xticks, df['Cum_Mom'], marker='o', label='AQR Mom')
        ax.plot(xticks, df['Cum_L'], marker='o', label='Long_Only')
        ax.plot(xticks, df['Cum_S'], marker='o', label='Short_Only')
        ax.plot(xticks, df['Cum_LS'], marker='o', label='Long_Short')

        ax.legend()
        ax.set_title('Cumulative Returns', fontsize=18)
        fig.tight_layout()


shift=-4
price_filepath = 'price_data_yahoo.csv'
data = PriceData(price_filepath)
price_df = data.calc_monthly_price(price_filepath,shift)

factors = Factors()
factors.combine_data(0)
f = factors.get_factors_df()

reg_df = pd.merge(f, price_df, left_on=['Ticker', 'year', 'month'], right_on=['TICKER', 'year', 'month'], how='inner')
print(reg_df.shape)
reg_df.drop_duplicates(subset=['Ticker', 'year', 'month'], inplace=True)
print(reg_df.shape)


train = Training(reg_df)
port = Portfolio(price_df)
startDate = pd.to_datetime('20000128', format='%Y%m%d')
endDate = pd.to_datetime('20171128', format='%Y%m%d')
#endDate = pd.to_datetime('20000428',format='%Y%m%d')
train_window = 12  # in months
test_windon = 1  # in months
interpolation = 'linear'
price_buckets = 'five_bucket'
portfolio_buckets = [-2, 2]
algos = ['AdaBoost','GradientBoost','RandomForest']
valuation = 'mean'
filterStocks = 'no_rule'
tr_cost = True
# algos = algos[1:]
if not os.path.exists("./Results"):
    os.mkdir("./Results")

for algo in algos:
    print(algo)
    returns_df, feature_imp, accuracy_df = port.returns(train, startDate, endDate, train_window, test_windon,
                                                        price_buckets, portfolio_buckets, algo, interpolation,valuation,filterStocks,tr_cost)
    returns_df.to_csv('./Results/' + algo + '_'+valuation+'_'+filterStocks+'_' +str(tr_cost) + '_' + interpolation + '_returns_lagged.csv')
    feature_imp.to_csv('./Results/' + algo + '_'+valuation+'_'+filterStocks+'_' +str(tr_cost) + '_' + interpolation + '_feature_importance_lagged.csv')
    accuracy_df.to_csv('./Results/' + algo + '_'+valuation+'_'+filterStocks+'_'+str(tr_cost) + '_'  + interpolation + '_accuracy_lagged.csv')
    # print(returns_df)
