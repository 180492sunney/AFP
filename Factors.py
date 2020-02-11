import pandas as pd
import numpy as np
from pdb import set_trace

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
        price_df.drop_duplicates(keep = 'first', inplace = True)
        price_df['ret'] = price_df.groupby(['TICKER'], as_index=False).PRC.pct_change()
        price_df.dropna(how='any', axis=0, inplace=True)

        # check for later if median is negative in a month
        med_ret_df = price_df.groupby(['date'], as_index=False).median()
        med_ret_df.rename(columns={"ret": 'med_ret'}, inplace=True)
        med_ret_df.drop(['PRC'], axis=1, inplace=True)
        price_df = pd.merge(price_df, med_ret_df, on='date', how='inner')
        price_df['quantile'] = 0
        price_df.loc[price_df.ret >= price_df.med_ret, 'quantile'] = 1
        price_df.loc[price_df.ret < price_df.med_ret, 'quantile'] = -1
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
        return price_df


class Training:

    def __init__(self, data, window):
        self.data = data
        self.window = window  # in months

    def get_cleaned_date(self, startDate):
        data_processed = self.data[(self.data['public_date'] >= startDate) & (
                    self.data['public_date'] < (startDate + pd.DateOffset(months=self.window)))]
        data_processed = data_processed.groupby('Ticker', as_index=False).fillna(method='backfill')
        data_processed = data_processed.groupby('Ticker', as_index=False).fillna(method='ffill')
        # regress_cols = ['Ticker', 'public_date', 'bm', 'pe_exi', 'pe_op_dil', 'evm', 'debt_at', 'de_ratio', 'liquidity', 'roe', 'roa', 'roce', 'DIVYIELD', 'dpr', 'intcov_ratio', 'debt_ebitda', 'rect_turn', 'pay_turn', 'at_turn', 'inv_turn', 'cash_ratio', 'quick_ratio', 'curr_ratio', 'cash_conversion', '1M_vol', '3M_vol', 'debt_cov', 'Industry', '3M_mom', '12M_mom', 'b_mkt', 'b_smb', 'b_hml', 'b_umd']
        regress_cols = ['Ticker', 'public_date', 'bm', 'pe_exi', 'pe_op_dil', 'evm', 'debt_at', 'de_ratio', 'liquidity',
                        'roe', 'roa', 'roce', 'dpr', 'intcov_ratio', 'debt_ebitda', 'rect_turn', 'pay_turn', 'at_turn',
                        'inv_turn', 'cash_ratio', 'quick_ratio', 'curr_ratio', 'cash_conversion', '1M_vol', '3M_vol',
                        'debt_cov', 'Industry', '3M_mom', '12M_mom', 'b_mkt', 'b_smb', 'b_hml', 'b_umd']
        data_processed = data_processed[regress_cols]
        null_aggr = data_processed.isnull().sum()
        null_aggr_list = null_aggr[null_aggr != 0].index.tolist()
        for col in null_aggr_list:
            a = data_processed.groupby('Ticker').apply(lambda x: x[col].isnull().sum())
            empty_tickers = a[a != 0].index.tolist()
            for ticker in empty_tickers:
                # print(col, ticker)
                ind = data_processed[data_processed['Ticker'] == ticker]['Industry'].head(1).values[0]
                data_processed.loc[data_processed[data_processed['Ticker'] == ticker].index.tolist(), col] = \
                data_processed[data_processed['Industry'] == ind][col].mean()
        return data_processed


price_filepath = 'price_data.csv'
data = PriceData(price_filepath)
price_df = data.calc_monthly_price(price_filepath)

factors = Factors()
factors.combine_data()
f = factors.get_factors_df()

reg_df = pd.merge(f,price_df,left_on =['Ticker','year','month'],right_on=['TICKER','year','month'],how='inner')
print(reg_df.shape)

train = Training(reg_df, 12)
proc_data = train.get_cleaned_date(pd.to_datetime('2014-02-28'))
print(proc_data.shape)