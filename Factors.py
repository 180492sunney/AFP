import pandas as pd
import numpy as np
class Factors:
    def __init__(self,fundamental_filepath,price_filepath):
        self.fundamentals = pd.read_csv(fundamental_filepath)
        self.price_data = pd.read_csv(price_filepath)

    def get_factors_df(self):
        self.price_data['PRC'] = self.price_data['PRC'].apply(lambda x: -x if x<0 else x)#handling negative values in the column, negativ values show bid/ask average
        self.price_data['ADJPRC'] = self.price_data['PRC']/self.price_data['CFACPR']# see https://wrds-www.wharton.upenn.edu/pages/support/data-overview/wrds-overview-crsp-us-stock-database/
        self.price_data['ADJSHRS'] = self.price_data['SHROUT']*self.price_data['CFACSHR'] # see https://wrds-www.wharton.upenn.edu/pages/support/data-overview/wrds-overview-crsp-us-stock-database/
        b_m = self.fundamentals['bm']#book-to-market
        p_e = self.fundamentals['pe_exi'] #diluted price to earnings excluding extraordinary items
        p_oe = self.fundamentals['pe_op_dil'] #diluted price to operating earnings
        evm = self.fundamentals['evm'] #enterprise value/ebitda
        lev = self.fundamentals['debt_at'].fillna(method='backfill',inplace=True) #leverage, total debt/assets
        lev_e = self.fundamentals['de_ratio'].fillna(method='backfill',inplace=True) #leverage, total debt/equity
        lev_hybrid = self.fundamentals['de_ratio']+self.fundamentals['debt_at'] #leverage, we are using sum of total debt/equity + total debt/assets
        size = np.log(self.price_data['ADJSHRS']*(100/21)*self.price_data['ADJPRC'])
        liquidity = self.price_data['VOL']/self.price_data['ADJSHRS']
        roe = self.fundamentals['roe']
        roa = self.fundamentals['roa']
        roce = self.fundamentals['roce']
        div_yield = self.fundamentals['divyield']
        dpr = self.fundamentals['dpr']
        int_cov = self.fundamentals['intcov_ratio']
        debt_cov = 1/self.fundamentals['debt_ebitda']
        recv_trnvr = self.fundamentals['rect_turn']
        pay_trnvr = self.fundamentals['pay_turn']
        asset_trnvr = self.fundamentals['at_turn']
        inventory_trnvr = self.fundamentals['inv_turn']
        cash_ratio = self.fundamentals['cash_ratio']
        curr_ratio = self.fundamentals['curr_ratio']
        quick_ratio = self.fundamentals['quick_ratio']
        cash_conversion_cycle = self.fundamentals['cash_conversion']
