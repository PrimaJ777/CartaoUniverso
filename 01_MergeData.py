import os
import pandas as pd
import pickle
import numpy as np

os.chdir(r'C:\Users\Ricardo\Desktop\PITAO\[PROJ] CartaoContinente')

dim_customer = pd.read_csv(r'raw_data\dim_customer.csv', encoding = 'cp1252')
dim_hierarchy_agreg = pd.read_csv(r'raw_data\dim_hierarchy_agreg.csv', encoding='cp1252')
dim_merchant_hierarchy = pd.read_csv(r'raw_data\dim_merchant_hierarchy.csv', encoding='cp1252')
dim_merchant = pd.read_csv(r'raw_data\dim_merchant.csv', encoding='cp1252')
dim_segment_baby = pd.read_csv(r'raw_data\dim_segment_baby.csv', encoding='cp1252')
dim_segment_junior = pd.read_csv(r'raw_data\dim_segment_junior.csv', encoding='cp1252')
dim_segment_lifestage = pd.read_csv(r'raw_data\dim_segment_lifestage.csv', encoding='cp1252')
dim_time = pd.read_csv(r'raw_data\dim_time.csv', encoding='cp1252')
dim_transactions = pd.read_csv(r'raw_data\dim_transactions.csv', encoding='cp1252')

dim_total = dim_transactions.merge(dim_customer, on = 'CUSTOMER_CARD_KEY_MASK', how = 'left')
dim_total = dim_total.merge(dim_merchant_hierarchy, on = 'MERCH_KEY', how = 'left')
dim_total = dim_total.merge(dim_hierarchy_agreg, left_on = 'hierarchy_cd', right_on = 'HIERARCHY_CD', how = 'left')
dim_total = dim_total.merge(dim_time, left_on = 'TRANSACTION_TIME_KEY', right_on = 'TIME_KEY', how = 'left')
dim_total = dim_total.merge(dim_merchant, on = 'MERCH_KEY', how = 'left')
dim_total = dim_total.merge(dim_segment_junior, on = 'SEGM_JUNIOR_CD', how = 'left')
dim_total = dim_total.merge(dim_segment_baby, on='SEGM_BABY_CD', how = 'left')
dim_total = dim_total.merge(dim_segment_lifestage, left_on = 'SEG_LIFESTAGE_KEY', right_on = 'segm_lifestage_cd', how = 'left')

dim_total = dim_total[['CUSTOMER_CARD_KEY_MASK', 'CUSTOMER_ACCOUNT_MASK', 'GENDER_M', 'GENDER_F', 'AGE', 'DISTRICT', 'REGION', 'segm_baby_dsc', 'segm_junior_dsc', 'segm_lifestage_dsc', 
'HIERARCHY_PARENT_DSC', 'hierarchy_sub_level_dsc', 'ECOSSISTEMA_CC', 'PARTNER_BRAND', 'YEAR', 'QUARTER', 'FULLDATE', 'MONTH', 'WEEK', 'DAY', 'DAYOFWEEK', 'TRANSACTION_TIME_KEY', 
'TRANSACTION_HOUR_KEY', 'TIME_KEY_x', 'MERCH_DSC_x', 'hierarchy_dsc', 'COUNTRY', 'GROSS_SLS_AMT', 'HIERARCHY_SUB_LEVEL_CD', 'FLAG_ONLINE', 'PAYMENT_SERVICE']]

"""  Type Conversion  """

typeconv_guide = {
    'object': ['CUSTOMER_CARD_KEY_MASK', 'CUSTOMER_ACCOUNT_MASK', 'MERCH_DSC_x', 'hierarchy_dsc', 'hierarchy_sub_level_dsc', 'HIERARCHY_SUB_LEVEL_CD'],
    'bool': ['GENDER_M', 'GENDER_F', 'ECOSSISTEMA_CC', 'FLAG_ONLINE'],
    'Int64': ['AGE', 'YEAR', 'QUARTER', 'MONTH', 'WEEK', 'DAY', 'TIME_KEY_x'],
    'category': ['DISTRICT', 'REGION', 'segm_baby_dsc', 'segm_junior_dsc', 'segm_lifestage_dsc', 'HIERARCHY_PARENT_DSC', 'PARTNER_BRAND', 'DAYOFWEEK', 'COUNTRY', 'PAYMENT_SERVICE'],
    'float64': ['GROSS_SLS_AMT']
}

for datatype in typeconv_guide:
    for _ in typeconv_guide[datatype]:
        dim_total[_] = dim_total[_].astype(datatype)

dim_total['TRANSACTION_TIME_KEY'] = pd.to_datetime(dim_total['TRANSACTION_TIME_KEY'], format='%Y%m%d')
dim_total['REGION'] = dim_total['REGION'].cat.add_categories('Unknown').fillna(value = 'Unknown')
dim_total.loc[dim_total.AGE > 108, 'AGE'] = np.nan
dim_total['HIERARCHY_PARENT_DSC'] = dim_total['HIERARCHY_PARENT_DSC'].cat.add_categories('Unknown').fillna(value = 'Unknown') ## These NA's happen because there's hierarchy_cd = NaN on dim_merchant_hierarchy


with open('01_merged.pickle', 'wb') as f:
    pickle.dump(dim_total, f)