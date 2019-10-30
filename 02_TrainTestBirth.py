import os
import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta

os.chdir(r'C:\Users\Ricardo\Desktop\PITAO\[PROJ] CartaoContinente')

def loadData(data_source = '01_merged.pickle'):

    """
    Loads the aggregated data.
    Usefull function to avoid repetition.
    """

    with open(data_source, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data

def summarizeClients(data, churn_onset, time_window = 90):

    """
    Summarizes the features of the clients up to a certain point in time (churn_onset).
    It picks the information from the "data" parameter.
    One of the features is a boolean which shows if the client churned in the next "time_window" (in days).
    Usefull to make the future train and test datasets.
    """

    time_window = timedelta(time_window)
    
    activeClients = np.array(data.loc[(churn_onset > data['TRANSACTION_TIME_KEY']) & (data['TRANSACTION_TIME_KEY'] > (churn_onset - time_window)), 'CUSTOMER_ACCOUNT_MASK'].unique())
    final = pd.DataFrame(activeClients, columns = ['AC']).sort_values(by = ['AC'], ascending = True)
    
    data_work = final.merge(data[data.TRANSACTION_TIME_KEY <= churn_onset], left_on = 'AC', right_on = 'CUSTOMER_ACCOUNT_MASK', how = 'left')

    ### tDiff // meanDiff // stdDiff --- Days between each purchase // Mean of that array for each client // StandardDeviation of tDiff
    data_work['lastPurchaseTime'] = data_work.sort_values(by = ['TRANSACTION_TIME_KEY'], ascending = True).groupby('CUSTOMER_ACCOUNT_MASK')['TRANSACTION_TIME_KEY'].shift(1)
    data_work['tDiff'] = data_work['TRANSACTION_TIME_KEY'] - data_work['lastPurchaseTime']
    data_work['tDiff'] = data_work['tDiff'].apply(lambda x: x.days)
    data_work['meanDiff'] = data_work.groupby('CUSTOMER_ACCOUNT_MASK')['tDiff'].transform('mean')
    data_work['stdDiff'] = data_work.groupby('CUSTOMER_ACCOUNT_MASK')['tDiff'].transform('std')

    ### countFreq --- Number of purchases 
    
    data_work['countFreq'] = data_work.groupby('CUSTOMER_ACCOUNT_MASK')['TRANSACTION_TIME_KEY'].transform('size')

    ### totalAmount //  meanAmount // minAmount // maxAmount
    data_work['totalAmount'] = data_work.groupby('CUSTOMER_ACCOUNT_MASK')['GROSS_SLS_AMT'].transform('sum')
    data_work['meanAmount'] = data_work.groupby('CUSTOMER_ACCOUNT_MASK')['GROSS_SLS_AMT'].transform('mean')
    data_work['minAmount'] = data_work.groupby('CUSTOMER_ACCOUNT_MASK')['GROSS_SLS_AMT'].transform('min')
    data_work['maxAmount'] = data_work.groupby('CUSTOMER_ACCOUNT_MASK')['GROSS_SLS_AMT'].transform('max')

    ### timeSinceLastP // lifetime
    
    data_work['timeSinceLastP'] = churn_onset - data_work['TRANSACTION_TIME_KEY']
    data_work['firstPTime'] = data_work.sort_values(by = 'TRANSACTION_TIME_KEY', ascending = True).groupby('CUSTOMER_ACCOUNT_MASK')['TRANSACTION_TIME_KEY'].transform('first')
    data_work['lifetime'] = churn_onset - data_work['firstPTime']

    ### sumCountry 
    
    data_work['sumCountry'] = data_work.groupby('CUSTOMER_ACCOUNT_MASK')['COUNTRY'].transform(lambda x: len(x.unique()))

    ### highValueC --- If a customer does only buy on the most "expensive" sectors (Electronics, Cars, Travelling, Health)
    
    highValueSectorList = ['Eletrónica', 'Automóvel', 'Informática e Online', 'Saúde', 'Viagens']
    data_work['highValueSector'] = data_work.HIERARCHY_PARENT_DSC.apply(lambda x: True if x in highValueSectorList else False)
    data_work['highValueC'] = data_work.groupby('CUSTOMER_ACCOUNT_MASK')['highValueSector'].transform('min')

    ### lastPSector // lastPAmount // lastPRatio --- In which sector was made the last purchase // How much was spent // Comparing to the mean spent (includes the last P)

    lastP = data_work.sort_values(by = 'TRANSACTION_TIME_KEY', ascending = False).groupby('CUSTOMER_ACCOUNT_MASK')[['GROSS_SLS_AMT', 'HIERARCHY_PARENT_DSC', 'meanAmount']].first()
    lastP = lastP.rename({'GROSS_SLS_AMT': 'lastPAmount', 'HIERARCHY_PARENT_DSC': 'lastPSector'}, axis = 'columns')
    lastP['lastPRatio'] = lastP['lastPAmount']/lastP['meanAmount']
    final = final.merge(lastP.drop('meanAmount', axis = 1), left_on = 'AC', right_on = 'CUSTOMER_ACCOUNT_MASK', how = 'left')

    ### prefBA --- Shows the prefered sector of the client

    # data_work['prefBA'] = data_work.merge(data_work.groupby('CUSTOMER_ACCOUNT_MASK').HIERARCHY_PARENT_DSC.apply(lambda x: x.mode()), on = 'CUSTOMER_ACCOUNT_MASK', how ='left')

    ### churned --- Variable to indicate if the consumer didn't buy in the following time window

    data['outOfChurnTime'] = data['TRANSACTION_TIME_KEY'].apply(lambda x: False if (x > churn_onset) & (x < churn_onset + time_window) else True)
    data['churned'] = data.groupby('CUSTOMER_ACCOUNT_MASK')['outOfChurnTime'].transform('min')
    data_work = data_work.merge(data.groupby('CUSTOMER_ACCOUNT_MASK')['churned'].first(), on = 'CUSTOMER_ACCOUNT_MASK', how = 'left')

    ### Just before cleaning, we need to remove NaNs from the set
    ### for tDiff / meanDiff / stdDiff we will replace with an high value
    
    data_work.loc[:, ['tDiff', 'meanDiff', 'stdDiff']] = data_work[['tDiff', 'meanDiff', 'stdDiff']].fillna(value = 700)
    data_work['AGE'].fillna(value=data.AGE.median(), inplace=True)
    
    ### Clean to last line and merge into final

    columns_to_merge = ['AGE', 'GENDER_M', 'GENDER_F', 'REGION', 'meanDiff', 'tDiff', 'stdDiff', 'countFreq', 'totalAmount', 'meanAmount', 'minAmount', 'maxAmount', 'timeSinceLastP', 'lifetime', 'sumCountry', 'highValueC', 'churned']
    data_work = data_work.sort_values(by = 'TRANSACTION_TIME_KEY', ascending = False).groupby('CUSTOMER_ACCOUNT_MASK')[columns_to_merge].first()
    final = final.merge(data_work, left_on = 'AC', right_on = 'CUSTOMER_ACCOUNT_MASK', how = 'left')
    
    final['timeSinceLastP'] = final['timeSinceLastP'].apply(lambda x: x.days)
    final['lifetime'] = final['lifetime'].apply(lambda x: x.days)
    final.loc[final.countFreq == 1, 'lastPRatio'] = 50 ### lastPRatio useless on people with 1 Prchs. High value helps the model to set them aside
    
    return final.set_index('AC')

train_set = summarizeClients(loadData(), datetime(2018, 6, 30), time_window = 90)
test_set = summarizeClients(loadData(), datetime(2018, 9, 30), time_window = 90)

with open('train_test_set.pickle', 'wb') as f:
    pickle.dump([train_set, test_set], f)
    f.close()_