# -*- coding: utf-8 -*-
"""
Preprocessing script.
"""

import numpy as np
import pandas as pd


df = pd.read_csv('data/train.csv.gz')
testdf = pd.read_csv('data/test.csv.gz')

################################################ 
# Construct the dataset required for model 1
################################################

# This involves:
# 1st, convert the 'FinelineNumber' values of into individual columns
# 2nd, collapse the rows based on 'VisitNumber'

# Convert the 'FinelineNumber' values of into individual columns
# Note : the train and test have different number of 'FinelineNumber'
df['FinelineNumber'].fillna(999999, inplace=True)
testdf['FinelineNumber'].fillna(999999, inplace=True)
flns = sorted(list(set(pd.unique(testdf.FinelineNumber.ravel()).tolist())))
flns.extend(sorted(pd.unique(df.FinelineNumber.ravel()).tolist()))
flns = sorted(list(set(flns)))
flns = [int(x) for x in flns]
# Approach 1 : get_dummies method -- VERY SLOW and consumes 99% of memory
#dft = pd.concat([df, pd.get_dummies(df['FinelineNumber'], 
#                                    prefix = 'Prod').astype(np.int8)], axis=1)
# Approach 2 : 
for fln in flns:
    colname = ('Prod%s' % str(fln).zfill(4)) if fln<10000 else 'Prod'+str(fln)
    df[colname] = df.FinelineNumber.apply(lambda x: 1 if x == fln else 0)\
                    .astype(np.int8)
#df.to_csv('data/train-data-wide-stg-1-model-1.txt',sep='\t', index=False)
# 647054 x 5205

# Debug : Print the first 10 columns
print list(df.columns.values)[:10]
# Debug : Print the first 10 rows and first 7 columns
print df.ix[:10, :7]

# Get rid of columns we don't need - before we collapse rows on 'VisitNumber'
df.drop(['Weekday','Upc','DepartmentDescription','FinelineNumber'], 
        axis=1, inplace=True)
# 647054 x 5357

# Collapse the rows based on 'VisitNumber'
# Approach 1 : use groupby -- VERY SLOW
#df.reset_index().groupby("VisitNumber").sum()
# Approach 2 : make a copy of the 'df' 
# retain only 'VisitNumber' and the 'ProdXXXX' columns
l = list(df.columns.values)
for c in ['TripType', 'ScanCount']:
    l.remove(c)
df2 = df[l]
print df2.ix[:10, :7]
df2 = df2.groupby('VisitNumber').sum()
df2.reset_index(level=0, inplace=True)
df2.to_csv('data/train-data-wide-stg-2-model-1.txt',sep='\t', index=True)
df3 = df[['VisitNumber','TripType']]
df3 = df3.drop_duplicates(cols=['VisitNumber'], keep='last')

# Construct the dataset required for model 1
model_1_train = pd.merge(df2, df3, on='VisitNumber')
model_1_train.to_csv('data/train-dataset-1.txt',sep='\t', index=False)
print model_1_train.ix[:10, :7]
print model_1_train.ix[:10, 5349:]

######
# Convert the test dataset in the form required for model 1
testdf = testdf[['VisitNumber','FinelineNumber']]

# Convert the 'FinelineNumber' values of into individual columns
for fln in flns:
    colname = ('Prod%s' % str(fln).zfill(4)) if fln<10000 else 'Prod'+str(fln)
    testdf[colname] = testdf.FinelineNumber.\
                      apply(lambda x: 1 if x == fln else 0).astype(np.int8)
testdf.drop(['FinelineNumber'], axis=1, inplace=True)

# Collapse the rows based on 'VisitNumber'
model_1_test = testdf.groupby('VisitNumber').sum()
model_1_test.reset_index(level=0, inplace=True)
model_1_test.to_csv('data/test-dataset-1.txt',sep='\t', index=True)
