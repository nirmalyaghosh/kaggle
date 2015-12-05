# -*- coding: utf-8 -*-
"""
Preprocessing script.
"""

from sklearn import preprocessing
import gc
import numpy as np
import pandas as pd


pd.options.display.multi_sparse = False


df = pd.read_csv('data/train.csv.gz')
testdf = pd.read_csv('data/test.csv.gz')


def create_dataset_1():
    # Construct dataset 1
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
    


def create_dataset_2(df, le_dept_id, le_dow, is_train=True):
    # Columns : VisitNumber, DoW, DeptXX, DeptYY, ... , TripType
    
    # Create a new DeptID column based on the DepartmentDescription column
    dept_name_col_vals = df.DepartmentDescription.ravel().tolist()
    dept_id_col_vals = le_dept_id.transform(dept_name_col_vals)
    dept_name_id_map = dict(zip(dept_name_col_vals, dept_id_col_vals))
    df["DeptID"] = dept_id_col_vals
    
    # Drop columns we no longer need
    df.drop(["DepartmentDescription", "FinelineNumber", "Upc"], axis=1, 
        inplace=True)
    
    # Group by TripType, VisitNumber, Weekday & DeptID --> sum the ScanCount
    cols = ["VisitNumber", "Weekday", "DeptID"]
    if is_train:
        cols.insert(0, "TripType")
    s = df.groupby(cols).sum()
    df = s.reset_index()
    df.rename(columns={0L:'bb'}, inplace=True) # last column is OL, so rename it
    
    # Convert the 'DeptID' values into individual columns
    # Taking the approach 2 previously used for dataset 1
    dept_ids = sorted(list(set(dept_name_id_map.values())))
    # Check and add missing columns
    missing_dept_ids = missing_elements(dept_ids)
    if len(missing_dept_ids)>0:
        dept_ids.extend(missing_dept_ids)
        dept_ids = sorted(dept_ids)
    # Convert the 'DeptID' values into individual columns
    deptcols = []
    for dept_id in dept_ids:
        colname = "Dept%s" % str(dept_id).zfill(2)
        df[colname] = 0
        deptcols.append(colname)
        df[colname] = df.DeptID.apply(lambda x: 1 if x == dept_id else 0)\
                        .astype(np.int8)
    gc.collect()
    
    # Multiply the DeptXX columns with the ScanCount column
    df_prod = df[deptcols].multiply(df["ScanCount"], axis="index")
    cols = ["VisitNumber", "Weekday", "DeptID", "ScanCount"]
    if is_train:
        cols.insert(0, "TripType")
    df_left = df[cols]
    df = pd.concat([df_left, df_prod], axis=1)
    (df_left, df_prod) = None, None
    df.drop(["DeptID", "ScanCount"], axis=1, inplace=True) # we don't need it
    
    # Convert the Weekday column
    df["DoW"] = le_dow.transform(df.Weekday.ravel().tolist())
    # list(le_dow.inverse_transform([0, 1, 2, 3, 4, 5, 6])) # check
    # Reposition the new DoW column
    cols = df.columns.tolist()
    cols.insert(2, cols.pop(-1))
    df = df[cols]
    df.drop(["Weekday"], axis=1, inplace=True) # we don't need it
    
    # Collapse the rows
    cols = ["VisitNumber", "DoW"]
    if is_train:
        cols.insert(0, "TripType")
    df = df.groupby(cols).aggregate(np.sum)
    df.reset_index(level=[0,1,2] if is_train else [0,1], inplace=True)
    
    # Reposition the TripType column (when present in train)
    cols = ["VisitNumber", "DoW"]
    cols.extend(deptcols)
    if is_train:
        cols.append("TripType")
    df = df[cols]
    
    # Sort by VisitNumber
    df = df.sort_values(by="VisitNumber", ascending=True)
    
    # Persisting
    filename = "data/%s-dataset-2.txt" % ("train" if is_train else "test")
    print filename
    print df.shape
    df.to_csv(filename, sep='\t', index=False)
    
    return df


def main():
    # Prepare to create new DeptID column based on DepartmentDescription column
    # NOTE : Training dataset has 1 extra department 'HEALTH AND BEAUTY AIDS'
    df = pd.read_csv("data/train.csv.gz")
    dept_names = sorted(pd.unique(df.DepartmentDescription.ravel()).tolist())
    le1 = preprocessing.LabelEncoder()
    le1.fit(dept_names)
    
    # Prepare to create new DoW column based on Weekday column
    le2 = preprocessing.LabelEncoder()
    le2.fit(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", 
             "Saturday", "Sunday"])
    
    create_dataset_2(df, le1, le2, True)
    gc.collect()
    testdf = pd.read_csv("data/test.csv.gz")
    create_dataset_2(testdf, le1, le2, False)
    gc.collect()


def missing_elements(L):
    start, end = L[0], L[-1]
    return sorted(set(range(start, end + 1)).difference(L))


if __name__ == '__main__':
    main()
