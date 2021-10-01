import csv
import numpy as np
import pandas as pd
from pandas import Series
import datetime
import time
import os
from fancyimpute import KNN, MatrixFactorization
import math

path = r"C:\Users\ln\Desktop\Bldg59"
files = os.listdir(path)
path_postprocess = r"C:\Users\ln\Desktop\Bldg59_postprocess"

#read data files and adjust time format
for filename in files:
    row = pd.read_csv(path+'\\'+filename)
    row['date'] = pd.to_datetime(row['date'], format="%m/%d/%Y %H:%M") 
    helper=pd.DataFrame({'date': pd.date_range(row['date'].min(), row['date'].max(), freq='15min')})
    row = pd.merge(row, helper, on='date', how='outer').sort_values('date')
    count_out = Series([0],index=['date']) #count of outlier values
    count_gap = Series([0],index=['date']) #count of gap
    count_outgap = Series([0],index=['date']) #count of large gap (e.g., one day)
    gap_max=Series([0],index=['date']) #maximum gap
    #calculate the count of gap and do the interpolation based on the gap size 
    for i in range(1, len(row.columns)):
        k = 0
        out_gapcount=0
        start_index = {}
        starttime = {}
        end_index = {}
        endtime = {}
        gap = {}
        if pd.isnull(row.iloc[len(row.index)-1,i]) == True or math.isnan(row.iloc[len(row.index)-1,i])==True:
            row.iloc[len(row.index)-1,i]=0
        for j in range(0, len(row.index)):
            if (pd.isnull(row.iloc[j,i]) or math.isnan(row.iloc[j,i]))and pd.isnull(row.iloc[j-1,i]) == False:
                starttime[k]=row.iloc[j-1,0] #start time of the gap
                start_index[k]=j-1
            elif (pd.isnull(row.iloc[j-1,i]) or math.isnan(row.iloc[j-1,i])) and pd.isnull(row.iloc[j,i]) == False:
                endtime[k]=row.iloc[j,0] #end time of the gap
                end_index[k]=j
                k=k+1
        if k != 0:
            for m in range(k):
                starttime_struct=datetime.datetime.strptime(str(starttime[m]), '%Y-%m-%d %H:%M:%S')
                endtime_struct = datetime.datetime.strptime(str(endtime[m]), '%Y-%m-%d %H:%M:%S')
                gap[m]=(endtime_struct-starttime_struct).total_seconds()
                if  gap[m]<= 3600: #linear interpolation if the gap is less than one hour
                    row.iloc[start_index[m]:end_index[m]+1,i]=row.iloc[start_index[m]:end_index[m]+1,i].interpolate(method='linear')
                elif gap[m] >3600*24:
                    out_gapcount=out_gapcount+1
            maxgap = max(gap.values())/60
            gap_max=gap_max.append(Series(maxgap,index=[row.columns[i]]))
        outcount=np.sum(row.iloc[:, i]<0)/len(row)
        count_out=count_out.append(Series(outcount, index=[row.columns[i]]))
        count_gap= count_gap.append(Series(k, index=[row.columns[i]]))
        count_outgap = count_outgap.append(Series(out_gapcount,index=[row.columns[i]]))
        row_interpolation=np.array(row.iloc[:,1:])
    row_interpolation= KNN(k=3).fit_transform(row_interpolation) #Apply knn algorithm if the gap is larger than one hour
    for i in range(1, len(row.columns)):
        k=0
        start_index = {}
        starttime = {}
        end_index = {}
        endtime = {}
        for j in range(0, len(row.index)):
            if pd.isnull(row.iloc[j,i]) and pd.isnull(row.iloc[j-1,i]) == False:
                starttime[k]=row.iloc[j-1,0]
                start_index[k]=j-1
            elif pd.isnull(row.iloc[j-1,i]) and pd.isnull(row.iloc[j,i]) == False:
                endtime[k]=row.iloc[j,0]
                end_index[k]=j
                k=k+1
        for m in range(k):
            starttime_struct=datetime.datetime.strptime(str(starttime[m]), '%Y-%m-%d %H:%M:%S')
            endtime_struct = datetime.datetime.strptime(str(endtime[m]), '%Y-%m-%d %H:%M:%S')
            gap[m]=(endtime_struct-starttime_struct).total_seconds()
            if  gap[m]>= 3600*24:
                row_interpolation[start_index[m]:end_index[m]+1,i-1]=None
    if out_gapcount !=0:
        row_interpolation= MatrixFactorization().fit_transform(row_interpolation) #Apply MF algorithm if the gap is larger than one day         
    row.iloc[:,1:]=row_interpolation
    cols_not_null = (len(row)-row.count(axis=0))/len(row)
    data=pd.DataFrame({'missingrate':cols_not_null,'outrate':count_out,'count_outgap':count_outgap,'count_gap':count_gap,'maxgap':gap_max})
    data.to_csv(path_postprocess+'\\'+'parameter_'+filename, sep=',', header=True, index=True)
    row.to_csv(path_postprocess+'\\'+'data_'+filename, sep=',', header=True, index=False)

