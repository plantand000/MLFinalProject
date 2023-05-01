import time
import pandas as pd
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data(x_path):
    return pd.read_csv(x_path, low_memory=False)


def split_data(x, y, split=0.8):
    # Your code here
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=split, random_state=42)
    return X_train, X_test, y_train, y_test


def preprocess_x(df_x, df_y):

    df_x = df_x.drop(df_x.columns[[0]], axis=1)
    df_y = df_y.drop(df_y.columns[[0]], axis=1)

    data = None
    #########
    # FIRST #
    # Process patient demographic data: admissionheight, admissionweight, age, ethnicity, gender, unitvisitnumber

    demographicCols = ['admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'patientunitstayid',  'unitvisitnumber']
    maskCols = ['admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender']
    # make a mask to get the non null rows for these cols
    mask = df_x[maskCols].notnull().any(axis=1)
    demographicData = df_x.loc[mask, demographicCols]

    # height and weight estimates for average man and woman for nan rows
    averageMaleHeight = 175
    averageFemaleHeight = 162
    averageMaleWeight = 71.9
    averageFemaleWeight = 56.5

    # replace null rows for height and weight with average male and female height and weights
    for i, row in demographicData.iterrows():
        if pd.isnull(row['admissionheight']):
            if row['gender'] == 'Male':
                demographicData.at[i, 'admissionheight'] = averageMaleHeight
            elif row['gender'] == 'Female':
                demographicData.at[i, 'admissionheight'] = averageFemaleHeight
        if pd.isnull(row['admissionweight']):
            if row['gender'] == 'Male':
                demographicData.at[i, 'admissionweight'] = averageMaleWeight
            elif row['gender'] == 'Female':
                demographicData.at[i, 'admissionweight'] = averageFemaleWeight
        if pd.isnull(row['age']):
            demographicData.at[i, 'age'] = '40'

    # convert ages to integers and replace the > 89 to 90 so that it can be an integer
    demographicData['age'] = demographicData['age'].replace({'> 89': '90'}).astype(int)

    # convert categorical columns to numerical values
    numericEthnicity = pd.get_dummies(demographicData['ethnicity'])
    numericGender = pd.get_dummies(demographicData['gender'])

    # remove categorical columns and combine numeric data with the rest
    demographicData = demographicData.drop(['gender', 'ethnicity'], axis=1)
    demographicData = pd.concat([demographicData, numericEthnicity, numericGender], axis=1)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    ##########
    # SECOND #
    # this data all has to do with the capillary refill test
    # Process cell data: cellattributevalue, celllabel

    # make a mask to get the non null rows for these cols
    cellCols = ['cellattributevalue', 'celllabel', 'offset', 'patientunitstayid']
    maskCols = ['cellattributevalue', 'celllabel']
    mask = df_x[maskCols].notnull().any(axis=1)
    cellData = df_x.loc[mask, cellCols]
    
    # I think hands and feet are just classifiers saying where the test was test while, normal, > 2 seconds, < 2 seconds are the actual data from the test
    # remove all rows that say hands or feet
    cellData = cellData[~cellData['cellattributevalue'].isin(['hands', 'feet'])]
    
    cellData['offset'] = pd.to_numeric(cellData['offset'])
    groups = cellData.groupby('patientunitstayid')
    
    # loop through all patient groups and find the max offset
    # the max offset is the most recent test time. for this test it should only be the results of the most recent test that matters
    # capillary test
    highestOffsets = []
    for _, group in groups:
        maxRow = group.loc[group['offset'].idxmax()]
        highestOffsets.append(maxRow)
    
    mostRecentCellTestData = pd.DataFrame(highestOffsets)
    mostRecentCellTestData = mostRecentCellTestData.rename(columns={'offset': 'maxcelltime'})

    # convert categorical columns to numerical values
    numericCellAttributeValue = pd.get_dummies(mostRecentCellTestData['cellattributevalue'])
    numericCellLabel = pd.get_dummies(mostRecentCellTestData['celllabel'])

    # add numeric columns to rest of data
    mostRecentCellTestData = mostRecentCellTestData.drop(['cellattributevalue', 'celllabel', ], axis=1)
    mostRecentCellTestData = pd.concat([mostRecentCellTestData, numericCellAttributeValue, numericCellLabel], axis=1)

    # merge the demographic data on cellData, data now holds all preprocessed data up to this point
    data = demographicData.merge(mostRecentCellTestData, on='patientunitstayid', how='left')
    data = data.fillna(value=0)

    #########
    # THIRD #
    # Process lab data: labmeasurenamesystem, labname, labresult + patient id

    labCols = ['labmeasurenamesystem', 'labname', 'labresult', 'offset', 'patientunitstayid']
    maskCols = ['labmeasurenamesystem', 'labname', 'labresult']
    mask = df_x[maskCols].notnull().any(axis=1)
    labData = df_x.loc[mask, labCols]

    mostRecentLabData = []
    # group data by patient stay id and loop over the users
    patientGroups = labData.groupby('patientunitstayid')
    for stayid, group in patientGroups:
        # group by each user's labs and loop over those groups to find the most recent test data
        testGroups = group.groupby('labname')

        glucoseRow = None
        glucoseData = {
            'labmeasurenamesystem glucose': [np.nan],
            'labname glucose': [np.nan],
            'labresult glucose': [np.nan],
            'maxlabtime glucose': [np.nan],
            'patientunitstayid': [stayid]
        }
        pHRow = None
        pHData = {
            'labmeasurenamesystem pH': [np.nan],
            'labname pH': [np.nan],
            'labresult pH': [np.nan],
            'maxlabtime pH': [np.nan],
            'patientunitstayid': [stayid]
        }

        for chartTypeName, testGroup in testGroups:
            maxRow = testGroup.loc[testGroup['offset'].idxmax()]
            if chartTypeName == 'glucose':
                glucoseData = {
                    'labmeasurenamesystem glucose': [maxRow['labmeasurenamesystem']],
                    'labname glucose': [maxRow['labname']],
                    'labresult glucose': [maxRow['labresult']],
                    'maxlabtime glucose': [maxRow['offset']],
                    'patientunitstayid': [maxRow['patientunitstayid']]
                }
            elif chartTypeName == 'pH':
                pHData = {
                    'labmeasurenamesystem pH': [maxRow['labmeasurenamesystem']],
                    'labname pH': [maxRow['labname']],
                    'labresult pH': [maxRow['labresult']],
                    'maxlabtime pH': [maxRow['offset']],
                    'patientunitstayid': [maxRow['patientunitstayid']]
                }
    
        glucoseRow = pd.DataFrame(glucoseData)
        pHRow = pd.DataFrame(pHData)
        mergedRows = glucoseRow.merge(pHRow, on='patientunitstayid', how='left')
        mostRecentLabData.append(mergedRows)

    mostRecentLabTestData = pd.concat(mostRecentLabData, ignore_index=True)

    # convert categorical columns to numerical values
    numericLabNameSystemGlucose = pd.get_dummies(mostRecentLabTestData['labmeasurenamesystem glucose'])
    numericLabNameGlucose = pd.get_dummies(mostRecentLabTestData['labname glucose'])
    numericLabNameSystemPH = pd.get_dummies(mostRecentLabTestData['labmeasurenamesystem pH'])
    numericLabNamePH = pd.get_dummies(mostRecentLabTestData['labname pH'])

    # add numeric columns to rest of data
    mostRecentLabTestData = mostRecentLabTestData.drop(['labmeasurenamesystem glucose', 'labname glucose', 'labmeasurenamesystem pH', 'labname pH'], axis=1)
    mostRecentLabTestData = pd.concat([mostRecentLabTestData, numericLabNameSystemGlucose, numericLabNameGlucose,numericLabNameSystemPH, numericLabNamePH], axis=1)

    # merge the rest of the data with the lab test data, data now holds all preprocessed data up to this point
    data = data.merge(mostRecentLabTestData, on='patientunitstayid', how='left')
    data = data.fillna(value=0)

    ##########
    # FOURTH #
    # Process nursing data: nursingchartcelltypevalname, nursingchartvalue + patient id

    nursingCols = ['nursingchartcelltypevalname', 'nursingchartvalue', 'offset', 'patientunitstayid']
    maskCols = ['nursingchartcelltypevalname', 'nursingchartvalue']
    mask = df_x[maskCols].notnull().any(axis=1)
    nursingData = df_x.loc[mask, nursingCols]

    # group data by patient stay id and loop over the users
    mostRecentNursingData = []
    nursingChartCellType = df_x['nursingchartcelltypevalname'].dropna().unique()
    patientGroups = nursingData.groupby('patientunitstayid')
    for stayid, group in patientGroups:
        # group by each user's nursing charts and loop over those groups to find the most recent data
        chartTypeGroups = group.groupby('nursingchartcelltypevalname')

        # loop through all the chart types and initialize their dicts with nan data to be filled in later
        chartTypeDict = {}
        for chartType in nursingChartCellType:
            chartTypeDict[chartType] = {
                'nursingchartcelltypevalname ' + chartType: [np.nan],
                'nursingchartvalue ' + chartType: [np.nan],
                'maxlabtime ' + chartType: [np.nan],
                'patientunitstayid': [stayid]
            }

        # loop through the actual chart data and fill in the respective dicts
        for chartTypeName, chartTypeGroup in chartTypeGroups:
            maxRow = chartTypeGroup.loc[chartTypeGroup['offset'].idxmax()]
            chartTypeDict[chartTypeName]['nursingchartcelltypevalname ' + chartTypeName] = [maxRow['nursingchartcelltypevalname']]
            chartTypeDict[chartTypeName]['nursingchartvalue ' + chartTypeName] = [maxRow['nursingchartvalue']]
            chartTypeDict[chartTypeName]['maxlabtime ' + chartTypeName] = [maxRow['offset']]
            chartTypeDict[chartTypeName]['patientunitstayid'] = [maxRow['patientunitstayid']]

        # convert each of the dicts to a list of dataframes
        chartDataFrames = [pd.DataFrame(chartTypeDict[chart]) for chart in chartTypeDict]

        # merge all of the data frames together on the patient id
        mainDataFrame = chartDataFrames[0]
        for df in chartDataFrames[1:]:
            mainDataFrame = mainDataFrame.merge(df, on='patientunitstayid', how='left')
        
        mostRecentNursingData.append(mainDataFrame)
    
    # combine all of the rows made in the loop above into one singel data frame
    mostRecentNursingData = pd.concat(mostRecentNursingData, ignore_index=True)

    # convert all of the categorical data to numerical data
    chartNumericalData = [pd.get_dummies(mostRecentNursingData['nursingchartcelltypevalname ' + chartType]) for chartType in nursingChartCellType]
    chartCategoricalDataNames = ['nursingchartcelltypevalname ' + chartType for chartType in nursingChartCellType]

    mostRecentNursingData = mostRecentNursingData.drop(chartCategoricalDataNames, axis=1)
    mostRecentNursingData = pd.concat([mostRecentNursingData, *chartNumericalData], axis=1)

    mostRecentNursingData['nursingchartvalue GCS Total'] = mostRecentNursingData['nursingchartvalue GCS Total'].replace({'Unable to score due to medication': '0'})

    data = data.merge(mostRecentNursingData, on='patientunitstayid', how='left')
    data = data.fillna(value=0)

    categoricalCols = ['African American', 'Asian', 'Caucasian', 'Hispanic', 'Native American', 'Other/Unknown', 'Female', 'Male', 'mg/dL', 'glucose', 'pH',
                        'Respiratory Rate', 'O2 Saturation', 'Heart Rate', 'Non-Invasive BP Systolic', 'Non-Invasive BP Diastolic', 'Invasive BP Diastolic', 'Invasive BP Systolic',
                        'GCS Total', 'Non-Invasive BP Mean', 'Invasive BP Mean', 'patientunitstayid']
    

    if not 'Native American' in data.columns:
        data.insert(9, 'Native American', False)

    data = data.reindex(columns=['admissionheight', 'admissionweight', 'age', 'patientunitstayid',
       'unitvisitnumber', 'African American', 'Asian', 'Caucasian', 'Hispanic',
       'Native American', 'Other/Unknown', 'Female', 'Male', 'maxcelltime',
       '< 2 seconds', '> 2 seconds', 'normal', 'Capillary Refill',
       'labresult glucose', 'maxlabtime glucose', 'labresult pH',
       'maxlabtime pH', 'mg/dL', 'glucose', 'pH',
       'nursingchartvalue Respiratory Rate', 'maxlabtime Respiratory Rate',
       'nursingchartvalue O2 Saturation', 'maxlabtime O2 Saturation',
       'nursingchartvalue Heart Rate', 'maxlabtime Heart Rate',
       'nursingchartvalue Non-Invasive BP Systolic',
       'maxlabtime Non-Invasive BP Systolic',
       'nursingchartvalue Non-Invasive BP Diastolic',
       'maxlabtime Non-Invasive BP Diastolic',
       'nursingchartvalue Invasive BP Diastolic',
       'maxlabtime Invasive BP Diastolic',
       'nursingchartvalue Invasive BP Systolic',
       'maxlabtime Invasive BP Systolic', 'nursingchartvalue GCS Total',
       'maxlabtime GCS Total', 'nursingchartvalue Non-Invasive BP Mean',
       'maxlabtime Non-Invasive BP Mean', 'nursingchartvalue Invasive BP Mean',
       'maxlabtime Invasive BP Mean', 'Respiratory Rate', 'O2 Saturation',
       'Heart Rate', 'Non-Invasive BP Systolic', 'Non-Invasive BP Diastolic',
       'Invasive BP Diastolic', 'Invasive BP Systolic', 'GCS Total',
       'Non-Invasive BP Mean', 'Invasive BP Mean'])
    
 
    numericalData = data.drop(categoricalCols, axis=1)
    numericalData = data.drop(categoricalCols, axis=1).astype(float)
    

    # scale numerical columns using StandardScaler
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(numericalData)
    scaledData = pd.DataFrame(scaledData, columns=numericalData.columns)
    
    # combine categorical and scaled numerical columns
    data = pd.concat([data[categoricalCols], scaledData], axis=1)

    # save preprocessed data to data/logs folder
    # data.to_csv(f'data/logs/data_{time.time()}.csv', header=True, index=False)
    xy = data.merge(df_y, on='patientunitstayid', how='left')
    
    data_y = xy['hospitaldischargestatus']

    patientunit = xy['patientunitstayid']
    
    data_x = xy.drop(['patientunitstayid', 'hospitaldischargestatus'], axis=1)

    return data_x, data_y, patientunit
