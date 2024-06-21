import pandas as pd
from sklearn import preprocessing
from dataprep.eda import create_report, plot_correlation
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from pyathena import connect
import boto3
import os
import re
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker import get_execution_role
from sagemaker.session import Session
import time
from time import gmtime, strftime, sleep

# Your data preprocessing code here

# Load datasets
df1 = pd.read_csv('heart_2020_cleaned.csv') 
df2 = pd.read_csv('heart_2022_no_nans.csv') 

# Adjust df2 to match df1
df2_bis = pd.DataFrame({
    'HeartDisease': df2.HighRiskLastYear,
    'BMI': df2.BMI,
    'Smoking': df2.SmokerStatus,
    'AlcoholDrinking': df2.AlcoholDrinkers,
    'Stroke': df2.HadStroke,
    'PhysicalHealth': df2.PhysicalHealthDays,
    'MentalHealth': df2.MentalHealthDays,
    'DiffWalking': df2.DifficultyWalking,
    'Sex': df2.Sex,
    'AgeCategory': df2.AgeCategory,
    'Race': df2.RaceEthnicityCategory,
    'Diabetic': df2.HadDiabetes,
    'PhysicalActivity': df2.PhysicalActivities,
    'GenHealth': df2.GeneralHealth,
    'SleepTime': df2.SleepHours,
    'Asthma': df2.HadAsthma,
    'KidneyDisease': df2.HadKidneyDisease,
    'SkinCancer': df2.HadSkinCancer
})

# Uniform df1 and df2_bis
df2_bis.loc[df2_bis.Smoking == 'Former smoker', 'Smoking'] = 'No'
df2_bis.loc[df2_bis.Smoking == 'Never smoked', 'Smoking'] = 'No'
df2_bis.loc[df2_bis.Smoking == 'Current smoker - now smokes every day', 'Smoking'] = 'Yes'
df2_bis.loc[df2_bis.Smoking == 'Current smoker - now smokes some days', 'Smoking'] = 'Yes'

df2_bis.loc[df2_bis.AgeCategory == 'Age 18 to 24', 'AgeCategory'] = '18-24'
df2_bis.loc[df2_bis.AgeCategory == 'Age 25 to 29', 'AgeCategory'] = '25-29'
df2_bis.loc[df2_bis.AgeCategory == 'Age 30 to 34', 'AgeCategory'] = '30-34'
df2_bis.loc[df2_bis.AgeCategory == 'Age 35 to 39', 'AgeCategory'] = '35-39'
df2_bis.loc[df2_bis.AgeCategory == 'Age 40 to 44', 'AgeCategory'] = '40-44'
df2_bis.loc[df2_bis.AgeCategory == 'Age 45 to 49', 'AgeCategory'] = '45-49'
df2_bis.loc[df2_bis.AgeCategory == 'Age 50 to 54', 'AgeCategory'] = '50-54'
df2_bis.loc[df2_bis.AgeCategory == 'Age 55 to 59', 'AgeCategory'] = '55-59'
df2_bis.loc[df2_bis.AgeCategory == 'Age 60 to 64', 'AgeCategory'] = '60-64'
df2_bis.loc[df2_bis.AgeCategory == 'Age 65 to 69', 'AgeCategory'] = '65-69'
df2_bis.loc[df2_bis.AgeCategory == 'Age 70 to 74', 'AgeCategory'] = '70-74'
df2_bis.loc[df2_bis.AgeCategory == 'Age 75 to 79', 'AgeCategory'] = '75-79'
df2_bis.loc[df2_bis.AgeCategory == 'Age 80 or older', 'AgeCategory'] = '80 or older'

df2_bis.loc[df2_bis.Race == 'White only, Non-Hispanic', 'Race'] = 'White'
df2_bis.loc[df2_bis.Race == 'Black only, Non-Hispanic', 'Race'] = 'Black'
df2_bis.loc[df2_bis.Race == 'Multiracial, Non-Hispanic', 'Race'] = 'Multiracial'
df2_bis.loc[df2_bis.Race == 'Other race only, Non-Hispanic', 'Race'] = 'Other'
df1.loc[df1.Race == 'Asian', 'Race'] = 'Multiracial'
df1.loc[df1.Race == 'American Indian/Alaskan Native', 'Race'] = 'Multiracial'

# Convert categorical to numerical features
def cat_to_num(dataset):
    dataset.HeartDisease.replace(('No', 'Yes'), (0, 1), inplace=True)
    dataset.Smoking.replace(('No', 'Yes'), (0, 1), inplace=True)
    dataset.AlcoholDrinking.replace(('No', 'Yes'), (0, 1), inplace=True)
    dataset.Stroke.replace(('No', 'Yes'), (0, 1), inplace=True)
    dataset.DiffWalking.replace(('No', 'Yes'), (0, 1), inplace=True)
    dataset.Sex.replace(('Female', 'Male'), (0, 1), inplace=True)
    dataset.PhysicalActivity.replace(('No', 'Yes'), (0, 1), inplace=True)
    dataset.Asthma.replace(('No', 'Yes'), (0, 1), inplace=True)
    dataset.KidneyDisease.replace(('No', 'Yes'), (0, 1), inplace=True)
    dataset.SkinCancer.replace(('No', 'Yes'), (0, 1), inplace=True)
    label_encoder = preprocessing.LabelEncoder()
    dataset.AgeCategory = label_encoder.fit_transform(dataset.AgeCategory)
    dataset.Race = label_encoder.fit_transform(dataset.Race)
    dataset.Diabetic = label_encoder.fit_transform(dataset.Diabetic)
    dataset.GenHealth = label_encoder.fit_transform(dataset.GenHealth)
    return dataset

df1 = cat_to_num(df1)
df2_bis = cat_to_num(df2_bis)

# Create S3 bucket
session = boto3.session.Session()
region = session.region_name
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
s3 = sagemaker_session.boto_session.resource("s3")

# Upload df1 and df2_bis to S3
df1.to_csv('results_2020.csv', sep=",", index=False)
df2_bis.to_csv('results_2022.csv', sep=",", index=False)
filename1 = 'results_2020.csv'
prefix1 = os.path.basename(filename1)
filename_key1 = filename1.split(".")[0]
s3.Bucket(bucket).upload_file(filename1, "{}/{}/{}".format(prefix1, filename_key1, filename1))

filename2 = 'results_2022.csv'
prefix2 = os.path.basename(filename2)
filename_key2 = filename2.split(".")[0]
s3.Bucket(bucket).upload_file(filename2, "{}/{}/{}".format(prefix2, filename_key2, filename2))

# Create Athena databases for df1 and df2_bis
databasename_csv1 = "tabular_results_2020"
s3_staging_dir = "s3://{0}/athena/staging".format(bucket)
conn = connect(region_name=region, s3_staging_dir=s3_staging_dir)
statement1 = "CREATE DATABASE IF NOT EXISTS {}".format(databasename_csv1)
pd.read_sql(statement1, conn)

databasename_csv2 = "tabular_results_2022"
statement2 = "CREATE DATABASE IF NOT EXISTS {}".format(databasename_csv2)
pd.read_sql(statement2, conn)

# Register tables with Athena for df1 and df2_bis
data_s3_location1 = "s3://{}/{}/{}/".format(bucket, prefix1, filename_key1)
statement1 = """CREATE EXTERNAL TABLE IF NOT EXISTS {}.{}(
         HeartDisease int,
         BMI float,
         Smoking int,
         AlcoholDrinking int,
         Stroke int,
         PhysicalHealth float,
         MentalHealth float,
         DiffWalking int,
         Sex int,
         AgeCategory int,
         Race int,
         Diabetic int,
         PhysicalActivity int,
         GenHealth int,
         SleepTime float,
         Asthma int,
         KidneyDisease int,
         SkinCancer int
) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\\n' LOCATION '{}'
TBLPROPERTIES ('skip.header.line.count'='1')""".format(
    databasename_csv1, 'results_csv_2020', data_s3_location1)
pd.read_sql(statement1, conn)

data_s3_location2 = "s3://{}/{}/{}/".format(bucket, prefix2, filename_key2)
statement2 = """CREATE EXTERNAL TABLE IF NOT EXISTS {}.{}(
         HeartDisease int,
         BMI float,
         Smoking int,
         AlcoholDrinking int,
         Stroke int,
         PhysicalHealth float,
         MentalHealth float,
         DiffWalking int,
         Sex int,
         AgeCategory int,
         Race int,
         Diabetic int,
         PhysicalActivity int,
         GenHealth int,
         SleepTime float,
         Asthma int,
         KidneyDisease int,
         SkinCancer int
) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\\n' LOCATION '{}'
TBLPROPERTIES ('skip.header.line.count'='1')""".format(
    databasename_csv2, 'results_csv_2022', data_s3_location2)
pd.read_sql(statement2, conn)

# Ingest data into FeatureStore
region = boto3.Session().region_name
boto_session = boto3.Session(region_name=region)
sagemaker_client = boto_session.client(service_name='sagemaker', region_name=region)
featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime', region_name=region)
feature_store_session = Session(boto_session=boto_session, sagemaker_client=sagemaker_client, sagemaker_featurestore_runtime_client=featurestore_runtime)
default_s3_bucket_name = feature_store_session.default_bucket()
prefix = 'sagemaker-featurestore-heart-project'
role = get_execution_role()

# Define FeatureGroups
data_2020_feature_group_name = 'data-2020-feature-group-' + strftime('%d-%H-%M-%S', gmtime())
data_2022_feature_group_name = 'data-2022-feature-group-' + strftime('%d-%H-%M-%S', gmtime())
data_2020_feature_group = FeatureGroup(name=data_2020_feature_group_name, sagemaker_session=feature_store_session)
data_2022_feature_group = FeatureGroup(name=data_2022_feature_group_name, sagemaker_session=feature_store_session)

# Cast object dtype to string
def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label] == 'object':
            data_frame[label] = data_frame[label].astype('str').astype('string')

cast_object_to_string(df1)
cast_object_to_string(df2_bis)

record_identifier_feature_name = 'heartdisease'
event_time_feature_name = 'event_time'
current_time_sec = int(round(time.time()))

df1[event_time_feature_name] = pd.Series([current_time_sec] * len(df1), dtype='float64')
df2_bis[event_time_feature_name] = pd.Series([current_time_sec] * len(df2_bis), dtype='float64')

data_2020_feature_group.load_feature_definitions(data_frame=df1)
data_2022_feature_group.load_feature_definitions(data_frame=df2_bis)

data_2020_feature_group.create(
    s3_uri=f"s3://{default_s3_bucket_name}/{prefix}",
    record_identifier_name=record_identifier_feature_name,
    event_time_feature_name=event_time_feature_name,
    role_arn=role,
    enable_online_store=True
)
data_2022_feature_group.create(
    s3_uri=f"s3://{default_s3_bucket_name}/{prefix}",
    record_identifier_name=record_identifier_feature_name,
    event_time_feature_name=event_time_feature_name,
    role_arn=role,
    enable_online_store=True
)

def wait_for_feature_group_creation_complete(feature_group):
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print("Waiting for Feature Group Creation")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    if status != "Created":
        raise RuntimeError(f"Failed to create feature group {feature_group.name}")
    print(f"FeatureGroup {feature_group.name} successfully created.")

wait_for_feature_group_creation_complete(feature_group=data_2020_feature_group)
wait_for_feature_group_creation_complete(feature_group=data_2022_feature_group)

data_2020_feature_group.ingest(data_frame=df1, max_workers=3, wait=True)
data_2022_feature_group.ingest(data_frame=df2_bis, max_workers=3, wait=True)
