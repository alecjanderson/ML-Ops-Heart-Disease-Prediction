import pandas as pd
from sklearn import preprocessing
import boto3

# Load data
df1 = pd.read_csv('heart_2020_cleaned.csv')
df2 = pd.read_csv('heart_2022_no_nans.csv')

# Preprocess data
df2_bis = pd.DataFrame({'HeartDisease': df2.HighRiskLastYear,
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
                       'SkinCancer': df2.HadSkinCancer})

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

# Save preprocessed data
df1.to_csv('preprocessed_2020.csv', index=False)
df2_bis.to_csv('preprocessed_2022.csv', index=False)

# Upload to S3
s3 = boto3.client('s3')
bucket = 'your-s3-bucket-name'
s3.upload_file('preprocessed_2020.csv', bucket, 'data/preprocessed_2020.csv')
s3.upload_file('preprocessed_2022.csv', bucket, 'data/preprocessed_2022.csv')
