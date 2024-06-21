import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn import metrics
from sklearn.metrics import accuracy_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import boto3
import sagemaker
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner
from sagemaker.session import Session
from sagemaker import get_execution_role
import os
import time
from time import gmtime, strftime

# Your model training code here

# Load and preprocess data
df = pd.read_csv('results_2020.csv')
scaler = MinMaxScaler(feature_range=(0, 1))
df[['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']] = scaler.fit_transform(df[['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']])
x = df.drop(columns='heartdisease')
y = df['heartdisease']
chi_scores, p_values = chi2(x, y)
important_features_chi = np.array(x.columns)[p_values < 0.05]
x_important = x[important_features_chi]
smote = SMOTE(random_state=42)
x_res, y_res = smote.fit_resample(x_important, y)
data_final = pd.concat([y_res, x_res], axis=1)

# Split data into training, validation, and batch sets
rand_split = np.random.rand(len(data_final))
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.9)
batch_list = rand_split >= 0.9
data_train = data_final[train_list]
data_val = data_final[val_list]
data_batch = data_final[batch_list].drop(["heartdisease"], axis=1)

# Upload data to S3
role = sagemaker.get_execution_role()
sess = sagemaker.Session()
bucket = sess.default_bucket()
prefix = "heart-disease-prediction-xgboost"
train_file = "train_data.csv"
data_train.to_csv(train_file, index=False, header=False)
sess.upload_data(train_file, key_prefix="{}/train".format(prefix))

validation_file = "validation_data.csv"
data_val.to_csv(validation_file, index=False, header=False)
sess.upload_data(validation_file, key_prefix="{}/validation".format(prefix))

batch_file = "batch_data.csv"
data_batch.to_csv(batch_file, index=False, header=False)
sess.upload_data(batch_file, key_prefix="{}/batch".format(prefix))

# Create XGBoost model
job_name = "xgb-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
output_location = "s3://{}/{}/output/{}".format(bucket, prefix, job_name)
image = sagemaker.image_uris.retrieve(framework="xgboost", region=boto3.Session().region_name, version="1.7-1")

sm_estimator = sagemaker.estimator.Estimator(
    image,
    role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size=50,
    input_mode="File",
    output_path=output_location,
    sagemaker_session=sess,
)

sm_estimator.set_hyperparameters(
    objective="binary:logistic",
    max_depth=6,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    verbosity=0,
    num_round=100,
)

train_data = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/train".format(bucket, prefix),
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
)
validation_data = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/validation".format(bucket, prefix),
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
)
data_channels = {"train": train_data, "validation": validation_data}

sm_estimator.fit(inputs=data_channels, job_name=job_name, logs=True)
