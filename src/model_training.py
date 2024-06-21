import pandas as pd
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2

# SageMaker session and role
sess = sagemaker.Session()
role = get_execution_role()
bucket = sess.default_bucket()
prefix = "heart-disease-prediction-xgboost"

# Load preprocessed data from S3
s3 = boto3.client('s3')
s3.download_file(bucket, 'data/preprocessed_2020.csv', 'preprocessed_2020.csv')
df = pd.read_csv('preprocessed_2020.csv')

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
df[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']] = scaler.fit_transform(df[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']])

# Chi-square test
x = df.drop(columns='HeartDisease')
y = df['HeartDisease']
chi_scores, p_values = chi2(x, y)
important_features_chi = np.array(x.columns)[p_values < 0.05]
x_important = x[important_features_chi]

# SMOTE
smote = SMOTE(random_state=42)
x_res, y_res = smote.fit_resample(x_important, y)
data_final = pd.concat([pd.DataFrame(y_res, columns=['HeartDisease']), pd.DataFrame(x_res, columns=important_features_chi)], axis=1)

# Split data
train_data, val_data = train_test_split(data_final, test_size=0.2)
train_data.to_csv('train_data.csv', index=False, header=False)
val_data.to_csv('val_data.csv', index=False, header=False)
s3.upload_file('train_data.csv', bucket, f'{prefix}/train/train_data.csv')
s3.upload_file('val_data.csv', bucket, f'{prefix}/validation/val_data.csv')

# XGBoost estimator
image_uri = sagemaker.image_uris.retrieve("xgboost", boto3.Session().region_name, version="1.2-1")
estimator = Estimator(
    image_uri,
    role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{bucket}/{prefix}/output",
    sagemaker_session=sess,
)

# Hyperparameter tuning
hyperparameter_ranges = {
    'max_depth': IntegerParameter(3, 12),
    'eta': ContinuousParameter(0.05, 0.5),
    'gamma': ContinuousParameter(0, 10),
    'min_child_weight': IntegerParameter(2, 8),
    'subsample': ContinuousParameter(0.5, 0.9),
}

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name='validation:f1',
    hyperparameter_ranges=hyperparameter_ranges,
    objective_type='Maximize',
    max_jobs=20,
    max_parallel_jobs=3
)

train_input = sagemaker.inputs.TrainingInput(f"s3://{bucket}/{prefix}/train", content_type="text/csv")
val_input = sagemaker.inputs.TrainingInput(f"s3://{bucket}/{prefix}/validation", content_type="text/csv")
tuner.fit({'train': train_input, 'validation': val_input})
tuner.wait()

best_estimator = tuner.best_estimator()
best_estimator.model_data
