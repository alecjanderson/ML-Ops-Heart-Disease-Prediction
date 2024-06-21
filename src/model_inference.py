import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
import boto3
import re
import time
from time import gmtime, strftime, sleep
from sagemaker.session import Session
from sagemaker import get_execution_role
import sagemaker

# Your model inference code here

# Helper function to get CSV output from S3
def get_csv_output_from_s3(s3uri, batch_file):
    file_name = "{}.out".format(batch_file)
    match = re.match("s3://([^/]+)/(.*)", "{}/{}".format(s3uri, file_name))
    output_bucket, output_prefix = match.group(1), match.group(2)
    s3 = boto3.client("s3")
    s3.download_file(output_bucket, output_prefix, file_name)
    return pd.read_csv(file_name, sep=",", header=None)

# Load batch data
data_batch = pd.read_csv('batch_data.csv')

# Invoke the deployed endpoint
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=boto3.Session().region_name)
response = sagemaker_runtime.invoke_endpoint(
    EndpointName='xgb-tuned-endpoint',
    ContentType='text/csv',
    Body=data_batch.to_csv(header=None, index=False).strip('\n').split('\n')[0]
)
print(response['Body'].read().decode('utf-8'))
