import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import boto3
import sagemaker

# Load test data
s3 = boto3.client('s3')
bucket = 'your-s3-bucket-name'
s3.download_file(bucket, 'data/preprocessed_2022.csv', 'preprocessed_2022.csv')
df_test = pd.read_csv('preprocessed_2022.csv')
X_test = df_test.drop('HeartDisease', axis=1)
y_test = df_test['HeartDisease']

# Load model
model_name = 'your_model_name'  # Replace with actual model name
predictor = sagemaker.predictor.Predictor(endpoint_name=model_name)

# Get predictions
predictions = predictor.predict(X_test.values)
predicted_classes = [1 if p > 0.5 else 0 for p in predictions]

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted_classes))
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes))
print(f"F1 Score: {f1_score(y_test, predicted_classes)}")

