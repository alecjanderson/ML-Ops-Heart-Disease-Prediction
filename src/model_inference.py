import boto3
import sagemaker
import pandas as pd

# Load test data
s3 = boto3.client('s3')
bucket = 'your-s3-bucket-name'
s3.download_file(bucket, 'data/preprocessed_2022.csv', 'preprocessed_2022.csv')
df_test = pd.read_csv('preprocessed_2022.csv')
X_test = df_test.drop('HeartDisease', axis=1)

# Load model
model_name = 'your_model_name'  # Replace with actual model name
predictor = sagemaker.predictor.Predictor(endpoint_name=model_name)

# Get predictions
predictions = predictor.predict(X_test.values)
predicted_classes = [1 if p > 0.5 else 0 for p in predictions]

# Save predictions
predictions_df = pd.DataFrame({'predictions': predicted_classes})
predictions_df.to_csv('predictions.csv', index=False)

