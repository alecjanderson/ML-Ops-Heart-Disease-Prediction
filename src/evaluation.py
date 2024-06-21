import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Your evaluation code here

# Load predictions and true labels
predictions = pd.read_csv('predictions.csv')
true = pd.read_csv('true_labels.csv')

# Evaluate model performance
predicted = list(map(lambda x: 0 if x else 1, (predictions < 0.5).values))
cm = confusion_matrix(true, predicted)
cm_image = ConfusionMatrixDisplay(cm, display_labels=['no heart disease', 'heart disease'])
cm_image.plot()

cr = classification_report(true, predicted)
print(cr)

roc_auc = roc_auc_score(true, predictions)
print('ROC-AUC score:', round(roc_auc, 2))

# Save confusion matrix plot
plt.savefig('Confusion_Matrix.jpeg')
plt.show()
