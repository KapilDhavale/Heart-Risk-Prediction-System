# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib  # Added for model and scaler saving

# Load dataset
data = pd.read_csv('heart.csv')

# Data Preprocessing
missing_values = data.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)
print("\nSummary of missing values:")
print(data.isnull().sum().describe())

# Data Splitting
X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Scaling of Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
print("Scaled Data:")
print(scaled_df.head())

# Save the scaler
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to {scaler_filename}")

# Check the scaler type and print scaling parameters
if isinstance(scaler, StandardScaler):
    print("Scaling parameters (mean and standard deviation) for each feature:")
    for feature, mean, std in zip(X.columns, scaler.mean_, scaler.scale_):
        print(f"{feature}: mean={mean}, std={std}")
elif isinstance(scaler, MinMaxScaler):
    print("Scaling parameters (min and max) for each feature:")
    for feature, min_val, max_val in zip(X.columns, scaler.data_min_, scaler.data_max_):
        print(f"{feature}: min={min_val}, max={max_val}")
else:
    print("Unknown scaler type. Cannot determine scaling parameters.")

# Hyperparameter Tuning
rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid, n_iter=100, cv=5, random_state=42)
random_search.fit(X_train_scaled, y_train)
best_rf_model = random_search.best_estimator_
print("Best hyperparameters:", random_search.best_params_)
print("Best accuracy score:", random_search.best_score_)

# Model Training
train_scores = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=10)
best_rf_model.fit(X_train_scaled, y_train)
train_accuracy = best_rf_model.score(X_train_scaled, y_train)
test_accuracy = best_rf_model.score(X_test_scaled, y_test)
print("Training Accuracy (CV): {:.2f}%".format(train_scores.mean() * 100))
print("Training Accuracy (Overall): {:.2f}%".format(train_accuracy * 100))
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

# Save the model
model_filename = 'best_rf_model.pkl'
joblib.dump(best_rf_model, model_filename)
print(f"Model saved to {model_filename}")

# Load the model (for future use)
loaded_model = joblib.load(model_filename)
print("Model loaded successfully")

# Feature Importance
importances = best_rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Select top 10 most important features
top_10_features = feature_importance_df.head(10)

print("Top 10 Most Important Features:")
print(top_10_features)

# Plot top 10 most important features
plt.figure(figsize=(10, 6))
plt.barh(top_10_features['Feature'], top_10_features['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()
plt.show()

# Performance Metrics
y_pred = best_rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1))
print("Confusion Matrix:")
print(conf_matrix)

y_train_pred = best_rf_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Training Precision: {:.2f}".format(train_precision))
print("Training Recall: {:.2f}".format(train_recall))
print("Training F1-score: {:.2f}".format(train_f1))
print("Training Confusion Matrix:")
print(train_conf_matrix)

y_prob = best_rf_model.predict_proba(X_test_scaled)[:, 1]
print("Predicted Probabilities:")
print(y_prob)

class_distribution = pd.Series(y_train).value_counts()
plt.figure(figsize=(6, 4))
class_distribution.plot(kind='bar', color='blue')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
print("Class Distribution:")
print(class_distribution)

# Indices Of Test Data
test_indices_list = list(X_test.index)
print("All indices of the test data:")
print(test_indices_list)

# Data Prediction Sample
test_results = pd.DataFrame(data=X_test, columns=X.columns)
test_results['Actual_Label'] = y_test
test_results['Predicted_Label'] = y_pred
print("Test Data with Predicted and Actual Labels:")
print(test_results)

# Entering The Test Data
def get_input_values(feature_names):
    input_values = []
    for feature in feature_names:
        value = float(input(f"Enter value for {feature}: "))
        input_values.append(value)
    return input_values

def predict(input_values, model, scaler, feature_names):
    # Create DataFrame for scaling
    input_df = pd.DataFrame([input_values], columns=feature_names)
    
    # Scale the input values
    scaled_input = scaler.transform(input_df)
    
    # Predict using the model
    prediction = model.predict(scaled_input)
    return prediction[0]

if __name__ == "__main__":
    input_values = get_input_values(X.columns)
    prediction = predict(input_values, best_rf_model, scaler, X.columns)
    print("Prediction for target column:", prediction)
