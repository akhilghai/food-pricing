import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle
import data_collection_preprocessing
import json
from sklearn.model_selection import cross_val_score, KFold

# List of commodities
commodities = data_collection_preprocessing.commodity_tickers.keys()
# os.makedirs("models_and_reports")

# Function to create lagged features
def create_lagged_features(data, lag):
    df = data.copy()
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['value'].shift(i)
    return df.dropna()


# Load the dataset
# def train_model(combined_data=data_collection_preprocessing.fetch_from_mysql("commodity_data"),MODEL_DIR="models_and_reports"):
def train_model_and_save(combined_data,MODEL_DIR):

    all_metrics = {}  # Dictionary to store metrics for all commodities
    for commodity in commodities:
        model_path = os.path.join(MODEL_DIR, f"{commodity}_rf_model.pkl")
        # report_path = os.path.join(MODEL_DIR, f"{commodity}_accuracy_report.txt")
        print(f"Training model for {commodity}...")
        # Filter the data for the current commodity
        commodity_data = combined_data[['Date', commodity]].dropna()

        # Convert 'Date' column to datetime format
        # commodity_data['Date'] = pd.to_datetime(commodity_data['Date'], errors='coerce')
        commodity_data = commodity_data.dropna(subset=['Date'])
        commodity_data.set_index('Date', inplace=True)

        # Prepare supervised learning data
        commodity_data = commodity_data.rename(columns={commodity: 'value'})
        supervised_data = create_lagged_features(commodity_data, lag=12)

        # Train-test split
        X = supervised_data.drop(columns=['value'])
        y = supervised_data['value']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions and evaluate
        y_pred = rf_model.predict(X_test)

        # Calculate metrics
        metrics = {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R-Square": r2_score(y_test, y_pred)
        }

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf_model, X, y, scoring='neg_mean_squared_error', cv=kf)
        metrics["Cross-Validation RMSE"] = np.sqrt(-np.mean(cv_scores))

        # Add metrics to the dictionary
        all_metrics[commodity] = metrics

        residuals_df = pd.DataFrame()
        # Store residuals in a DataFrame
        temp_residuals_df = pd.DataFrame({
            "Date": y_test.index,
            "Actual": y_test,
            "Predicted": y_pred,
            "Residuals": y_test - y_pred,
            "Commodity": commodity
        })
        residuals_df = pd.concat([residuals_df, temp_residuals_df], ignore_index=True)

        

        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model, f)

    # Save all metrics to a JSON file
    metrics_json_path = os.path.join(MODEL_DIR, "evaluation_metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)

    # Save residuals DataFrame as CSV
    residuals_csv_path = os.path.join(MODEL_DIR, "residuals_data.csv")
    residuals_df.to_csv(residuals_csv_path, index=False)

    print(f"All evaluation metrics saved successfully in {metrics_json_path}")
    print(f"Residuals DataFrame saved successfully as CSV in {residuals_csv_path}")


# train_model()