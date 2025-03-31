# src/ai/inventory_predictor.py
import pandas as pd
from prophet import Prophet
import joblib # For saving/loading models
import os
import streamlit as st
from datetime import date, timedelta

# Directory to store trained models
MODEL_DIR = "trained_inventory_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def get_model_path(item_id: int) -> str:
    """Generates the file path for a given item's trained model."""
    return os.path.join(MODEL_DIR, f"inventory_model_{item_id}.joblib")

@st.cache_resource(ttl=3600) # Cache the loaded model for an hour
def load_model(item_id: int):
    """Loads a trained Prophet model for a specific item ID."""
    model_path = get_model_path(item_id)
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model for item {item_id}: {e}", icon="🚨")
            return None
    else:
        # Model doesn't exist yet, will be trained on first prediction attempt
        return None

def train_and_save_model(item_id: int, df_prepared: pd.DataFrame):
    """
    Trains a Prophet model on the prepared data and saves it.

    Args:
        item_id (int): The ID of the inventory item.
        df_prepared (pd.DataFrame): DataFrame with 'ds' and 'y' columns.

    Returns:
        Prophet: The trained model, or None if training failed.
    """
    if df_prepared.empty or len(df_prepared) < 2:
        st.warning(f"Cannot train model for item {item_id}: Insufficient data.", icon="⚠️")
        return None

    model_path = get_model_path(item_id)

    # Initialize and train the Prophet model
    # Consider adding seasonality options if applicable (e.g., weekly_seasonality=True)
    # Adjust changepoint_prior_scale for flexibility (higher value = more flexible)
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True, # Assume weekly patterns might exist
        daily_seasonality=False,
        changepoint_prior_scale=0.05 # Default, adjust based on results
    )

    try:
        # Suppress Prophet's informational messages during fitting
        # Note: This requires redirecting stdout/stderr, which can be complex in Streamlit.
        # For simplicity, we'll let Prophet print its messages for now.
        # Consider using logging configuration if messages become too noisy.
        print(f"Training Prophet model for item ID: {item_id}...")
        model.fit(df_prepared)
        print(f"Training complete for item ID: {item_id}.")

        # Save the trained model
        joblib.dump(model, model_path)
        st.info(f"Trained and saved model for item {item_id}.", icon="💾")
        # Clear the cached resource for this model to force reload next time
        st.cache_resource.clear()
        load_model.clear() # Clear specific function cache if needed
        return model
    except Exception as e:
        st.error(f"Error training model for item {item_id}: {e}", icon="🚨")
        # Clean up potentially corrupted model file if save failed midway (optional)
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except OSError:
                pass # Ignore error if removal fails
        return None

def predict_future_stock(item_id: int, df_prepared: pd.DataFrame, days_ahead: int = 30):
    """
    Loads a trained model (or trains if not found) and predicts future stock levels.

    Args:
        item_id (int): The ID of the inventory item.
        df_prepared (pd.DataFrame): Prepared historical data ('ds', 'y').
        days_ahead (int): Number of days into the future to forecast.

    Returns:
        pd.DataFrame: DataFrame containing the forecast ('ds', 'yhat', 'yhat_lower', 'yhat_upper'),
                      or an empty DataFrame if prediction fails.
    """
    if df_prepared.empty:
        # Warning already shown by get_preprocessed_item_history
        return pd.DataFrame()

    model = load_model(item_id)

    # If model not loaded (doesn't exist or error loading), try training it now
    if model is None:
        st.info(f"No pre-trained model found for item {item_id}. Training now...", icon="⏳")
        model = train_and_save_model(item_id, df_prepared)
        if model is None:
            st.error(f"Failed to train model for item {item_id}. Cannot generate forecast.", icon="🚨")
            return pd.DataFrame() # Return empty if training failed

    try:
        # Create future dataframe for prediction
        future = model.make_future_dataframe(periods=days_ahead)

        # Generate forecast
        # Suppress Prophet messages here too if needed
        print(f"Generating forecast for item ID: {item_id}...")
        forecast = model.predict(future)
        print(f"Forecast complete for item ID: {item_id}.")

        # Select relevant columns and filter for future dates only
        forecast_filtered = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        # Ensure 'ds' is date only for comparison if needed, though datetime is fine
        forecast_filtered['ds'] = pd.to_datetime(forecast_filtered['ds']).dt.tz_localize(None) # Remove timezone if present

        # Keep only future predictions (after the last date in historical data)
        last_historical_date = df_prepared['ds'].max()
        forecast_future = forecast_filtered[forecast_filtered['ds'] > last_historical_date].reset_index(drop=True)

        # Ensure predicted values ('yhat') don't go below zero (inventory can't be negative)
        forecast_future['yhat'] = forecast_future['yhat'].clip(lower=0)
        forecast_future['yhat_lower'] = forecast_future['yhat_lower'].clip(lower=0)
        forecast_future['yhat_upper'] = forecast_future['yhat_upper'].clip(lower=0)

        # Optional: Round predictions to nearest integer if quantities must be whole numbers
        # forecast_future[['yhat', 'yhat_lower', 'yhat_upper']] = forecast_future[['yhat', 'yhat_lower', 'yhat_upper']].round().astype(int)

        return forecast_future

    except Exception as e:
        st.error(f"Error generating forecast for item {item_id}: {e}", icon="🚨")
        return pd.DataFrame()
