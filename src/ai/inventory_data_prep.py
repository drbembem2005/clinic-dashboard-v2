# src/ai/inventory_data_prep.py
import pandas as pd
import streamlit as st
from src import data_loader
from datetime import timedelta

@st.cache_data(ttl=3600) # Cache the preprocessed data for an hour
def get_preprocessed_item_history(item_id: int, item_name: str = "Selected Item"):
    """
    Fetches inventory log data for a specific item, preprocesses it for time-series forecasting.

    Args:
        item_id (int): The ID of the inventory item.
        item_name (str): The name of the item (for error messages).

    Returns:
        pd.DataFrame: Preprocessed DataFrame with columns 'ds' (datetime) and 'y' (quantity),
                      or an empty DataFrame if insufficient data or error.
    """
    df_log = data_loader.get_inventory_log(item_id)

    if df_log.empty:
        st.warning(f"No historical log data found for item ID {item_id} ({item_name}). Cannot generate forecast.", icon="⚠️")
        return pd.DataFrame()

    # Ensure correct data types and sort
    df_log['timestamp'] = pd.to_datetime(df_log['timestamp'])
    df_log['new_quantity'] = pd.to_numeric(df_log['new_quantity'], errors='coerce')
    df_log = df_log.dropna(subset=['timestamp', 'new_quantity'])
    df_log = df_log.sort_values(by='timestamp')

    if df_log.empty:
        st.warning(f"Log data for item ID {item_id} ({item_name}) is invalid after cleaning. Cannot generate forecast.", icon="⚠️")
        return pd.DataFrame()

    # Set timestamp as index for resampling
    df_log = df_log.set_index('timestamp')

    # Resample to daily frequency, taking the last known quantity of the day
    # This represents the closing stock for the day
    df_daily = df_log['new_quantity'].resample('D').last()

    # Forward fill missing values (days with no log entries inherit the previous day's closing stock)
    df_daily = df_daily.ffill()

    # Reset index to get 'ds' column, rename columns for Prophet compatibility
    df_prepared = df_daily.reset_index()
    df_prepared.columns = ['ds', 'y']

    # Ensure 'ds' is datetime
    df_prepared['ds'] = pd.to_datetime(df_prepared['ds'])

    # Check if we have enough data points for forecasting (e.g., at least 2)
    if len(df_prepared) < 2:
        st.warning(f"Insufficient historical data points ({len(df_prepared)}) for item ID {item_id} ({item_name}) after resampling. Need at least 2 days of data.", icon="⚠️")
        return pd.DataFrame()

    # Optional: Check for constant value (Prophet might struggle)
    if df_prepared['y'].nunique() == 1:
         st.info(f"Inventory level for item ID {item_id} ({item_name}) has remained constant. Forecast will reflect this.", icon="ℹ️")
         # We can still proceed, Prophet will predict the constant value

    return df_prepared
