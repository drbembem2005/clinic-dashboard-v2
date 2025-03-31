# src/data_loader.py
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, date, time # Added date, time
import sqlite3 # Added for SQLite
import uuid # Added for generating unique IDs
import os # Added for constructing DB path

@st.cache_data(ttl=3600)
def load_data():
    """
    Loads and preprocesses data from the clinic_financial_ai.xlsx file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    try:
        file_path = "clinic_financial_ai.xlsx"
        xls = pd.ExcelFile(file_path)
        df_data = pd.read_excel(xls, sheet_name="data")
    except FileNotFoundError:
        st.error(f"Error: The file {file_path} was not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {e}")
        st.stop()

    # --- Data Preprocessing ---
    # Convert date column
    df_data["date"] = pd.to_datetime(df_data["date"], errors='coerce')
    df_data.dropna(subset=["date"], inplace=True) # Remove rows where date conversion failed

    # Fill missing numeric values with 0 - Be cautious with this approach
    numeric_cols = df_data.select_dtypes(include=np.number).columns
    df_data[numeric_cols] = df_data[numeric_cols].fillna(0)

    # --- Commission Calculations ---
    # Ensure commission columns exist and are numeric
    commission_cols = [
        "doctor comission payed", "com to be payed", "T.doc.com", "com pay", "gross income"
    ]
    for col in commission_cols:
        if col not in df_data.columns:
            st.warning(f"Column '{col}' not found in data. Filling with 0.")
            df_data[col] = 0
        else:
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce').fillna(0)

    # If "com to be payed" has value, doctor receives monthly commission only
    df_data["commission_paid_daily"] = df_data["doctor comission payed"]  # Daily commission for doctors
    df_data["commission_paid_monthly"] = df_data["com to be payed"]  # Monthly commission for doctors
    df_data["total_commission"] = df_data["T.doc.com"] # Total doctor commission

    # Handle advertising company commissions
    df_data["advertising_commission"] = df_data["com pay"]

    # --- Profit Calculation ---
    # Ensure 'profit' column exists or calculate it
    if 'profit' not in df_data.columns:
        df_data["total_deductions"] = df_data["total_commission"] + df_data["advertising_commission"]
        df_data["profit"] = df_data["gross income"] - df_data["total_deductions"]
    else:
        # Ensure profit is numeric if it exists
        df_data['profit'] = pd.to_numeric(df_data['profit'], errors='coerce').fillna(0)


    # Avoid division by zero for profit margin
    df_data["profit_margin_pct"] = np.where(
        df_data["gross income"] > 0,
        (df_data["profit"] / df_data["gross income"]) * 100,
        0
    )

    # --- Derived Date/Time Columns ---
    df_data["month"] = df_data["date"].dt.month
    df_data["month_name"] = df_data["date"].dt.month_name()
    df_data["month_year"] = df_data["date"].dt.strftime('%Y-%m')
    df_data["day"] = df_data["date"].dt.day
    df_data["day_of_week"] = df_data["date"].dt.day_name()
    # Use .astype(int) after ensuring the Series contains only valid integer representations or NaNs handled
    df_data["week"] = df_data["date"].dt.isocalendar().week.fillna(0).astype(int) # Fill NA before converting
    df_data["hour"] = df_data["date"].dt.hour

    # --- Categorical Columns ---
    # Payment method categorization
    df_data["payment_method"] = np.where(df_data["cash pay"] > 0, "Cash",
                                np.where(df_data["visa pay"] > 0, "Visa", "Other"))

    # Ensure 'visit type' exists
    if "visit type" not in df_data.columns:
        st.warning("Column 'visit type' not found. Using 'Unknown'.")
        df_data["visit type"] = "Unknown"
    else:
        df_data["visit type"] = df_data["visit type"].fillna("Unknown").astype(str)

    # Ensure 'Doctor' exists
    if "Doctor" not in df_data.columns:
        st.warning("Column 'Doctor' not found. Using 'Unknown'.")
        df_data["Doctor"] = "Unknown"
    else:
        df_data["Doctor"] = df_data["Doctor"].fillna("Unknown").astype(str)

    # Ensure 'Patient' exists
    if "Patient" not in df_data.columns:
        st.warning("Column 'Patient' not found. Using 'Unknown'.")
        df_data["Patient"] = "Unknown" # Or generate unique IDs if needed
    else:
        # Ensure Patient IDs are consistent, maybe string type
        df_data["Patient"] = df_data["Patient"].fillna("Unknown").astype(str)

    # --- Simulated Data (Placeholder) ---
    # Calculate visit duration (random for demonstration - replace if real data exists)
    if "visit_duration_mins" not in df_data.columns:
        np.random.seed(42)
        df_data["visit_duration_mins"] = np.random.randint(15, 60, size=len(df_data))

    # --- Final Checks ---
    # Ensure essential columns are present
    essential_cols = ["date", "gross income", "profit", "Doctor", "Patient", "visit type", "id"]
    for col in essential_cols:
        if col not in df_data.columns:
            # If 'id' is missing, try to create a unique identifier
            if col == 'id':
                st.warning("Column 'id' not found. Creating a unique ID from index.")
                df_data['id'] = df_data.index.astype(str)
            # Skip profit check here as we calculate it above if missing
            elif col != 'profit':
                st.error(f"Essential column '{col}' is missing after preprocessing. Stopping.")
                st.stop()

    # Convert 'id' to string if it's potentially numeric to avoid issues
    if 'id' in df_data.columns and pd.api.types.is_numeric_dtype(df_data['id']):
        df_data['id'] = df_data['id'].astype(str)


    return df_data


# --- Appointment Data Handling (SQLite) ---

APPOINTMENTS_DB_PATH = "clinic_appointments.db"

def connect_db():
    """Connects to the SQLite database."""
    conn = sqlite3.connect(APPOINTMENTS_DB_PATH)
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    return conn

def create_appointments_table():
    """Creates the appointments table if it doesn't exist."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            AppointmentID TEXT PRIMARY KEY,
            PatientName TEXT NOT NULL,
            DoctorName TEXT NOT NULL,
            AppointmentDateTime TEXT NOT NULL, -- Store as ISO format string
            AppointmentType TEXT,
            AppointmentStatus TEXT DEFAULT 'Scheduled',
            BookingDateTime TEXT, -- Store as ISO format string
            CancellationDateTime TEXT, -- Store as ISO format string
            ConfirmationStatus TEXT,
            ReminderType TEXT,
            BookingChannel TEXT, -- Added
            ReferralSource TEXT, -- Added
            PatientArrivalTime TEXT, -- Added - Store as ISO time string HH:MM:SS
            AppointmentStartTime TEXT, -- Added - Store as ISO time string HH:MM:SS
            AppointmentEndTime TEXT -- Added - Store as ISO time string HH:MM:SS
        )
    """)
    conn.commit()
    conn.close()

# Ensure the table exists when the module is loaded and update schema if needed
# This simple check might not handle all schema migration scenarios robustly
def update_schema():
    conn = connect_db()
    cursor = conn.cursor()
    try:
        # Check for one of the new columns
        cursor.execute("PRAGMA table_info(appointments)")
        columns = [info['name'] for info in cursor.fetchall()]
        if 'BookingChannel' not in columns:
            cursor.execute("ALTER TABLE appointments ADD COLUMN BookingChannel TEXT")
        if 'ReferralSource' not in columns:
            cursor.execute("ALTER TABLE appointments ADD COLUMN ReferralSource TEXT")
        if 'PatientArrivalTime' not in columns:
            cursor.execute("ALTER TABLE appointments ADD COLUMN PatientArrivalTime TEXT")
        if 'AppointmentStartTime' not in columns:
            cursor.execute("ALTER TABLE appointments ADD COLUMN AppointmentStartTime TEXT")
        if 'AppointmentEndTime' not in columns:
            cursor.execute("ALTER TABLE appointments ADD COLUMN AppointmentEndTime TEXT")
        conn.commit()
    except sqlite3.Error as e:
        st.warning(f"Could not update table schema: {e}") # Warn instead of error
    finally:
        conn.close()


# --- Inventory Reporting Functions ---

def get_consumption_data(start_date, end_date):
    """
    Calculates inventory consumption within a date range based on log entries.

    Args:
        start_date (date): The start date of the period.
        end_date (date): The end date of the period.

    Returns:
        pd.DataFrame: DataFrame with item_name, category, and total_consumed quantity.
                      Returns empty DataFrame on error or if no consumption.
    """
    conn = connect_db()
    cursor = conn.cursor()
    query = """
        SELECT
            i.item_name,
            i.category,
            SUM(CASE WHEN il.quantity_change < 0 THEN ABS(il.quantity_change) ELSE 0 END) as total_consumed
        FROM inventory_log il
        JOIN inventory i ON il.item_id = i.id
        WHERE
            il.timestamp >= ? AND il.timestamp <= ?
            AND il.quantity_change < 0 -- Only count negative changes (usage/adjustments down)
            AND il.change_type != 'Item Deleted' -- Exclude deletions from consumption
        GROUP BY
            i.id, i.item_name, i.category
        HAVING
            total_consumed > 0
        ORDER BY
            i.category, i.item_name;
    """
    # Convert dates to datetime objects at the beginning and end of the day for timestamp comparison
    start_dt_str = datetime.combine(start_date, time.min).isoformat()
    end_dt_str = datetime.combine(end_date, time.max).isoformat()

    try:
        cursor.execute(query, (start_dt_str, end_dt_str))
        consumption_data = cursor.fetchall()
        df_consumption = pd.DataFrame([dict(row) for row in consumption_data])
        return df_consumption
    except sqlite3.Error as e:
        st.error(f"Database error getting consumption data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_expired_stock_data(reference_date, days_ahead=30):
    """
    Retrieves items that are expired or expiring soon, calculating their value.

    Args:
        reference_date (date): The date to compare expiration dates against (usually today).
        days_ahead (int): The number of days into the future to check for expiring items.

    Returns:
        pd.DataFrame: DataFrame with item details, expiration status, and estimated value.
                      Returns empty DataFrame on error or if no relevant items.
    """
    conn = connect_db()
    cursor = conn.cursor()
    target_date = reference_date + timedelta(days=days_ahead)
    reference_date_str = reference_date.isoformat()
    target_date_str = target_date.isoformat()

    query = """
        SELECT
            id,
            item_name,
            category,
            current_quantity,
            unit_cost,
            expiration_date,
            (current_quantity * COALESCE(unit_cost, 0)) as estimated_value,
            CASE
                WHEN expiration_date < ? THEN 'Already Expired'
                WHEN expiration_date >= ? AND expiration_date <= ? THEN 'Expiring Soon'
                ELSE 'Other' -- Should not happen with WHERE clause, but for safety
            END as expiration_status
        FROM inventory
        WHERE
            expiration_date IS NOT NULL
            AND expiration_date <= ? -- Only include items expiring before the target date
        ORDER BY
            expiration_date ASC;
    """
    try:
        cursor.execute(query, (reference_date_str, reference_date_str, target_date_str, target_date_str))
        expired_data = cursor.fetchall()
        df_expired = pd.DataFrame([dict(row) for row in expired_data])
        if not df_expired.empty:
             df_expired['expiration_date'] = pd.to_datetime(df_expired['expiration_date']).dt.date
        return df_expired
    except sqlite3.Error as e:
        st.error(f"Database error getting expired stock data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_slow_moving_items(days_threshold):
    """
    Identifies items that haven't had a log entry within the specified threshold.

    Args:
        days_threshold (int): The number of days to look back. Items with no log
                              entry within this period are considered slow-moving.

    Returns:
        pd.DataFrame: DataFrame with details of slow-moving items.
                      Returns empty DataFrame on error or if no slow-moving items.
    """
    conn = connect_db()
    cursor = conn.cursor()
    threshold_date = datetime.now() - timedelta(days=days_threshold)
    threshold_date_str = threshold_date.isoformat()

    # Find the last log timestamp for each item
    # Then join with inventory and filter items whose last log is before the threshold
    query = """
        WITH LastLog AS (
            SELECT
                item_id,
                MAX(timestamp) as last_log_timestamp
            FROM inventory_log
            GROUP BY item_id
        )
        SELECT
            i.id,
            i.item_name,
            i.category,
            i.current_quantity,
            i.last_updated, -- Last update time on the item itself
            ll.last_log_timestamp
        FROM inventory i
        LEFT JOIN LastLog ll ON i.id = ll.item_id
        WHERE
            -- Include items never logged OR logged before the threshold
            (ll.last_log_timestamp IS NULL OR ll.last_log_timestamp < ?)
            AND i.current_quantity > 0 -- Optional: Only show if there's stock
        ORDER BY
            ll.last_log_timestamp ASC, -- Show oldest first
            i.item_name ASC;
    """
    try:
        cursor.execute(query, (threshold_date_str,))
        slow_items_data = cursor.fetchall()
        df_slow_items = pd.DataFrame([dict(row) for row in slow_items_data])
        if not df_slow_items.empty:
             df_slow_items['last_updated'] = pd.to_datetime(df_slow_items['last_updated'])
             df_slow_items['last_log_timestamp'] = pd.to_datetime(df_slow_items['last_log_timestamp'])
        return df_slow_items
    except sqlite3.Error as e:
        st.error(f"Database error getting slow-moving items: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

create_appointments_table()
update_schema() # Attempt to update schema after ensuring table exists

def add_appointment(patient_name, doctor_name, appointment_datetime, appointment_type,
                    booking_datetime=None, confirmation_status=None, reminder_type=None,
                    booking_channel=None, referral_source=None, patient_arrival_time=None,
                    appointment_start_time=None, appointment_end_time=None): # Added new params
    """Adds a new appointment to the database."""
    conn = connect_db()
    cursor = conn.cursor()
    appointment_id = str(uuid.uuid4()) # Generate a unique ID
    booking_dt_str = booking_datetime.isoformat() if booking_datetime else None
    appointment_dt_str = appointment_datetime.isoformat()
    # Convert time objects to ISO strings
    arrival_time_str = patient_arrival_time.isoformat() if isinstance(patient_arrival_time, time) else patient_arrival_time
    start_time_str = appointment_start_time.isoformat() if isinstance(appointment_start_time, time) else appointment_start_time
    end_time_str = appointment_end_time.isoformat() if isinstance(appointment_end_time, time) else appointment_end_time


    try:
        cursor.execute("""
            INSERT INTO appointments (
                AppointmentID, PatientName, DoctorName, AppointmentDateTime, AppointmentType,
                AppointmentStatus, BookingDateTime, ConfirmationStatus, ReminderType,
                BookingChannel, ReferralSource, PatientArrivalTime, AppointmentStartTime, AppointmentEndTime
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            appointment_id, patient_name, doctor_name, appointment_dt_str, appointment_type,
            'Scheduled', booking_dt_str, confirmation_status, reminder_type,
            booking_channel, referral_source, arrival_time_str, start_time_str, end_time_str
        ))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Database error adding appointment: {e}")
        return False
    finally:
        conn.close()

def get_appointments(start_date_filter=None, end_date_filter=None, doctor_filter=None, status_filter=None):
    """Retrieves appointments from the database, optionally filtered."""
    conn = connect_db()
    cursor = conn.cursor()
    query = "SELECT * FROM appointments"
    filters = []
    params = []

    if start_date_filter:
        # Assuming start_date_filter is a date object
        start_dt_str = datetime.combine(start_date_filter, time.min).isoformat()
        filters.append("AppointmentDateTime >= ?")
        params.append(start_dt_str)
    if end_date_filter:
        # Assuming end_date_filter is a date object
        end_dt_str = datetime.combine(end_date_filter, time.max).isoformat()
        filters.append("AppointmentDateTime <= ?")
        params.append(end_dt_str)
    if doctor_filter and doctor_filter != "All":
        filters.append("DoctorName = ?")
        params.append(doctor_filter)
    if status_filter and status_filter != "All":
        filters.append("AppointmentStatus = ?")
        params.append(status_filter)

    if filters:
        query += " WHERE " + " AND ".join(filters)

    query += " ORDER BY AppointmentDateTime ASC" # Order by date/time

    try:
        cursor.execute(query, params)
        appointments = cursor.fetchall()
        # Convert to DataFrame
        df_appointments = pd.DataFrame([dict(row) for row in appointments])
        # Convert datetime columns back from string
        if not df_appointments.empty:
            for col in ['AppointmentDateTime', 'BookingDateTime', 'CancellationDateTime']:
                if col in df_appointments.columns:
                    df_appointments[col] = pd.to_datetime(df_appointments[col], errors='coerce')
        return df_appointments
    except sqlite3.Error as e:
        st.error(f"Database error getting appointments: {e}")
        return pd.DataFrame() # Return empty DataFrame on error
    finally:
        conn.close()


def update_appointment(appointment_id, updates):
    """Updates an existing appointment."""
    conn = connect_db()
    cursor = conn.cursor()
    set_clauses = []
    params = []
    current_time_str = datetime.now().isoformat()

    # Handle Cancellation Time automatically if status is set to Cancelled
    if updates.get('AppointmentStatus') == 'Cancelled' and 'CancellationDateTime' not in updates:
        updates['CancellationDateTime'] = current_time_str
    # Clear cancellation time if status is changed away from Cancelled
    elif 'AppointmentStatus' in updates and updates['AppointmentStatus'] != 'Cancelled':
         updates['CancellationDateTime'] = None # Set to NULL in DB

    # Convert datetime/time objects to ISO strings for storage
    if 'AppointmentDateTime' in updates and isinstance(updates['AppointmentDateTime'], datetime):
        updates['AppointmentDateTime'] = updates['AppointmentDateTime'].isoformat()
    if 'BookingDateTime' in updates and isinstance(updates['BookingDateTime'], datetime):
        updates['BookingDateTime'] = updates['BookingDateTime'].isoformat()
    # CancellationDateTime is handled above or passed as string
    if 'PatientArrivalTime' in updates and isinstance(updates['PatientArrivalTime'], time):
        updates['PatientArrivalTime'] = updates['PatientArrivalTime'].isoformat()
    if 'AppointmentStartTime' in updates and isinstance(updates['AppointmentStartTime'], time):
        updates['AppointmentStartTime'] = updates['AppointmentStartTime'].isoformat()
    if 'AppointmentEndTime' in updates and isinstance(updates['AppointmentEndTime'], time):
        updates['AppointmentEndTime'] = updates['AppointmentEndTime'].isoformat()


    for key, value in updates.items():
        # Ensure we only try to update valid columns
        valid_columns = [
            "PatientName", "DoctorName", "AppointmentDateTime", "AppointmentType",
            "AppointmentStatus", "BookingDateTime", "CancellationDateTime",
            "ConfirmationStatus", "ReminderType", "BookingChannel", "ReferralSource",
            "PatientArrivalTime", "AppointmentStartTime", "AppointmentEndTime"
        ]
        if key in valid_columns:
            set_clauses.append(f"{key} = ?")
            params.append(value)

    if not set_clauses:
        st.warning("No valid fields provided for update.")
        conn.close()
        return False

    params.append(appointment_id) # For the WHERE clause
    query = f"UPDATE appointments SET {', '.join(set_clauses)} WHERE AppointmentID = ?"

    try:
        cursor.execute(query, params)
        conn.commit()
        return cursor.rowcount > 0 # Return True if a row was updated
    except sqlite3.Error as e:
        st.error(f"Database error updating appointment {appointment_id}: {e}")
        return False
    finally:
        conn.close()

def delete_appointment(appointment_id):
    """Deletes an appointment from the database."""
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM appointments WHERE AppointmentID = ?", (appointment_id,))
        conn.commit()
        return cursor.rowcount > 0 # Return True if a row was deleted
    except sqlite3.Error as e:
        st.error(f"Database error deleting appointment {appointment_id}: {e}")
        return False
    finally:
        conn.close()

# Modified to get doctors from the main financial DataFrame
def get_distinct_doctors(df_financial_data):
    """Gets a list of distinct doctor names from the financial data DataFrame."""
    if df_financial_data is not None and "Doctor" in df_financial_data.columns:
        try:
            # Get unique, non-null doctor names and sort them
            doctors = df_financial_data["Doctor"].dropna().unique().tolist()
            doctors.sort()
            return doctors
        except Exception as e:
            st.error(f"Error getting distinct doctors from DataFrame: {e}")
            return []
    else:
        st.warning("Financial data or 'Doctor' column not available for fetching doctor list.")
        return []


def get_distinct_patients():
    """Gets a list of distinct patient names from the appointments table."""
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT PatientName FROM appointments ORDER BY PatientName")
        patients = [row['PatientName'] for row in cursor.fetchall()]
        return patients
    except sqlite3.Error as e:
        st.error(f"Database error getting distinct patients: {e}")
        return []
    finally:
        conn.close()


# --- Cost Data Handling (SQLite) ---

def create_costs_table():
    """Creates the costs table if it doesn't exist."""
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expense_date TEXT NOT NULL, -- Renamed from entry_date (Date cost incurred)
                payment_date TEXT,          -- Date cost was paid (can be NULL if not yet paid)
                category TEXT NOT NULL,
                item TEXT NOT NULL,
                amount REAL NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit() # Commit table creation first

        # Now, attempt to create indexes separately
        try:
            cursor.execute("DROP INDEX IF EXISTS idx_costs_entry_date") # Drop old index if exists
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_costs_expense_date ON costs (expense_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_costs_payment_date ON costs (payment_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_costs_category ON costs (category)")
            conn.commit() # Commit index creation
        except sqlite3.Error as e_index:
            # Warn if index creation fails, but don't stop the app
            st.warning(f"Database warning creating indexes for costs table: {e_index}")

    except sqlite3.Error as e_table:
        st.error(f"Database error creating costs table: {e_table}")
    finally:
        conn.close()

def update_costs_schema():
    """Adds new columns to the costs table if they don't exist."""
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(costs)")
        columns = [info['name'] for info in cursor.fetchall()]
        if 'expense_date' not in columns:
            # If expense_date doesn't exist, assume entry_date might, try renaming first
            try:
                cursor.execute("ALTER TABLE costs RENAME COLUMN entry_date TO expense_date")
                st.info("Renamed 'entry_date' column to 'expense_date' in costs table.")
            except sqlite3.Error:
                # If rename fails (e.g., entry_date also doesn't exist), add expense_date
                cursor.execute("ALTER TABLE costs ADD COLUMN expense_date TEXT")
                st.info("Added 'expense_date' column to costs table.")
        if 'payment_date' not in columns:
            cursor.execute("ALTER TABLE costs ADD COLUMN payment_date TEXT")
            st.info("Added 'payment_date' column to costs table.")
        conn.commit()
    except sqlite3.Error as e:
        st.warning(f"Could not update costs table schema: {e}")
    finally:
        conn.close()

# Ensure the costs table exists and schema is updated when the module is loaded
create_costs_table()
update_costs_schema() # Call the new schema update function

def add_cost(expense_date, payment_date, category, item, amount):
    """Adds a new cost entry to the database."""
    conn = connect_db()
    cursor = conn.cursor()
    # Ensure dates are stored in YYYY-MM-DD format
    expense_date_str = expense_date.isoformat() if isinstance(expense_date, date) else str(expense_date)
    # Payment date can be None
    payment_date_str = payment_date.isoformat() if isinstance(payment_date, date) else None
    try:
        cursor.execute("""
            INSERT INTO costs (expense_date, payment_date, category, item, amount)
            VALUES (?, ?, ?, ?, ?)
        """, (expense_date_str, payment_date_str, category, item, amount))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Database error adding cost: {e}")
        return False
    finally:
        conn.close()

def get_costs(start_date_filter=None, end_date_filter=None, date_column='expense_date'):
    """
    Retrieves cost entries from the database, optionally filtered by a specified date column.

    Args:
        start_date_filter (date, optional): Start date for filtering.
        end_date_filter (date, optional): End date for filtering.
        date_column (str, optional): The date column to filter on ('expense_date' or 'payment_date').
                                     Defaults to 'expense_date'.
    """
    conn = connect_db()
    cursor = conn.cursor()
    # Select all relevant columns
    query = "SELECT id, expense_date, payment_date, category, item, amount, recorded_at FROM costs"
    filters = []
    params = []

    # Validate date_column input
    if date_column not in ['expense_date', 'payment_date']:
        st.warning(f"Invalid date column '{date_column}' specified for filtering costs. Defaulting to 'expense_date'.")
        date_column = 'expense_date'

    if start_date_filter:
        start_date_str = start_date_filter.isoformat() if isinstance(start_date_filter, date) else str(start_date_filter)
        filters.append(f"{date_column} >= ?")
        params.append(start_date_str)
    if end_date_filter:
        end_date_str = end_date_filter.isoformat() if isinstance(end_date_filter, date) else str(end_date_filter)
        filters.append(f"{date_column} <= ?")
        params.append(end_date_str)

    # Ensure filtering by payment_date doesn't exclude unpaid items unless intended
    # If filtering by payment_date, we might only want rows where payment_date is not NULL
    # However, the current logic includes NULLs if they fall outside the date range, which might be okay.
    # Add this if you only want paid items within the range when filtering by payment_date:
    # if date_column == 'payment_date':
    #     filters.append("payment_date IS NOT NULL")


    if filters:
        query += " WHERE " + " AND ".join(filters)

    query += f" ORDER BY {date_column} DESC, recorded_at DESC" # Order by the filtered date column

    try:
        cursor.execute(query, params)
        costs = cursor.fetchall()
        # Convert to DataFrame
        df_costs = pd.DataFrame([dict(row) for row in costs])
        # Convert date columns back from string
        if not df_costs.empty:
            df_costs['expense_date'] = pd.to_datetime(df_costs['expense_date'], errors='coerce').dt.date
            # payment_date might be NaT if NULL in DB
            df_costs['payment_date'] = pd.to_datetime(df_costs['payment_date'], errors='coerce').dt.date
            df_costs['recorded_at'] = pd.to_datetime(df_costs['recorded_at'], errors='coerce')
        return df_costs
    except sqlite3.Error as e:
        st.error(f"Database error getting costs: {e}")
        return pd.DataFrame() # Return empty DataFrame on error
    finally:
        conn.close()

# --- Goal Setting Data Handling (SQLite) ---

def create_goals_table():
    """Creates the goals table if it doesn't exist."""
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                target_value REAL NOT NULL,
                time_period TEXT NOT NULL, -- 'Monthly', 'Quarterly', 'Yearly', 'Custom Range'
                start_date TEXT,          -- YYYY-MM-DD, only for 'Custom Range'
                end_date TEXT,            -- YYYY-MM-DD, only for 'Custom Range'
                is_active INTEGER DEFAULT 1, -- 1 for active, 0 for inactive
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        # Add indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_goals_metric_name ON goals (metric_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_goals_is_active ON goals (is_active)")
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error creating goals table: {e}")
    finally:
        conn.close()

def update_goals_schema():
    """Adds new columns to the goals table if they don't exist (for future use)."""
    # Placeholder for future schema changes if needed
    pass

# Ensure the goals table exists and schema is updated
create_goals_table()
update_goals_schema()


# --- Inventory Management Data Handling (SQLite) ---

def create_inventory_table():
    """Creates the inventory table if it doesn't exist."""
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name TEXT NOT NULL UNIQUE, -- Ensure item names are unique
                category TEXT,
                current_quantity INTEGER NOT NULL DEFAULT 0,
                reorder_level INTEGER DEFAULT 0,
                max_stock_level INTEGER, -- Added for overstock alerts
                unit_cost REAL,
                expiration_date TEXT, -- Store as YYYY-MM-DD
                supplier TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        # Add indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inventory_item_name ON inventory (item_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inventory_category ON inventory (category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inventory_expiration_date ON inventory (expiration_date)")
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error creating inventory table: {e}")
    finally:
        conn.close()

def update_inventory_schema():
    """Adds new columns to the inventory table if they don't exist (for future use)."""
    # Placeholder for future schema changes if needed
    conn = connect_db()
    # cursor = conn.cursor()
    # try:
    #     cursor.execute("PRAGMA table_info(inventory)")
    #     columns = [info['name'] for info in cursor.fetchall()]
    #     if 'new_column' not in columns:
    #         cursor.execute("ALTER TABLE inventory ADD COLUMN new_column TEXT")
    #         st.info("Added 'new_column' to inventory table.")
    #     conn.commit()
    # except sqlite3.Error as e:
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(inventory)")
        columns = [info['name'] for info in cursor.fetchall()]
        if 'max_stock_level' not in columns:
            cursor.execute("ALTER TABLE inventory ADD COLUMN max_stock_level INTEGER")
            st.info("Added 'max_stock_level' column to inventory table.")
        conn.commit()
    except sqlite3.Error as e:
        st.warning(f"Could not update inventory table schema: {e}")
    finally:
        conn.close()
    # pass # Removed pass as we added actual logic

# --- Initial Inventory Population ---
DEFAULT_INVENTORY_ITEMS = {
    "Consumables": [
        "Gauze Pads (Sterile, 4x4)", "Alcohol Wipes", "Cotton Balls", "Bandages (Assorted Sizes)",
        "Medical Tape", "Tongue Depressors", "Exam Table Paper", "Syringes (1ml, 3ml, 5ml)",
        "Needles (Assorted Gauges)", "Butterfly Needles", "Vacutainer Tubes (Red, Lavender, Blue)",
        "Urine Collection Cups", "Specimen Bags"
    ],
    "PPE (Personal Protective Equipment)": [
        "Disposable Gloves (Nitrile, S/M/L)", "Face Masks (Surgical)", "Face Shields", "Isolation Gowns"
    ],
    "Sterilization & Disinfection": [
        "Autoclave Pouches", "Surface Disinfectant Wipes", "Hand Sanitizer", "Sharps Containers"
    ],
    "Medications (Common - Placeholder)": [
        "Saline Solution (Sterile)", "Lidocaine (Injectable)", "Pain Relievers (OTC Sample)", "Antibiotic Ointment"
    ],
    "Instruments (Basic)": [
        "Stethoscope", "Blood Pressure Cuff (Manual/Digital)", "Thermometer (Digital/Temporal)", "Otoscope", "Ophthalmoscope"
    ],
    "Office Supplies": [
        "Printer Paper", "Pens", "Clipboards", "Appointment Cards"
    ]
}

def populate_initial_inventory():
    """Adds default inventory items if they don't exist."""
    conn = connect_db()
    cursor = conn.cursor()
    added_count = 0
    try:
        cursor.execute("SELECT item_name FROM inventory")
        existing_items = {row['item_name'] for row in cursor.fetchall()}

        items_to_add = []
        for category, items in DEFAULT_INVENTORY_ITEMS.items():
            for item_name in items:
                if item_name not in existing_items:
                    # Add with default values (0 quantity)
                    items_to_add.append((item_name, category, 0, 0, None, None, None, None)) # Matches add_inventory_item structure minus ID/timestamp

        if items_to_add:
            cursor.executemany("""
                INSERT INTO inventory (item_name, category, current_quantity, reorder_level, max_stock_level, unit_cost, expiration_date, supplier, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, items_to_add)
            conn.commit()
            added_count = len(items_to_add)
            # Use st.toast for less intrusive feedback if Streamlit version supports it
            # st.toast(f"Added {added_count} default inventory items.", icon="📦")
            print(f"INFO: Added {added_count} default inventory items.") # Print to console as fallback

            # Log the addition of these items (optional, could be noisy)
            # For simplicity, we might skip logging each default item addition here
            # Or log them with a specific type like 'System Initialized'

    except sqlite3.Error as e:
        st.warning(f"Database error populating initial inventory: {e}")
    finally:
        conn.close()
    # We don't strictly need to return anything here unless we act on it
    # return added_count > 0 # Return True if any items were added

# Ensure the inventory table exists and schema is updated
create_inventory_table()
update_inventory_schema() # Call the updated schema function
populate_initial_inventory() # Add default items if needed


# --- Inventory Log Handling ---

def create_inventory_log_table():
    """Creates the inventory_log table if it doesn't exist."""
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL, -- Foreign key to inventory table id
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                change_type TEXT NOT NULL, -- e.g., 'Initial Stock', 'Quantity Update', 'Item Added', 'Item Deleted'
                quantity_change INTEGER, -- Can be positive or negative
                new_quantity INTEGER,
                notes TEXT, -- Optional notes about the change
                FOREIGN KEY (item_id) REFERENCES inventory (id) ON DELETE CASCADE -- Optional: Cascade delete logs if item is deleted
            )
        """)
        conn.commit()
        # Add index
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inventory_log_item_id ON inventory_log (item_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inventory_log_timestamp ON inventory_log (timestamp)")
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error creating inventory_log table: {e}")
    finally:
        conn.close()

# Ensure the log table exists
create_inventory_log_table()

def log_inventory_change(item_id, change_type, quantity_change, new_quantity, notes=None):
    """Logs a change to the inventory_log table."""
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO inventory_log (item_id, change_type, quantity_change, new_quantity, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (item_id, change_type, quantity_change, new_quantity, notes))
        conn.commit()
    except sqlite3.Error as e:
        # Log error but don't stop the main operation
        st.warning(f"Database warning: Could not log inventory change for item ID {item_id}: {e}")
    finally:
        conn.close()


def add_inventory_item(item_name, category=None, current_quantity=0, reorder_level=0, max_stock_level=None, unit_cost=None, expiration_date=None, supplier=None):
    """Adds a new item to the inventory."""
    conn = connect_db()
    cursor = conn.cursor()
    exp_date_str = expiration_date.isoformat() if isinstance(expiration_date, date) else None
    try:
        cursor.execute("""
            INSERT INTO inventory (item_name, category, current_quantity, reorder_level, max_stock_level, unit_cost, expiration_date, supplier, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (item_name, category, current_quantity, reorder_level, max_stock_level, unit_cost, exp_date_str, supplier))
        item_id = cursor.lastrowid # Get the ID of the newly inserted item
        conn.commit()
        # Log the initial stock
        log_inventory_change(item_id, 'Item Added / Initial Stock', current_quantity, current_quantity, f"Item '{item_name}' created.")
        return True
    except sqlite3.IntegrityError:
         st.error(f"Database error: Inventory item '{item_name}' already exists.")
         return False
    except sqlite3.Error as e:
        st.error(f"Database error adding inventory item: {e}")
        return False
    finally:
        conn.close()

def get_inventory_items(filter_category=None, low_stock_only=False, expiring_soon_days=None):
    """Retrieves inventory items, optionally filtered."""
    conn = connect_db()
    cursor = conn.cursor()
    query = "SELECT * FROM inventory"
    filters = []
    params = []

    if filter_category and filter_category != "All":
        filters.append("category = ?")
        params.append(filter_category)

    if low_stock_only:
        # Assumes items with quantity <= reorder_level are low stock
        filters.append("current_quantity <= reorder_level")

    if expiring_soon_days is not None:
        try:
            days = int(expiring_soon_days)
            today = date.today()
            target_date = today + timedelta(days=days)
            target_date_str = target_date.isoformat()
            today_str = today.isoformat()
            # Filter for items expiring between today and the target date
            filters.append("expiration_date IS NOT NULL AND expiration_date >= ? AND expiration_date <= ?")
            params.extend([today_str, target_date_str])
        except ValueError:
            st.warning("Invalid number of days provided for expiration filter.")


    if filters:
        query += " WHERE " + " AND ".join(filters)

    query += " ORDER BY item_name ASC"

    try:
        cursor.execute(query, params)
        items = cursor.fetchall()
        df_inventory = pd.DataFrame([dict(row) for row in items])
        # Convert date/time columns
        if not df_inventory.empty:
            df_inventory['expiration_date'] = pd.to_datetime(df_inventory['expiration_date'], errors='coerce').dt.date
            df_inventory['last_updated'] = pd.to_datetime(df_inventory['last_updated'], errors='coerce')
        return df_inventory
    except sqlite3.Error as e:
        st.error(f"Database error getting inventory items: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def update_inventory_item(item_id, updates, change_type_override=None):
    """
    Updates an existing inventory item.

    Args:
        item_id (int): The ID of the item to update.
        updates (dict): Dictionary of column names and new values.
        change_type_override (str, optional): If provided, use this as the change_type
                                              in the inventory log instead of 'Quantity Update'.
                                              Useful for 'Stock Adjustment', 'Usage', etc.
    """
    conn = connect_db()
    cursor = conn.cursor()

    # --- Get current quantity before update for logging ---
    try:
        # Fetch current quantity and item name for logging context
        cursor.execute("SELECT current_quantity, item_name FROM inventory WHERE id = ?", (item_id,))
        result = cursor.fetchone()
        if result:
            current_quantity_before_update = result['current_quantity']
        else:
            st.error(f"Cannot update item ID {item_id}: Item not found.")
            conn.close()
            return False
    except sqlite3.Error as e:
        st.error(f"Database error fetching current quantity for item {item_id}: {e}")
        conn.close()
        return False
    # --- End get current quantity ---


    set_clauses = []
    params = []

    # Convert expiration date to string if present
    if 'expiration_date' in updates:
        if isinstance(updates['expiration_date'], date):
            updates['expiration_date'] = updates['expiration_date'].isoformat()
        elif updates['expiration_date'] is None:
             updates['expiration_date'] = None # Allow setting expiration to NULL
        # Add validation if needed to ensure it's a valid date string or None

    # Add last_updated automatically
    updates['last_updated'] = datetime.now().isoformat()

    # Store the override in the updates dict temporarily for logging logic later
    # We don't actually need to store it in the dict if we pass it separately to the logging function
    # Let's keep it simple and pass it directly later.

    for key, value in updates.items():
        # Ensure we only try to update valid columns
        valid_columns = [
            "item_name", "category", "current_quantity", "reorder_level", "max_stock_level",
            "unit_cost", "expiration_date", "supplier", "last_updated"
        ]
        if key in valid_columns:
            set_clauses.append(f"{key} = ?")
            params.append(value)

    if not set_clauses:
        st.warning("No valid fields provided for inventory item update.")
        conn.close()
        return False

    params.append(item_id) # For the WHERE clause
    query = f"UPDATE inventory SET {', '.join(set_clauses)} WHERE id = ?"

    rows_affected = 0 # Initialize rows_affected
    try:
        cursor.execute(query, params)
        conn.commit()
        rows_affected = cursor.rowcount

        # --- Log quantity change if it occurred (only after successful commit) ---
        if rows_affected > 0 and 'current_quantity' in updates:
            new_quantity = updates['current_quantity']
            quantity_change = new_quantity - current_quantity_before_update
            # Determine the change type: Use override if provided, else default
            change_type = change_type_override if change_type_override else 'Quantity Update'
            if quantity_change != 0: # Only log if quantity actually changed
                 log_inventory_change(item_id, change_type, quantity_change, new_quantity)
        # --- End logging ---

        return rows_affected > 0 # Return True if a row was updated

    except sqlite3.IntegrityError:
         st.error(f"Database error: An item with the new name might already exist.")
         return False # Return False within the except block
    except sqlite3.Error as e:
        st.error(f"Database error updating inventory item {item_id}: {e}")
        return False # Return False within the except block
    finally:
        # Ensure connection is closed even if logging fails or other errors occur
        conn.close() # This should be indented under finally

def delete_inventory_item(item_id):
    """Deletes an inventory item from the database."""
    conn = connect_db()
    cursor = conn.cursor()
    # --- Get details before delete for logging ---
    item_name_deleted = "Unknown"
    quantity_at_deletion = 0
    try:
        cursor.execute("SELECT item_name, current_quantity FROM inventory WHERE id = ?", (item_id,))
        result = cursor.fetchone()
        if result:
            item_name_deleted = result['item_name']
            quantity_at_deletion = result['current_quantity']
    except sqlite3.Error as e:
        st.warning(f"Could not fetch details before deleting item {item_id}: {e}")
    # --- End get details ---

    try:
        cursor.execute("DELETE FROM inventory WHERE id = ?", (item_id,))
        rows_affected = cursor.rowcount
        conn.commit()
        # --- Log deletion ---
        if rows_affected > 0:
            log_inventory_change(item_id, 'Item Deleted', -quantity_at_deletion, 0, f"Item '{item_name_deleted}' deleted.")
        # --- End logging ---
        return rows_affected > 0 # Return True if a row was deleted
    except sqlite3.Error as e:
        st.error(f"Database error deleting inventory item {item_id}: {e}")
        return False
    finally:
        conn.close()

def get_distinct_inventory_categories():
    """Gets a list of distinct inventory categories."""
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT category FROM inventory WHERE category IS NOT NULL ORDER BY category")
        categories = [row['category'] for row in cursor.fetchall()]
        return categories
    except sqlite3.Error as e:
        st.error(f"Database error getting distinct inventory categories: {e}")
        return []
    finally:
        conn.close()

def get_inventory_log(item_id, change_type_filter=None):
    """
    Retrieves the change log for a specific inventory item, optionally filtered by change type.

    Args:
        item_id (int): The ID of the inventory item.
        change_type_filter (str or list, optional): A specific change type or list of types to filter by.
                                                    If 'Purchases/Additions', filters for positive changes.
                                                    Defaults to None (all types).
    """
    conn = connect_db()
    cursor = conn.cursor()
    query = "SELECT timestamp, change_type, quantity_change, new_quantity, notes FROM inventory_log WHERE item_id = ?"
    params = [item_id]

    if change_type_filter:
        if change_type_filter == 'Purchases/Additions':
            # Filter for types indicating stock increase
            query += " AND (change_type = ? OR change_type = ? OR (change_type = ? AND quantity_change > 0))"
            params.extend(['Item Added / Initial Stock', 'Purchase', 'Quantity Update']) # Add 'Purchase' if used explicitly
        elif isinstance(change_type_filter, list):
            placeholders = ','.join('?' * len(change_type_filter))
            query += f" AND change_type IN ({placeholders})"
            params.extend(change_type_filter)
        elif isinstance(change_type_filter, str):
            query += " AND change_type = ?"
            params.append(change_type_filter)

    query += " ORDER BY timestamp DESC"

    try:
        cursor.execute(query, params)
        logs = cursor.fetchall()
        df_log = pd.DataFrame([dict(row) for row in logs])
        # Convert timestamp
        if not df_log.empty:
            df_log['timestamp'] = pd.to_datetime(df_log['timestamp'])
        return df_log
    except sqlite3.Error as e:
        st.error(f"Database error getting inventory log for item {item_id}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def add_goal(metric_name, target_value, time_period, start_date=None, end_date=None, is_active=1):
    """Adds a new goal to the database."""
    conn = connect_db()
    cursor = conn.cursor()
    start_date_str = start_date.isoformat() if isinstance(start_date, date) else None
    end_date_str = end_date.isoformat() if isinstance(end_date, date) else None
    try:
        cursor.execute("""
            INSERT INTO goals (metric_name, target_value, time_period, start_date, end_date, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (metric_name, target_value, time_period, start_date_str, end_date_str, is_active))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Database error adding goal: {e}")
        return False
    finally:
        conn.close()

def get_goals(active_only=False):
    """Retrieves goals from the database."""
    conn = connect_db()
    cursor = conn.cursor()
    query = "SELECT * FROM goals"
    params = []
    if active_only:
        query += " WHERE is_active = ?"
        params.append(1)
    query += " ORDER BY created_at DESC"
    try:
        cursor.execute(query, params)
        goals = cursor.fetchall()
        df_goals = pd.DataFrame([dict(row) for row in goals])
        # Convert date columns if they exist
        if not df_goals.empty:
            for col in ['start_date', 'end_date', 'created_at']:
                 if col in df_goals.columns:
                     df_goals[col] = pd.to_datetime(df_goals[col], errors='coerce')
                     # Convert start/end date back to date only if needed, handle NaT
                     if col in ['start_date', 'end_date']:
                         df_goals[col] = df_goals[col].dt.date
        return df_goals
    except sqlite3.Error as e:
        st.error(f"Database error getting goals: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def update_goal(goal_id, updates):
    """Updates an existing goal (e.g., toggle active status, change target)."""
    conn = connect_db()
    cursor = conn.cursor()
    set_clauses = []
    params = []

    # Convert dates to strings if present
    if 'start_date' in updates and isinstance(updates['start_date'], date):
        updates['start_date'] = updates['start_date'].isoformat()
    if 'end_date' in updates and isinstance(updates['end_date'], date):
        updates['end_date'] = updates['end_date'].isoformat()

    for key, value in updates.items():
        # Ensure we only try to update valid columns
        valid_columns = [
            "metric_name", "target_value", "time_period",
            "start_date", "end_date", "is_active"
        ]
        if key in valid_columns:
            set_clauses.append(f"{key} = ?")
            params.append(value)

    if not set_clauses:
        st.warning("No valid fields provided for goal update.")
        conn.close()
        return False

    params.append(goal_id) # For the WHERE clause
    query = f"UPDATE goals SET {', '.join(set_clauses)} WHERE id = ?"

    try:
        cursor.execute(query, params)
        conn.commit()
        return cursor.rowcount > 0 # Return True if a row was updated
    except sqlite3.Error as e:
        st.error(f"Database error updating goal {goal_id}: {e}")
        return False
    finally:
        conn.close()
