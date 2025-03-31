# src/sidebar.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def render_sidebar(df_data):
    """
    Renders the sidebar with grouped navigation and simplified filters.
    Returns the filtered DataFrame and filter details.
    """
 
     # --- Navigation Groups ---
    nav_structure = {
        "📊 ANALYTICS": [
            "📈 Executive Summary",
            "💰 Financial Performance",
            "👨‍⚕️ Doctor Analytics",
            "👥 Patient Insights",
            "🔍 Operational Metrics",
            "📊 Comparison Charts" # Added new comparison tab
        ],
        "📅 SCHEDULING": [
            "⏱️ Daily Workflow",
            "📅 Appointment Scheduling"
        ],
        "💵 COSTS": [
            "💸 Cost Entry",
            "📊 Cost Analysis"
        ],
        "🎯 GOALS": [
            "🎯 Goal Setting",
            "📈 Goal Tracking"
        ],
        "📦 INVENTORY": [
            "📦 Inventory Management",
            "📉 Inventory Tracking",
            "📄 Inventory Reports"
        ],
        "🤖 AI": [
            "🤖 AI Predictions",
            "📋 Detailed Reports"
        ]
    }

    # New navigation system
    selected_category = st.sidebar.selectbox(
        "Main Menu",
        list(nav_structure.keys()),
        format_func=lambda x: x.strip()
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {selected_category}")

    # Initialize session state for selected tab if it doesn't exist
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = nav_structure[selected_category][0] # Default to first item initially

    # Check if the category changed, if so, reset selected_tab to the first item of the new category
    if 'current_category' not in st.session_state or st.session_state.current_category != selected_category:
        st.session_state.current_category = selected_category
        st.session_state.selected_tab = nav_structure[selected_category][0]


    # Create buttons for subcategories, update session state on click
    for item in nav_structure[selected_category]:
        if st.sidebar.button(
            item,
            use_container_width=True,
            key=f"btn_{item}",
        ):
            st.session_state.selected_tab = item # Update session state

    # Use the selected tab from session state
    selected_tab = st.session_state.selected_tab

    st.sidebar.markdown("---")

    # --- Simplified Filters ---
    with st.sidebar.expander("🔍 Filters", expanded=False):
        # Quick date presets
        date_preset = st.selectbox(
            "Quick Select",
            ["Custom", "Today", "Last 7 Days", "Last 30 Days", "This Month", "Last Month"]
        )
        
        # Calculate preset dates
        today = datetime.now().date()
        if date_preset == "Today":
            start_date = end_date = today
        elif date_preset == "Last 7 Days":
            start_date = today - timedelta(days=7)
            end_date = today
        elif date_preset == "Last 30 Days":
            start_date = today - timedelta(days=30)
            end_date = today
        elif date_preset == "This Month":
            start_date = today.replace(day=1)
            end_date = today
        elif date_preset == "Last Month":
            last_month = today.replace(day=1) - timedelta(days=1)
            start_date = last_month.replace(day=1)
            end_date = last_month
        else:  # Custom
            min_date = df_data["date"].min().date()
            max_date = df_data["date"].max().date()
            date_range = st.date_input(
                "Custom Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date

        # Advanced filters
        st.divider()
        if "doctor" in df_data.columns:
            doctors = st.multiselect(
                "Doctors",
                options=sorted(df_data["doctor"].unique()),
                default=[]
            )
        
        if "department" in df_data.columns:
            departments = st.multiselect(
                "Departments",
                options=sorted(df_data["department"].unique()),
                default=[]
            )

        # Apply all filters
        filtered_df = df_data[
            (df_data["date"].dt.date >= start_date) & 
            (df_data["date"].dt.date <= end_date)
        ].copy()

        if "doctor" in df_data.columns and doctors:
            filtered_df = filtered_df[filtered_df["doctor"].isin(doctors)]
        
        if "department" in df_data.columns and departments:
            filtered_df = filtered_df[filtered_df["department"].isin(departments)]

    # Add refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    # Ensure filter variables exist even if columns are missing
    doctors = doctors if "doctor" in df_data.columns and 'doctors' in locals() else []
    departments = departments if "department" in df_data.columns and 'departments' in locals() else []

    filter_details = {
        "start_date": start_date,
        "end_date": end_date,
        "selected_tab": st.session_state.selected_tab, # Use session state value
        "doctors": doctors, # Already initialized safely above
        "departments": departments # Already initialized safely above
    }

    # Return the selected tab from session state as well
    return filtered_df, filter_details, st.session_state.selected_tab

# Add custom CSS to improve button styling
st.markdown("""
    <style>
    .stButton button {
        text-align: left;
        padding: 10px 15px;
        background-color: transparent;
        border: none;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: rgba(151, 166, 195, 0.15);
    }
    </style>
""", unsafe_allow_html=True)
