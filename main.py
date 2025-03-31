# main.py
import streamlit as st
import pandas as pd # Keep pandas for potential use
import sys # To modify path for imports
import os # To construct path

# --- Page Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Advanced Clinic Financial Analytics Dashboard",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src directory to Python path
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


# Import modules
# Use absolute imports relative to the src directory structure
try:
    from data_loader import load_data
    from sidebar import render_sidebar
    from tabs.executive_summary import render_executive_summary_tab
    from tabs.financial_performance import render_financial_performance_tab
    from tabs.doctor_analytics import render_doctor_analytics_tab
    from tabs.patient_insights import render_patient_insights_tab
    from tabs.operational_metrics import render_operational_metrics_tab
    from tabs.ai_predictions import render_ai_predictions_tab
    from tabs.detailed_reports import render_detailed_reports_tab
    from tabs.appointment_scheduling import render_appointment_scheduling_tab
    from tabs.daily_workflow import render_daily_workflow_tab # Added import for daily workflow
    from tabs.cost_entry import render_cost_entry_tab # Added import for cost entry
    from tabs.cost_analysis import render_cost_analysis_tab # Added import for cost analysis
    from tabs.goal_setting import render_goal_setting_tab # Added import for goal setting
    from tabs.goal_tracking import render_goal_tracking_tab # Added import for goal tracking
    from tabs.inventory_management import show as render_inventory_management_tab # Added import for inventory
    from tabs.inventory_tracking import show as render_inventory_tracking_tab # Added import for inventory tracking
    from tabs.inventory_reports import render_inventory_reports_tab # Added import for inventory reports
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error(f"Current sys.path: {sys.path}")
    st.error("Please ensure the 'src' directory and its contents are structured correctly relative to main.py.")
    st.stop()


# --- Main Header ---
# Page config moved to the top
st.title("Advanced Clinic Financial Analytics Dashboard")

# --- Load Data ---
# This now calls the function from data_loader.py
df_data = load_data()

# --- Render Sidebar and Get Filtered Data ---
# This calls the function from sidebar.py
# It returns the filtered dataframe, filter details, and selected tab
filtered_df, filter_details, selected_tab = render_sidebar(df_data)

# Extract filter details for use in tabs if needed
start_date = filter_details["start_date"]
end_date = filter_details["end_date"]
# selected_doctors = filter_details["selected_doctors"]
# selected_visit_types = filter_details["selected_visit_types"]
# selected_payment_methods = filter_details["selected_payment_methods"]

# --- Main Content Area ---
# Render content based on selected tab from sidebar
if selected_tab == "📈 Executive Summary":
    render_executive_summary_tab(filtered_df, df_data, start_date, end_date)
elif selected_tab == "💰 Financial Performance":
    render_financial_performance_tab(filtered_df)
elif selected_tab == "👨‍⚕️ Doctor Analytics":
    render_doctor_analytics_tab(filtered_df)
elif selected_tab == "👥 Patient Insights":
    render_patient_insights_tab(filtered_df, df_data, start_date, end_date)
elif selected_tab == "🔍 Operational Metrics":
    render_operational_metrics_tab(filtered_df, start_date, end_date)
elif selected_tab == "⏱️ Daily Workflow":
    render_daily_workflow_tab(df_data)
elif selected_tab == "📅 Appointment Scheduling":
    render_appointment_scheduling_tab(df_data)
elif selected_tab == "💸 Cost Entry":
    render_cost_entry_tab()
elif selected_tab == "📊 Cost Analysis":
    render_cost_analysis_tab(filtered_df, start_date, end_date)
elif selected_tab == "🎯 Goal Setting":
    render_goal_setting_tab()
elif selected_tab == "📈 Goal Tracking":
    render_goal_tracking_tab(filtered_df, df_data, start_date, end_date)
elif selected_tab == "📦 Inventory Management":
    render_inventory_management_tab()
elif selected_tab == "📉 Inventory Tracking":
    render_inventory_tracking_tab()
elif selected_tab == "📄 Inventory Reports":
    render_inventory_reports_tab()
elif selected_tab == "🤖 AI Predictions":
    render_ai_predictions_tab(filtered_df)
elif selected_tab == "📋 Detailed Reports":
    render_detailed_reports_tab(filtered_df, start_date, end_date)


# --- Footer or Final Message (Optional) ---
st.sidebar.markdown("---")
st.sidebar.info("Dashboard Refactored & Enhanced by Cline.")

# print("Dashboard structure updated successfully.") # Optional: for local debugging
