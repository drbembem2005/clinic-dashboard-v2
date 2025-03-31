# src/tabs/comparison_charts.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta, time # Added time
from data_loader import get_appointments, get_costs # Import functions to load data

# Define a consistent color palette/template
PLOTLY_TEMPLATE = "plotly_white"

def calculate_previous_period(start_date, end_date, comparison_option):
    """Calculates the start and end dates for the previous comparison period."""
    period_duration = (end_date - start_date).days + 1

    if comparison_option == "Previous Period":
        prev_start_date = start_date - timedelta(days=period_duration)
        prev_end_date = start_date - timedelta(days=1)
    elif comparison_option == "Previous Month":
        current_month_start = start_date.replace(day=1)
        prev_month_end = current_month_start - timedelta(days=1)
        prev_month_start = prev_month_end.replace(day=1)
        prev_start_date = prev_month_start
        prev_end_date = prev_month_end
    elif comparison_option == "Previous Quarter":
        current_quarter_start_month = ((start_date.month - 1) // 3) * 3 + 1
        current_quarter_start = datetime(start_date.year, current_quarter_start_month, 1).date()
        prev_quarter_end = current_quarter_start - timedelta(days=1)
        prev_quarter_start_month = ((prev_quarter_end.month - 1) // 3) * 3 + 1
        prev_quarter_start = datetime(prev_quarter_end.year, prev_quarter_start_month, 1).date()
        prev_start_date = prev_quarter_start
        prev_end_date = prev_quarter_end
    elif comparison_option == "Previous Year":
        prev_start_date = start_date.replace(year=start_date.year - 1)
        prev_end_date = end_date.replace(year=end_date.year - 1)
        # Adjust for leap years if necessary
        try:
            # Check if original end_date was Feb 29 and prev year is not leap
            if end_date.month == 2 and end_date.day == 29:
                 datetime(prev_end_date.year, 2, 29) # This will raise ValueError if not leap
        except ValueError:
            prev_end_date = prev_end_date.replace(day=28)
    else: # Default to previous period
        prev_start_date = start_date - timedelta(days=period_duration)
        prev_end_date = start_date - timedelta(days=1)

    return prev_start_date, prev_end_date

def calculate_change(current_val, prev_val):
    """Helper function to calculate percentage change, handling zero/NaN."""
    if prev_val is None or pd.isna(prev_val) or prev_val == 0:
        return None # Indicate change cannot be calculated
    return ((current_val - prev_val) / prev_val * 100)

def render_comparison_charts_tab(filtered_df, df_data, start_date, end_date):
    """
    Renders the Comparison Charts tab content.

    Args:
        filtered_df (pd.DataFrame): The filtered DataFrame based on sidebar selections.
        df_data (pd.DataFrame): The original unfiltered DataFrame.
        start_date (datetime.date): The start date from the filter.
        end_date (datetime.date): The end date from the filter.
    """
    st.header("📊 Performance Comparison")
    st.divider()

    # --- Comparison Period Selection ---
    comparison_option = st.selectbox(
        "Compare selected period to:",
        ("Previous Period", "Previous Month", "Previous Quarter", "Previous Year"),
        key="comparison_period_select"
    )

    prev_start_date, prev_end_date = calculate_previous_period(start_date, end_date, comparison_option)

    st.info(f"Comparing **{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}** with **{prev_start_date.strftime('%Y-%m-%d')} to {prev_end_date.strftime('%Y-%m-%d')}** ({comparison_option})")

    # Filter main data for the previous period
    # TODO: Enhance this to apply the same non-date filters as the current period.
    prev_period_df = df_data[
        (df_data["date"].dt.date >= prev_start_date) &
        (df_data["date"].dt.date <= prev_end_date)
    ].copy() # Use copy to avoid SettingWithCopyWarning

    # --- Load Additional Data for Comparison Periods ---
    # Load appointments and costs for the selected and previous periods
    current_appointments_df = get_appointments(start_date_filter=start_date, end_date_filter=end_date)
    prev_appointments_df = get_appointments(start_date_filter=prev_start_date, end_date_filter=prev_end_date)

    current_costs_df = get_costs(start_date_filter=start_date, end_date_filter=end_date, date_column='expense_date')
    prev_costs_df = get_costs(start_date_filter=prev_start_date, end_date_filter=prev_end_date, date_column='expense_date')

    # --- Calculate Waiting Time ---
    def calculate_avg_waiting_time(df_appts):
        if df_appts.empty or 'PatientArrivalTime' not in df_appts.columns or 'AppointmentStartTime' not in df_appts.columns or 'AppointmentDateTime' not in df_appts.columns:
            return 0

        # Ensure datetime column is datetime type
        df_appts['AppointmentDateTime'] = pd.to_datetime(df_appts['AppointmentDateTime'], errors='coerce')
        df_appts.dropna(subset=['AppointmentDateTime'], inplace=True) # Drop rows where conversion failed

        wait_times = []
        for _, row in df_appts.iterrows():
            try:
                # Combine appointment date with arrival/start times (stored as HH:MM:SS strings)
                arrival_str = row['PatientArrivalTime']
                start_str = row['AppointmentStartTime']
                appt_date = row['AppointmentDateTime'].date()

                if arrival_str and start_str:
                    arrival_time = time.fromisoformat(arrival_str)
                    start_time = time.fromisoformat(start_str)

                    arrival_dt = datetime.combine(appt_date, arrival_time)
                    start_dt = datetime.combine(appt_date, start_time)

                    # Only calculate if start is after arrival
                    if start_dt > arrival_dt:
                        wait_duration = (start_dt - arrival_dt).total_seconds() / 60 # Duration in minutes
                        wait_times.append(wait_duration)
            except (TypeError, ValueError):
                continue # Skip row if time parsing fails or data is missing

        return sum(wait_times) / len(wait_times) if wait_times else 0

    avg_wait_current = calculate_avg_waiting_time(current_appointments_df.copy()) # Use copy to avoid modifying original
    avg_wait_prev = calculate_avg_waiting_time(prev_appointments_df.copy())

    # --- Calculate Total Costs ---
    total_costs_current = current_costs_df['amount'].sum() if not current_costs_df.empty else 0
    total_costs_prev = prev_costs_df['amount'].sum() if not prev_costs_df.empty else 0

    # --- Warnings for missing data ---
    if filtered_df.empty:
        st.warning("No main financial data available for the selected filters in the current period.")
    if prev_period_df.empty:
        st.warning(f"No main financial data available for the comparison period ({prev_start_date.strftime('%Y-%m-%d')} to {prev_end_date.strftime('%Y-%m-%d')}).")
    if current_appointments_df.empty:
        st.warning("No appointment data available for the current period.")
    if prev_appointments_df.empty:
        st.warning("No appointment data available for the comparison period.")
    if current_costs_df.empty:
        st.warning("No cost data available for the current period.")
    if prev_costs_df.empty:
        st.warning("No cost data available for the comparison period.")

    st.divider()

    # --- KPI Comparison ---
    st.subheader("KPI Comparison")

    # Define KPIs to compare (using main filtered_df and prev_period_df)
    kpis_from_main_df = {
        "Total Revenue": ("gross income", "sum", "EGP{value:,.2f}"),
        "Total Profit": ("profit", "sum", "EGP{value:,.2f}"),
        "Unique Patients": ("Patient", "nunique", "{value:,}"),
        "Total Appointments": ("id", "count", "{value:,}"), # Count from main df (visits)
        "Avg. Revenue per Visit": ("gross income", "mean", "EGP{value:,.2f}"),
        "Avg. Profit per Visit": ("profit", "mean", "EGP{value:,.2f}"),
        "Avg. Visit Duration (Mins)": ("visit_duration_mins", "mean", "{value:.0f} mins"),
    }

    # KPIs calculated from separate data sources
    kpis_calculated = {
        "Avg. Waiting Time (Mins)": (avg_wait_current, avg_wait_prev, "{value:.0f} mins"),
        "Total Costs": (total_costs_current, total_costs_prev, "EGP{value:,.2f}"),
    }

    num_kpi_cols = 4 # Keep 4 columns for layout
    kpi_cols = st.columns(num_kpi_cols)
    col_index = 0

    # Display KPIs from main DataFrame
    for kpi_name, (col, agg, fmt) in kpis_from_main_df.items():
        current_val = 0
        prev_val = 0

        if not filtered_df.empty and col in filtered_df.columns:
            if agg == "sum":
                current_val = filtered_df[col].sum()
            elif agg == "mean":
                current_val = filtered_df[col].mean()
            elif agg == "nunique":
                 current_val = filtered_df[col].nunique()
            elif agg == "count":
                 current_val = filtered_df[col].count()

        if not prev_period_df.empty and col in prev_period_df.columns:
            if agg == "sum":
                prev_val = prev_period_df[col].sum()
            elif agg == "mean":
                prev_val = prev_period_df[col].mean()
            elif agg == "nunique":
                 prev_val = prev_period_df[col].nunique()
            elif agg == "count":
                 prev_val = prev_period_df[col].count()

        change = calculate_change(current_val, prev_val)
        delta_str = f"{change:.1f}%" if change is not None else "N/A"

        with kpi_cols[col_index % num_kpi_cols]:
            st.metric(
                label=f"{kpi_name}",
                value=fmt.format(value=current_val),
                delta=delta_str,
                help=f"Current: {fmt.format(value=current_val)}\nPrevious: {fmt.format(value=prev_val)}"
            )
        col_index += 1

    # Display calculated KPIs
    for kpi_name, (current_val, prev_val, fmt) in kpis_calculated.items():
        change = calculate_change(current_val, prev_val)
        delta_str = f"{change:.1f}%" if change is not None else "N/A"

        with kpi_cols[col_index % num_kpi_cols]:
             st.metric(
                label=f"{kpi_name}",
                value=fmt.format(value=current_val),
                delta=delta_str,
                help=f"Current: {fmt.format(value=current_val)}\nPrevious: {fmt.format(value=prev_val)}"
            )
        col_index += 1


    st.divider()

    # --- Chart Comparison ---
    st.subheader("Chart Comparison")

    # Row 1: Revenue & Appointments
    chart_row1_col1, chart_row1_col2 = st.columns(2)

    with chart_row1_col1:
        st.markdown("##### Revenue Trend Comparison")
        if not filtered_df.empty or not prev_period_df.empty:
            # Current Period Data Prep
            current_daily = filtered_df.groupby(filtered_df["date"].dt.date)["gross income"].sum().reset_index()
            current_daily.columns = ["date", "revenue"]
            current_daily['date'] = pd.to_datetime(current_daily['date'])
            current_daily['Period'] = 'Current'

            # Previous Period Data Prep
            prev_daily = prev_period_df.groupby(prev_period_df["date"].dt.date)["gross income"].sum().reset_index()
            prev_daily.columns = ["date", "revenue"]
            prev_daily['date'] = pd.to_datetime(prev_daily['date'])
            prev_daily['Period'] = 'Previous'

            combined_daily = pd.concat([current_daily, prev_daily], ignore_index=True)

            if not combined_daily.empty:
                fig_trend = px.line(
                    combined_daily,
                    x="date",
                    y="revenue",
                    color="Period",
                    title="Daily Revenue Trend Comparison",
                    labels={"date": "Date", "revenue": "Revenue (EGP)", "Period": "Period"},
                    line_shape="spline",
                    template=PLOTLY_TEMPLATE,
                    color_discrete_map={'Current': '#1976D2', 'Previous': '#FF9800'} # Assign specific colors
                )
                fig_trend.update_traces(mode="lines+markers", line=dict(width=2))
                fig_trend.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_tickformat="%b %d, %Y",
                    yaxis_tickprefix="EGP",
                    legend_title_text='Period'
                )
                st.plotly_chart(fig_trend, use_container_width=True, key="comp_revenue_trend")
            else:
                st.info("No revenue data to display for trend comparison.")
        else:
            st.info("No data available in either period for revenue trend comparison.")

    with chart_row1_col2:
        st.markdown("##### Appointments Trend Comparison")
        # Use the loaded appointment dataframes
        current_appts_daily = pd.DataFrame(columns=['date', 'appointments']) # Initialize empty df
        prev_appts_daily = pd.DataFrame(columns=['date', 'appointments']) # Initialize empty df

        if not current_appointments_df.empty and 'AppointmentDateTime' in current_appointments_df.columns:
            # Current Period Data Prep
            current_appts = current_appointments_df.copy()
            current_appts['AppointmentDateTime'] = pd.to_datetime(current_appts['AppointmentDateTime'], errors='coerce')
            current_appts = current_appts.dropna(subset=['AppointmentDateTime'])
            if not current_appts.empty: # Check again after dropna
                current_appts_daily = current_appts.groupby(current_appts["AppointmentDateTime"].dt.date).size().reset_index(name='appointments')
                current_appts_daily.columns = ["date", "appointments"]
                current_appts_daily.columns = ["date", "appointments"]
                current_appts_daily['date'] = pd.to_datetime(current_appts_daily['date'])
                current_appts_daily['Period'] = 'Current'

        if not prev_appointments_df.empty and 'AppointmentDateTime' in prev_appointments_df.columns:
            # Previous Period Data Prep
            prev_appts = prev_appointments_df.copy()
            prev_appts['AppointmentDateTime'] = pd.to_datetime(prev_appts['AppointmentDateTime'], errors='coerce')
            prev_appts = prev_appts.dropna(subset=['AppointmentDateTime'])
            if not prev_appts.empty: # Check again after dropna
                prev_appts_daily = prev_appts.groupby(prev_appts["AppointmentDateTime"].dt.date).size().reset_index(name='appointments')
                prev_appts_daily.columns = ["date", "appointments"]
                prev_appts_daily['date'] = pd.to_datetime(prev_appts_daily['date'])
                prev_appts_daily['Period'] = 'Previous'

        # Combine only if there's data
        if not current_appts_daily.empty or not prev_appts_daily.empty:
            combined_appts = pd.concat([current_appts_daily, prev_appts_daily], ignore_index=True)
            if not combined_appts.empty:
                fig_appts = px.line(
                    combined_appts,
                    x="date",
                    y="appointments",
                    color="Period",
                    title="Daily Appointments Trend Comparison",
                    labels={"date": "Date", "appointments": "Number of Appointments", "Period": "Period"},
                    line_shape="spline",
                    template=PLOTLY_TEMPLATE,
                    color_discrete_map={'Current': '#1976D2', 'Previous': '#FF9800'}
                )
                fig_appts.update_traces(mode="lines+markers", line=dict(width=2))
                fig_appts.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_tickformat="%b %d, %Y",
                    legend_title_text='Period'
                )
                st.plotly_chart(fig_appts, use_container_width=True, key="comp_appointments_trend")
            else:
                st.info("No appointment data to display for trend comparison.")
        else:
            st.info("No appointment data available in either period for trend comparison.")


    # Row 2: Visit Type and Doctor Revenue Comparison
    chart_row2_col1, chart_row2_col2 = st.columns(2)

    with chart_row2_col1:
        st.markdown("##### Revenue by Visit Type Comparison")
        if ('visit type' in filtered_df.columns and not filtered_df.empty) or \
           ('visit type' in prev_period_df.columns and not prev_period_df.empty):

            current_visit = filtered_df.groupby("visit type")["gross income"].sum().reset_index()
            current_visit.columns = ["visit_type", "revenue"]
            current_visit['Period'] = 'Current'

            prev_visit = prev_period_df.groupby("visit type")["gross income"].sum().reset_index()
            prev_visit.columns = ["visit_type", "revenue"]
            prev_visit['Period'] = 'Previous'

            combined_visit = pd.concat([current_visit, prev_visit], ignore_index=True)

            if not combined_visit.empty:
                fig_visit = px.bar(
                    combined_visit,
                    x="visit_type",
                    y="revenue",
                    color="Period",
                    barmode="group", # Group bars side-by-side
                    title="Revenue by Visit Type Comparison",
                    labels={"visit_type": "Visit Type", "revenue": "Revenue (EGP)", "Period": "Period"},
                    template=PLOTLY_TEMPLATE,
                    color_discrete_map={'Current': '#1976D2', 'Previous': '#FF9800'}
                )
                fig_visit.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    yaxis_tickprefix="EGP",
                    legend_title_text='Period',
                    xaxis_title=None
                )
                st.plotly_chart(fig_visit, use_container_width=True, key="comp_visit_revenue")
            else:
                st.info("No visit type revenue data for comparison.")
        else:
            st.info("Visit type column not found or no data for comparison.")


    with chart_row2_col2:
        st.markdown("##### Top Doctors Revenue Comparison")
        # Similar logic for doctors - get top N from current, then find their previous values
        if ('Doctor' in filtered_df.columns and not filtered_df.empty) or \
           ('Doctor' in prev_period_df.columns and not prev_period_df.empty):

            # Get top N doctors from the current period
            top_n = 10
            current_doctor = filtered_df.groupby("Doctor")["gross income"].sum().reset_index()
            current_doctor.columns = ["doctor", "revenue"]
            current_doctor = current_doctor.sort_values("revenue", ascending=False).head(top_n)
            current_doctor['Period'] = 'Current'
            top_doctor_names = current_doctor['doctor'].tolist()

            # Get revenue for the *same* top doctors in the previous period
            prev_doctor = prev_period_df[prev_period_df['Doctor'].isin(top_doctor_names)]
            prev_doctor = prev_doctor.groupby("Doctor")["gross income"].sum().reset_index()
            prev_doctor.columns = ["doctor", "revenue"]
            prev_doctor['Period'] = 'Previous'

            # Combine, ensuring all top doctors are present even if they had 0 revenue previously
            combined_doctor = pd.concat([current_doctor, prev_doctor], ignore_index=True)

            if not combined_doctor.empty:
                fig_doctor = px.bar(
                    combined_doctor,
                    x="revenue", # Horizontal bar chart
                    y="doctor",
                    color="Period",
                    barmode="group",
                    title=f"Top {top_n} Doctors Revenue Comparison",
                    labels={"doctor": "Doctor", "revenue": "Revenue (EGP)", "Period": "Period"},
                    template=PLOTLY_TEMPLATE,
                    color_discrete_map={'Current': '#1976D2', 'Previous': '#FF9800'},
                    orientation='h' # Horizontal
                )
                fig_doctor.update_layout(
                    height=400 + (top_n * 20), # Adjust height based on N
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_tickprefix="EGP",
                    yaxis_categoryorder='total ascending', # Order by current period revenue
                    legend_title_text='Period',
                    yaxis_title=None
                )
                st.plotly_chart(fig_doctor, use_container_width=True, key="comp_doctor_revenue")
            else:
                st.info("No doctor revenue data for comparison.")
        else:
            st.info("Doctor column not found or no data for comparison.")

    # Row 3: Costs and Waiting Time
    chart_row3_col1, chart_row3_col2 = st.columns(2)

    with chart_row3_col1:
        st.markdown("##### Total Costs Trend Comparison")
        # Use the loaded cost dataframes
        current_costs_daily = pd.DataFrame(columns=['date', 'cost']) # Initialize empty df
        prev_costs_daily = pd.DataFrame(columns=['date', 'cost']) # Initialize empty df

        if not current_costs_df.empty and 'expense_date' in current_costs_df.columns:
            # Current Period Data Prep
            current_costs = current_costs_df.copy()
            current_costs['expense_date'] = pd.to_datetime(current_costs['expense_date'], errors='coerce')
            current_costs = current_costs.dropna(subset=['expense_date'])
            if not current_costs.empty: # Check again after dropna
                current_costs_daily = current_costs.groupby(current_costs["expense_date"].dt.date)["amount"].sum().reset_index()
                current_costs_daily.columns = ["date", "cost"]
                current_costs_daily['date'] = pd.to_datetime(current_costs_daily['date'])
                current_costs_daily['Period'] = 'Current'

        if not prev_costs_df.empty and 'expense_date' in prev_costs_df.columns:
            # Previous Period Data Prep
            prev_costs = prev_costs_df.copy()
            prev_costs['expense_date'] = pd.to_datetime(prev_costs['expense_date'], errors='coerce')
            prev_costs = prev_costs.dropna(subset=['expense_date'])
            if not prev_costs.empty: # Check again after dropna
                prev_costs_daily = prev_costs.groupby(prev_costs["expense_date"].dt.date)["amount"].sum().reset_index()
                prev_costs_daily.columns = ["date", "cost"]
                prev_costs_daily['date'] = pd.to_datetime(prev_costs_daily['date'])
                prev_costs_daily['Period'] = 'Previous'

        # Combine only if there's data
        if not current_costs_daily.empty or not prev_costs_daily.empty:
            combined_costs = pd.concat([current_costs_daily, prev_costs_daily], ignore_index=True)
            if not combined_costs.empty:
                fig_costs = px.line(
                    combined_costs,
                    x="date",
                    y="cost",
                    color="Period",
                    title="Daily Total Costs Trend Comparison",
                    labels={"date": "Date", "cost": "Total Cost (EGP)", "Period": "Period"},
                    line_shape="spline",
                    template=PLOTLY_TEMPLATE,
                    color_discrete_map={'Current': '#D32F2F', 'Previous': '#FFA000'} # Different colors for cost
                )
                fig_costs.update_traces(mode="lines+markers", line=dict(width=2))
                fig_costs.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_tickformat="%b %d, %Y",
                    yaxis_tickprefix="EGP",
                    legend_title_text='Period'
                )
                st.plotly_chart(fig_costs, use_container_width=True, key="comp_costs_trend")
            else:
                st.info("No cost data to display for trend comparison.")
        else:
            st.info("No cost data available in either period for trend comparison.")

    with chart_row3_col2:
        st.markdown("##### Average Waiting Time Trend Comparison")
        # Use the loaded appointment dataframes and calculated average waiting times
        # Need to calculate daily average waiting time
        def calculate_daily_avg_wait(df_appts):
            required_cols = ['PatientArrivalTime', 'AppointmentStartTime', 'AppointmentDateTime']
            if df_appts.empty or not all(col in df_appts.columns for col in required_cols):
                return pd.DataFrame(columns=['date', 'avg_wait_time'])

            df = df_appts.copy() # Work on a copy
            df['AppointmentDateTime'] = pd.to_datetime(df['AppointmentDateTime'], errors='coerce')
            df.dropna(subset=['AppointmentDateTime'], inplace=True)
            if df.empty:
                 return pd.DataFrame(columns=['date', 'avg_wait_time'])

            # Calculate wait time safely
            wait_times = []
            for _, row in df.iterrows():
                try:
                    arrival_str = row['PatientArrivalTime']
                    start_str = row['AppointmentStartTime']
                    appt_date = row['AppointmentDateTime'].date()
                    if arrival_str and start_str:
                        arrival_time = time.fromisoformat(arrival_str)
                        start_time = time.fromisoformat(start_str)
                        arrival_dt = datetime.combine(appt_date, arrival_time)
                        start_dt = datetime.combine(appt_date, start_time)
                        if start_dt > arrival_dt:
                            wait_times.append((start_dt - arrival_dt).total_seconds() / 60)
                        else:
                            wait_times.append(None) # Or 0 if preferred for non-positive waits
                    else:
                        wait_times.append(None)
                except (TypeError, ValueError):
                    wait_times.append(None) # Handle parsing errors

            df['wait_time_mins'] = wait_times
            df.dropna(subset=['wait_time_mins'], inplace=True) # Remove rows where wait time couldn't be calculated

            if df.empty:
                 return pd.DataFrame(columns=['date', 'avg_wait_time'])

            daily_avg_wait = df.groupby(df['AppointmentDateTime'].dt.date)['wait_time_mins'].mean().reset_index()
            daily_avg_wait.columns = ['date', 'avg_wait_time']
            daily_avg_wait['date'] = pd.to_datetime(daily_avg_wait['date'])
            return daily_avg_wait

        current_wait_daily = calculate_daily_avg_wait(current_appointments_df.copy())
        prev_wait_daily = calculate_daily_avg_wait(prev_appointments_df.copy())

        if not current_wait_daily.empty or not prev_wait_daily.empty:
            current_wait_daily['Period'] = 'Current'
            prev_wait_daily['Period'] = 'Previous'
            combined_wait = pd.concat([current_wait_daily, prev_wait_daily], ignore_index=True)

            if not combined_wait.empty:
                fig_wait = px.line(
                    combined_wait,
                    x="date",
                    y="avg_wait_time",
                    color="Period",
                    title="Daily Average Waiting Time Comparison",
                    labels={"date": "Date", "avg_wait_time": "Avg Wait Time (Mins)", "Period": "Period"},
                    line_shape="spline",
                    template=PLOTLY_TEMPLATE,
                    color_discrete_map={'Current': '#7B1FA2', 'Previous': '#00BCD4'} # Different colors for wait time
                )
                fig_wait.update_traces(mode="lines+markers", line=dict(width=2))
                fig_wait.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_tickformat="%b %d, %Y",
                    yaxis_title="Avg Wait Time (Mins)",
                    legend_title_text='Period'
                )
                st.plotly_chart(fig_wait, use_container_width=True, key="comp_wait_time_trend")
            else:
                st.info("No waiting time data to display for trend comparison.")
        else:
            st.info("No appointment data available in either period to calculate waiting time trends.")

    # Add more comparison charts as needed (e.g., patient counts, inventory if applicable)
