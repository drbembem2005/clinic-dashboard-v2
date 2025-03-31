# src/tabs/inventory_reports.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
from src import data_loader
import io # For export

def render_inventory_reports_tab():
    """Renders the dedicated Inventory Reports tab."""
    st.title("📦 Inventory Reports")

    report_options = [
        "Consumption Report",
        "Expired Stock Report",
        "Stock Value Report",
        "Slow-Moving Stock Report"
    ]
    selected_report = st.radio(
        "Select Report Type:",
        options=report_options,
        horizontal=True,
        key="inv_report_type"
    )

    st.divider()
    display_df = pd.DataFrame() # Initialize for export

    # --- Consumption Report ---
    if selected_report == "Consumption Report":
        st.subheader("Consumption Report")
        st.markdown("Analyze how much of each item was consumed within a period.")

        # Date range filter
        today = date.today()
        default_start = today - timedelta(days=30)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=default_start, key="consump_start")
        with col2:
            end_date = st.date_input("End Date", value=today, key="consump_end")

        if start_date and end_date and start_date <= end_date:
            df_consumption = data_loader.get_consumption_data(start_date, end_date)

            if not df_consumption.empty:
                st.dataframe(
                    df_consumption,
                    column_config={
                        "item_name": "Item Name",
                        "category": "Category",
                        "total_consumed": st.column_config.NumberColumn("Total Consumed", format="%d units")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                display_df = df_consumption
            else:
                st.info("No consumption data found for the selected period.")
        else:
            st.warning("Please select a valid date range.")

    # --- Expired Stock Report ---
    elif selected_report == "Expired Stock Report":
        st.subheader("Expired & Expiring Stock Report")
        st.markdown("Identify expired items and those expiring soon.")

        days_options = [30, 60, 90, 180]
        days_ahead = st.selectbox(
            "Check for items expiring within the next (days):",
            options=days_options,
            index=0, # Default to 30 days
            key="exp_days_ahead"
        )

        reference_date = date.today()
        df_expired = data_loader.get_expired_stock_data(reference_date, days_ahead)

        if not df_expired.empty:
            total_expired_value = df_expired[df_expired['expiration_status'] == 'Already Expired']['estimated_value'].sum()
            total_expiring_value = df_expired[df_expired['expiration_status'] == 'Expiring Soon']['estimated_value'].sum()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Value of Already Expired Stock", f"EGP {total_expired_value:,.2f}")
            with col2:
                st.metric(f"Value of Stock Expiring in {days_ahead} Days", f"EGP {total_expiring_value:,.2f}")

            st.dataframe(
                df_expired,
                column_config={
                    "item_name": "Item Name",
                    "category": "Category",
                    "current_quantity": st.column_config.NumberColumn("Current Qty", format="%d"),
                    "unit_cost": st.column_config.NumberColumn("Unit Cost", format="EGP %.2f"),
                    "expiration_date": st.column_config.DateColumn("Expiration Date", format="YYYY-MM-DD"),
                    "estimated_value": st.column_config.NumberColumn("Estimated Value", format="EGP %.2f"),
                    "expiration_status": "Status"
                },
                use_container_width=True,
                hide_index=True
            )
            display_df = df_expired
        else:
            st.info(f"No items found that are already expired or expiring within the next {days_ahead} days.")

    # --- Stock Value Report ---
    elif selected_report == "Stock Value Report":
        st.subheader("Stock Value Report")
        st.markdown("View the current estimated value of your inventory.")

        df_inventory = data_loader.get_inventory_items()
        if not df_inventory.empty:
            # Calculate value
            df_inventory['unit_cost'] = pd.to_numeric(df_inventory['unit_cost'], errors='coerce').fillna(0)
            df_inventory['current_quantity'] = pd.to_numeric(df_inventory['current_quantity'], errors='coerce').fillna(0)
            df_inventory['estimated_value'] = df_inventory['current_quantity'] * df_inventory['unit_cost']

            total_value = df_inventory['estimated_value'].sum()
            st.metric("Total Estimated Inventory Value", f"EGP {total_value:,.2f}")

            # Value by category
            value_by_category = df_inventory.groupby('category')['estimated_value'].sum().reset_index()
            value_by_category.columns = ['Category', 'Total Value']
            value_by_category = value_by_category.sort_values('Total Value', ascending=False)

            st.dataframe(
                value_by_category,
                column_config={
                    "Category": "Category",
                    "Total Value": st.column_config.NumberColumn("Total Estimated Value", format="EGP %.2f")
                },
                use_container_width=True,
                hide_index=True
            )
            # Optionally show detailed breakdown
            with st.expander("Show Detailed Item Values"):
                 st.dataframe(
                    df_inventory[['item_name', 'category', 'current_quantity', 'unit_cost', 'estimated_value']],
                     column_config={
                        "item_name": "Item Name",
                        "category": "Category",
                        "current_quantity": st.column_config.NumberColumn("Current Qty", format="%d"),
                        "unit_cost": st.column_config.NumberColumn("Unit Cost", format="EGP %.2f"),
                        "estimated_value": st.column_config.NumberColumn("Estimated Value", format="EGP %.2f"),
                    },
                    use_container_width=True,
                    hide_index=True
                 )
            display_df = value_by_category # Export summary by default
        else:
            st.info("No inventory data available.")


    # --- Slow-Moving Stock Report ---
    elif selected_report == "Slow-Moving Stock Report":
        st.subheader("Slow-Moving Stock Report")
        st.markdown("Identify items that haven't been logged (used, adjusted, etc.) recently.")

        days_threshold = st.number_input(
            "Show items with no activity in the last (days):",
            min_value=30,
            max_value=365 * 2, # Max 2 years
            value=90, # Default to 90 days
            step=30,
            key="slow_days_thresh"
        )

        df_slow_items = data_loader.get_slow_moving_items(days_threshold)

        if not df_slow_items.empty:
            st.dataframe(
                df_slow_items,
                column_config={
                    "item_name": "Item Name",
                    "category": "Category",
                    "current_quantity": st.column_config.NumberColumn("Current Qty", format="%d"),
                    "last_log_timestamp": st.column_config.DatetimeColumn("Last Activity Logged", format="YYYY-MM-DD HH:mm"),
                    "last_updated": st.column_config.DatetimeColumn("Item Last Modified", format="YYYY-MM-DD HH:mm"),
                },
                 column_order=["item_name", "category", "current_quantity", "last_log_timestamp", "last_updated"],
                use_container_width=True,
                hide_index=True
            )
            display_df = df_slow_items
        else:
            st.info(f"No items found with zero activity logged in the last {days_threshold} days.")

    # --- Export Options ---
    if not display_df.empty:
        st.divider()
        st.subheader("📥 Export Report Data")

        export_col1, export_col2 = st.columns(2)
        # Sanitize report name for filename
        safe_report_name = selected_report.lower().replace(' ', '_')
        file_prefix = f"inventory_{safe_report_name}_report_{date.today()}"

        # CSV Export
        with export_col1:
            try:
                csv_data = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download as CSV",
                    data=csv_data,
                    file_name=f"{file_prefix}.csv",
                    mime="text/csv",
                    key=f"csv_download_{safe_report_name}"
                )
            except Exception as e:
                st.error(f"Error preparing CSV: {e}")

        # Excel Export
        with export_col2:
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    display_df.to_excel(writer, index=False, sheet_name='ReportData')
                excel_data = buffer.getvalue()
                st.download_button(
                    "Download as Excel",
                    data=excel_data,
                    file_name=f"{file_prefix}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"excel_download_{safe_report_name}"
                )
            except Exception as e:
                st.error(f"Error preparing Excel: {e}")
