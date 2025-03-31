# src/tabs/inventory_management.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from src import data_loader
import plotly.express as px # Added for charts
import plotly.graph_objects as go # For combining history and forecast plots
from src.ai import inventory_data_prep
from src.ai import inventory_predictor

def show():
    """Displays the Inventory Management tab."""
    st.title("Inventory Management")

    # --- Load Data ---
    df_inventory = data_loader.get_inventory_items()
    categories = ["All"] + data_loader.get_distinct_inventory_categories()

    # --- Sidebar Filters/Actions ---
    st.sidebar.header("Inventory Options")
    selected_category = st.sidebar.selectbox("Filter by Category", options=categories)
    low_stock_filter = st.sidebar.checkbox("Show Low Stock Items Only")
    expiring_soon_days = st.sidebar.number_input("Show Items Expiring Within (Days)", min_value=0, value=30)
    show_expiring = st.sidebar.checkbox("Filter by Expiration Date")

    # --- Main View ---
    st.subheader("Current Inventory")

    # Apply filters based on sidebar selections
    filtered_inventory = df_inventory.copy()
    if selected_category != "All":
        filtered_inventory = filtered_inventory[filtered_inventory['category'] == selected_category]
    if low_stock_filter:
        filtered_inventory = filtered_inventory[filtered_inventory['current_quantity'] <= filtered_inventory['reorder_level']]
    if show_expiring and expiring_soon_days > 0:
        today = date.today()
        target_date = today + timedelta(days=expiring_soon_days)
        # Ensure expiration_date is comparable (handle NaT)
        filtered_inventory = filtered_inventory[
            pd.notna(filtered_inventory['expiration_date']) &
            (filtered_inventory['expiration_date'] >= today) &
            (filtered_inventory['expiration_date'] <= target_date)
        ]
    elif show_expiring and expiring_soon_days == 0: # Show already expired
         today = date.today()
         filtered_inventory = filtered_inventory[
            pd.notna(filtered_inventory['expiration_date']) &
            (filtered_inventory['expiration_date'] < today)
        ]


    # Display Inventory Table
    if not filtered_inventory.empty:
        # Define columns to display and their order
        display_cols = [
            "item_name", "category", "current_quantity", "reorder_level",
            "unit_cost", "expiration_date", "supplier", "last_updated"
        ]
        # Reorder and select columns, handling potential missing columns gracefully
        cols_to_show = [col for col in display_cols if col in filtered_inventory.columns]
        st.dataframe(filtered_inventory[cols_to_show], use_container_width=True)

        # --- Alerts ---
        low_stock_items = df_inventory[df_inventory['current_quantity'] <= df_inventory['reorder_level']]
        if not low_stock_items.empty:
            st.warning(f"**Low Stock Alert:** {len(low_stock_items)} item(s) are at or below reorder level.")
            with st.expander("View Low Stock Items"):
                st.dataframe(low_stock_items[cols_to_show], use_container_width=True)

        today = date.today()
        expiring_soon_items = df_inventory[
            pd.notna(df_inventory['expiration_date']) &
            (df_inventory['expiration_date'] >= today) &
            (df_inventory['expiration_date'] <= today + timedelta(days=expiring_soon_days if expiring_soon_days else 30)) # Default 30 days if not specified
        ]
        if not expiring_soon_items.empty:
             st.warning(f"**Expiration Alert:** {len(expiring_soon_items)} item(s) are expiring within the next {expiring_soon_days if expiring_soon_days else 30} days.")
             with st.expander("View Items Expiring Soon"):
                 st.dataframe(expiring_soon_items[cols_to_show], use_container_width=True)

    else:
        st.info("No inventory items match the current filters.")

    # --- Inventory Analytics ---
    st.divider()
    st.subheader("Inventory Analytics")
    if not df_inventory.empty:
        # Calculate metrics
        total_items = df_inventory['current_quantity'].sum()
        distinct_categories = df_inventory['category'].nunique()
        # Calculate total value (handle potential NaN in unit_cost or quantity)
        df_inventory['item_value'] = pd.to_numeric(df_inventory['current_quantity'], errors='coerce').fillna(0) * \
                                     pd.to_numeric(df_inventory['unit_cost'], errors='coerce').fillna(0)
        total_value = df_inventory['item_value'].sum()

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Units", f"{total_items:,}")
        with col2:
            st.metric("Distinct Categories", f"{distinct_categories:,}")
        with col3:
            st.metric("Estimated Total Value", f"EGP {total_value:,.2f}")

        # Charts
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            # Items per category
            category_counts = df_inventory['category'].fillna('Uncategorized').value_counts().reset_index()
            category_counts.columns = ['Category', 'Number of Items']
            fig_cat_count = px.bar(category_counts, x='Category', y='Number of Items',
                                   title="Inventory Items per Category", text_auto=True)
            fig_cat_count.update_layout(xaxis_title=None)
            st.plotly_chart(fig_cat_count, use_container_width=True)

        with col_chart2:
            # Value per category
            category_value = df_inventory.groupby('category')['item_value'].sum().reset_index()
            category_value.columns = ['Category', 'Total Value']
            fig_cat_value = px.bar(category_value, x='Category', y='Total Value',
                                   title="Estimated Inventory Value per Category", text_auto='.2s')
            fig_cat_value.update_layout(yaxis_title="Total Value (EGP)", xaxis_title=None)
            st.plotly_chart(fig_cat_value, use_container_width=True)

    else:
        st.info("No inventory data available for analytics.")

    st.divider() # Add divider before management section

    # --- Add/Edit Inventory Item ---
    st.subheader("Manage Inventory Items")
    with st.expander("Add New Inventory Item"):
        with st.form("add_item_form", clear_on_submit=True):
            new_item_name = st.text_input("Item Name*", key="add_name")
            new_category = st.text_input("Category", key="add_cat")
            col_qty, col_reorder, col_max = st.columns(3)
            with col_qty:
                new_quantity = st.number_input("Current Quantity*", min_value=0, value=0, step=1, key="add_qty")
            with col_reorder:
                new_reorder = st.number_input("Reorder Level", min_value=0, value=0, step=1, key="add_reorder")
            with col_max:
                new_max_stock = st.number_input("Max Stock Level (Optional)", min_value=0, value=None, step=1, key="add_max_stock", help="Set a maximum desired stock level for overstock alerts.")

            new_cost = st.number_input("Unit Cost", min_value=0.0, value=0.0, format="%.2f", key="add_cost")
            new_exp_date = st.date_input("Expiration Date (Optional)", value=None, key="add_exp")
            new_supplier = st.text_input("Supplier", key="add_sup")

            submitted_add = st.form_submit_button("Add Item")
            if submitted_add:
                if not new_item_name or new_quantity is None:
                    st.error("Item Name and Current Quantity are required.")
                else:
                    success = data_loader.add_inventory_item(
                        item_name=new_item_name,
                        category=new_category if new_category else None,
                        current_quantity=new_quantity,
                        reorder_level=new_reorder,
                        max_stock_level=new_max_stock, # Added max stock level
                        unit_cost=new_cost if new_cost > 0 else None,
                        expiration_date=new_exp_date,
                        supplier=new_supplier if new_supplier else None
                    )
                    if success:
                        st.success(f"Item '{new_item_name}' added successfully!")
                        st.rerun() # Rerun to refresh the inventory list
                    # Error message is handled within add_inventory_item

    # --- Edit/Delete Section ---
    st.subheader("Edit or Delete Existing Item")
    if not df_inventory.empty:
        item_list = ["Select item to manage..."] + df_inventory['item_name'].tolist()
        selected_item_name = st.selectbox("Select Item", options=item_list, key="edit_select")

        if selected_item_name != "Select item to manage...":
            item_details = df_inventory[df_inventory['item_name'] == selected_item_name].iloc[0]
            try:
                item_id = int(item_details['id']) # Ensure ID is integer
            except (ValueError, TypeError):
                st.error(f"Invalid item ID format for {selected_item_name}.")
                st.stop() # Stop execution if ID is invalid

            with st.form("edit_item_form"):
                st.write(f"Editing: **{item_details['item_name']}** (ID: {item_id})")
                edit_category = st.text_input("Category", value=item_details.get('category', '') or '', key="edit_cat") # Handle None
                col_edit_qty, col_edit_reorder, col_edit_max = st.columns(3)
                with col_edit_qty:
                    edit_quantity = st.number_input("Current Quantity*", min_value=0, value=int(item_details.get('current_quantity', 0)), step=1, key="edit_qty")
                with col_edit_reorder:
                    edit_reorder = st.number_input("Reorder Level", min_value=0, value=int(item_details.get('reorder_level', 0)), step=1, key="edit_reorder")
                with col_edit_max:
                    # Handle potential None/NaN from DB for max_stock_level
                    current_max_stock = item_details.get('max_stock_level')
                    edit_max_stock = st.number_input("Max Stock Level (Optional)", min_value=0, value=int(current_max_stock) if pd.notna(current_max_stock) else None, step=1, key="edit_max_stock", help="Set a maximum desired stock level for overstock alerts.")

                edit_cost = st.number_input("Unit Cost", min_value=0.0, value=float(item_details.get('unit_cost', 0.0)), format="%.2f", key="edit_cost")
                # Handle potential NaT for date input
                current_exp_date = item_details.get('expiration_date')
                edit_exp_date = st.date_input("Expiration Date (Optional)", value=current_exp_date if pd.notna(current_exp_date) else None, key="edit_exp")
                edit_supplier = st.text_input("Supplier", value=item_details.get('supplier', ''), key="edit_sup")

                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    submitted_update = st.form_submit_button("Update Item")
                with col2:
                    submitted_delete = st.form_submit_button("Delete Item", type="primary") # Use primary for delete emphasis

                if submitted_update:
                    updates = {
                        "category": edit_category if edit_category else None,
                        "current_quantity": edit_quantity,
                        "reorder_level": edit_reorder,
                        "max_stock_level": edit_max_stock, # Added max stock level
                        "unit_cost": edit_cost if edit_cost > 0 else None,
                        "expiration_date": edit_exp_date,
                        "supplier": edit_supplier if edit_supplier else None
                        # item_name cannot be updated here to avoid complexity, delete and re-add if needed
                    }
                    success = data_loader.update_inventory_item(item_id, updates)
                    if success:
                        st.success(f"Item '{item_details['item_name']}' updated successfully!")
                        st.rerun()
                    # Error handled in update_inventory_item

                if submitted_delete:
                    # Add a confirmation step
                    if st.checkbox(f"Confirm deletion of '{item_details['item_name']}'?", key=f"del_confirm_{item_id}"):
                        success = data_loader.delete_inventory_item(item_id)
                        if success:
                            st.success(f"Item '{item_details['item_name']}' deleted successfully!")
                            st.rerun()
                        # Error handled in delete_inventory_item
                    else:
                        st.warning("Deletion not confirmed.")

            # --- AI Forecasting Section (within the selected item block) ---
            st.divider()
            st.subheader(f"AI-Assisted Forecast for {item_details['item_name']}")
            with st.spinner(f"Generating forecast for {item_details['item_name']}..."):
                # 1. Prepare data
                df_prepared = inventory_data_prep.get_preprocessed_item_history(item_id, item_details['item_name'])

                if not df_prepared.empty:
                    # 2. Get forecast
                    # Define forecast horizon (e.g., 60 days) and lead/safety times
                    forecast_days = 60
                    # TODO: Make lead_time and safety_stock configurable per item/category
                    lead_time_days = 7 # Example: 7 days lead time
                    safety_stock_days = 3 # Example: 3 days safety stock

                    df_forecast = inventory_predictor.predict_future_stock(item_id, df_prepared, days_ahead=forecast_days)

                    if not df_forecast.empty:
                        # 3. Display Forecast Chart
                        fig_forecast = go.Figure()
                        # Historical data
                        fig_forecast.add_trace(go.Scatter(x=df_prepared['ds'], y=df_prepared['y'], mode='lines', name='Historical Quantity'))
                        # Forecast line
                        fig_forecast.add_trace(go.Scatter(x=df_forecast['ds'], y=df_forecast['yhat'], mode='lines', name='Predicted Quantity', line=dict(dash='dash')))
                        # Confidence interval
                        fig_forecast.add_trace(go.Scatter(
                            x=df_forecast['ds'].tolist() + df_forecast['ds'].tolist()[::-1], # x values for fill
                            y=df_forecast['yhat_upper'].tolist() + df_forecast['yhat_lower'].tolist()[::-1], # y values for fill
                            fill='toself',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=False,
                            name='Confidence Interval'
                        ))
                        # Add current reorder level line
                        current_reorder_level = item_details.get('reorder_level', 0)
                        if pd.notna(current_reorder_level) and current_reorder_level > 0:
                             fig_forecast.add_hline(y=current_reorder_level, line_dash="dot", line_color="grey",
                                   annotation_text=f"Manual Reorder ({int(current_reorder_level)})",
                                   annotation_position="bottom right")

                        fig_forecast.update_layout(title=f"Stock Level Forecast ({forecast_days} days)",
                                                   xaxis_title="Date", yaxis_title="Quantity", hovermode="x unified")
                        st.plotly_chart(fig_forecast, use_container_width=True)

                        # 4. Calculate and Display AI Recommendations
                        st.subheader("AI Recommendations")
                        try:
                            # Find predicted quantity at the end of lead time and safety stock period
                            today_dt = pd.to_datetime(date.today())
                            lead_time_end_date = today_dt + timedelta(days=lead_time_days)
                            safety_stock_end_date = lead_time_end_date + timedelta(days=safety_stock_days)

                            # Get predicted value closest to the target dates
                            pred_at_lead_time = df_forecast[df_forecast['ds'] <= lead_time_end_date]['yhat'].iloc[-1] if not df_forecast[df_forecast['ds'] <= lead_time_end_date].empty else df_prepared['y'].iloc[-1]
                            pred_at_safety_end = df_forecast[df_forecast['ds'] <= safety_stock_end_date]['yhat'].iloc[-1] if not df_forecast[df_forecast['ds'] <= safety_stock_end_date].empty else pred_at_lead_time

                            # Calculate predicted usage during lead time
                            current_qty_for_calc = df_prepared['y'].iloc[-1] # Use last known historical value
                            predicted_usage_lead_time = max(0, current_qty_for_calc - pred_at_lead_time)
                            predicted_usage_safety = max(0, pred_at_lead_time - pred_at_safety_end)

                            # AI Reorder Point: Level needed to cover safety stock usage after lead time
                            # Simplified: The predicted level after lead_time + safety_stock days
                            # More robust: Level that ensures stock doesn't drop below safety threshold during lead time
                            # Let's use a simpler definition for now: predicted quantity needed for safety stock period
                            ai_safety_stock_level = predicted_usage_safety # How much is used during safety period
                            # AI Reorder Point = Level needed to survive lead time + safety stock
                            # Find the lowest predicted point within lead_time + safety_stock_days
                            lookahead_period = df_forecast[df_forecast['ds'] <= safety_stock_end_date]
                            min_pred_in_lookahead = lookahead_period['yhat_lower'].min() if not lookahead_period.empty else current_qty_for_calc

                            # Define AI Reorder Point as the level needed to ensure stock stays above 0 (or a minimum safety level)
                            # during the lead time. Let's aim to have `ai_safety_stock_level` remaining when new stock arrives.
                            ai_reorder_point = ai_safety_stock_level + predicted_usage_lead_time
                            ai_reorder_point = max(0, round(ai_reorder_point)) # Ensure non-negative and integer

                            # Suggested Order Quantity
                            # Aim to reach max stock level if defined, otherwise aim for reorder point + buffer
                            max_stock = item_details.get('max_stock_level')
                            target_stock = int(max_stock) if pd.notna(max_stock) and max_stock > 0 else ai_reorder_point + predicted_usage_lead_time # Target buffer above reorder point
                            suggested_order_qty = max(0, round(target_stock - current_qty_for_calc))

                            col_ai1, col_ai2 = st.columns(2)
                            with col_ai1:
                                st.metric("AI Reorder Point", f"{ai_reorder_point}",
                                          help=f"Suggested reorder level based on predicted usage during lead time ({lead_time_days} days) + safety stock ({safety_stock_days} days).")
                            with col_ai2:
                                st.metric("Suggested Order Qty", f"{suggested_order_qty}",
                                          help=f"Estimated quantity to order now to reach target stock level ({target_stock}).")

                        except Exception as e:
                            st.error(f"Error calculating AI recommendations: {e}", icon="📉")

                    else:
                        st.warning(f"Could not generate forecast for {item_details['item_name']}.", icon="⚠️")
                # else: # Warning already shown by data prep function
                #    st.info(f"Insufficient data to generate forecast for {item_details['item_name']}.")

            # Item history log is now displayed in the 'Inventory Tracking' tab.

    else:
        st.info("No inventory items available to edit or delete.")


    # --- Stock Adjustment Section ---
    st.divider()
    st.subheader("Stock Adjustment / Physical Count")
    if not df_inventory.empty:
        item_list_adjust = ["Select item to adjust..."] + df_inventory['item_name'].tolist()
        selected_item_adjust_name = st.selectbox("Select Item for Adjustment", options=item_list_adjust, key="adjust_select")

        if selected_item_adjust_name != "Select item to adjust...":
            item_adjust_details = df_inventory[df_inventory['item_name'] == selected_item_adjust_name].iloc[0]
            try:
                item_adjust_id = int(item_adjust_details['id']) # Ensure ID is integer
            except (ValueError, TypeError):
                st.error(f"Invalid item ID format for {selected_item_adjust_name}.")
                st.stop() # Stop execution if ID is invalid
            current_qty_display = item_adjust_details['current_quantity']

            with st.form("adjust_stock_form", clear_on_submit=True):
                st.write(f"Adjusting: **{item_adjust_details['item_name']}**")
                st.caption(f"Current System Quantity: {current_qty_display}")

                physical_quantity = st.number_input("Enter Actual Physical Quantity*", min_value=0, step=1, key="adjust_physical_qty")

                submitted_adjust = st.form_submit_button("Update Stock Count")

                if submitted_adjust:
                    if physical_quantity is None:
                         st.error("Please enter the actual physical quantity.")
                    else:
                        updates = {"current_quantity": physical_quantity}
                        success = data_loader.update_inventory_item(
                            item_id=item_adjust_id,
                            updates=updates,
                            change_type_override='Stock Adjustment' # Specify the log type
                        )
                        if success:
                            st.success(f"Stock for '{item_adjust_details['item_name']}' adjusted to {physical_quantity}.")
                            st.rerun()
                        # Error handled in update_inventory_item
    else:
        st.info("No inventory items available to adjust.")
