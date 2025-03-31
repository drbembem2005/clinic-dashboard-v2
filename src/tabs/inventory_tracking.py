# src/tabs/inventory_tracking.py
import streamlit as st
import pandas as pd
from src import data_loader
import plotly.express as px
import plotly.graph_objects as go # To be used later if we add forecast charts here
import math # For ceiling function
from src.ai import inventory_data_prep
from src.ai import inventory_predictor

def get_status_info(current_qty, reorder_level, max_stock_level):
    """Determines the status text, color, and progress value for an item."""
    status = "✅ OK"
    color = "green"
    progress_value = 0.5 # Default to middle if no levels set

    # Ensure levels are numeric and valid before comparison
    reorder_level = pd.to_numeric(reorder_level, errors='coerce')
    max_stock_level = pd.to_numeric(max_stock_level, errors='coerce')
    current_qty = pd.to_numeric(current_qty, errors='coerce')

    if pd.isna(current_qty):
        return "❓ Unknown Qty", "grey", 0.0 # Handle case where current quantity is somehow invalid

    is_reorder_valid = pd.notna(reorder_level) and reorder_level >= 0
    is_max_valid = pd.notna(max_stock_level) and max_stock_level > 0

    if is_reorder_valid and current_qty <= reorder_level:
        status = f"⚠️ Below Reorder ({int(reorder_level)})"
        color = "red"
        progress_value = 0.1 # Low progress for below reorder
    elif is_max_valid and current_qty > max_stock_level:
        status = f"🟧 OVERSTOCK (Max: {int(max_stock_level)})" # Changed icon and text
        color = "orange"
        progress_value = 1.0 # Full progress for overstock
    elif is_reorder_valid and is_max_valid and reorder_level < max_stock_level:
        # Calculate progress between reorder and max levels
        if current_qty > reorder_level:
             # Scale progress between reorder (just above 0.1) and max (just below 1.0)
             range_size = max_stock_level - reorder_level
             position_in_range = current_qty - reorder_level
             progress_value = 0.2 + 0.7 * (position_in_range / range_size) # Scale between 0.2 and 0.9
             progress_value = max(0.2, min(progress_value, 0.9)) # Clamp value
        else: # Should be caught by the first condition, but as fallback
             progress_value = 0.1
    elif is_reorder_valid: # Only reorder level is set
        if current_qty > reorder_level:
            progress_value = 0.7 # Indicate it's above reorder
        else:
             progress_value = 0.1
    elif is_max_valid: # Only max level is set
        if current_qty <= max_stock_level:
            progress_value = 0.7 # Indicate it's below max
        else:
            progress_value = 1.0

    return status, color, progress_value


def show():
    """Displays the Inventory Tracking tab with Status Summary and Detailed History."""
    st.title("Inventory Tracking & Status")

    # --- Load Inventory Data ---
    df_inventory = data_loader.get_inventory_items()
    if df_inventory.empty:
        st.warning("No inventory items found. Please add items in the Inventory Management tab first.")
        st.stop()

    # --- Inventory Status Summary ---
    st.subheader("Inventory Status Summary")
    st.markdown("Overview of all inventory items with status indicators.")

    # Display all items, sorted
    all_items_sorted = df_inventory.sort_values(by=['category', 'item_name'])

    if not all_items_sorted.empty:
        num_items = len(all_items_sorted)
        # Adjust columns based on total items, maybe more columns if many items?
        # Let's stick to max 4 for now to avoid excessive width.
        cols = st.columns(min(num_items, 4))
        item_index = 0
        for _, item in all_items_sorted.iterrows():
            with cols[item_index % len(cols)]:
                # Ensure quantity is treated as integer for display if possible
                try:
                    current_qty_display = int(item['current_quantity'])
                except (ValueError, TypeError):
                     current_qty_display = item['current_quantity'] # Fallback if not convertible

                current_qty = item['current_quantity'] # Keep original for logic
                reorder_level = item['reorder_level']
                max_stock = item['max_stock_level']
                status, color, progress = get_status_info(current_qty, reorder_level, max_stock)

                st.markdown(f"**{item['item_name']}**")
                st.caption(f"Category: {item.get('category', 'N/A')}")
                st.metric(label="Current Quantity", value=f"{int(current_qty)}")

                # Display status with color
                st.markdown(f"Status: <span style='color:{color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)

                # Display progress bar reflecting status
                st.progress(progress)

                # --- AI Forecast Metrics (in Status Card) ---
                with st.spinner(f"AI Forecast..."): # Keep it brief, it's per item
                    df_prepared_card = inventory_data_prep.get_preprocessed_item_history(item['id'], item['item_name'])
                    if not df_prepared_card.empty:
                        forecast_days_card = 30 # Shorter forecast for card view
                        df_forecast_card = inventory_predictor.predict_future_stock(item['id'], df_prepared_card, days_ahead=forecast_days_card)
                        if not df_forecast_card.empty:
                            # Get AI Recommendations (simplified for card view)
                            pred_qty_30_days = df_forecast_card['yhat'].iloc[-1] # Predicted qty in 30 days
                            ai_reorder_point_card = max(0, int(pred_qty_30_days * 1.1)) # Example: Reorder at ~10% above predicted in 30 days
                            suggested_order_qty_card = max(0, int(ai_reorder_point_card - current_qty)) # Simple suggestion

                            st.metric("AI Reorder Point", f"{ai_reorder_point_card}")
                            st.metric("Order Suggestion", f"{suggested_order_qty_card}")
                        else:
                            st.caption("No AI Forecast Available") # If forecast fails for card
                    else:
                        st.caption("No AI Data Available") # If no history for card

                st.markdown("---") # Separator

            item_index += 1
    else:
        st.success("✅ All inventory items are within defined stock levels.")

    st.divider()

    # --- Detailed History (in Expander) ---
    with st.expander("Detailed History & Log", expanded=False):
        st.subheader("Track Specific Item History")
        st.write("Visualize the quantity changes for a specific inventory item over time.")

        item_names = ["Select an item..."] + sorted(df_inventory['item_name'].unique().tolist())

        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            selected_item_name = st.selectbox("Select Inventory Item to Track", options=item_names, key="history_item_select")
        with col2:
            show_purchases_only = st.checkbox("Show Purchases/Additions Only", value=False, key="history_purchases_only")

        if selected_item_name != "Select an item...":
            # Get details for the selected item name
            item_details = df_inventory[df_inventory['item_name'] == selected_item_name].iloc[0]
            item_id = item_details['id']
            reorder_level_hist = item_details.get('reorder_level', 0)
            max_stock_level_hist = item_details.get('max_stock_level') # Might be None or NaN

            # --- Load Log Data ---
            log_filter = 'Purchases/Additions' if show_purchases_only else None
            df_log = data_loader.get_inventory_log(item_id, change_type_filter=log_filter)

            if not df_log.empty:
                st.markdown(f"#### Tracking History for: {selected_item_name}")

                # Ensure timestamp is sorted for plotting
                df_log = df_log.sort_values(by='timestamp')

                # --- Quantity Over Time Chart ---
                chart_title = 'Quantity Over Time' + (' (Purchases/Additions Only)' if show_purchases_only else '')
                fig = px.line(df_log, x='timestamp', y='new_quantity', title=chart_title, markers=True)

                # Add reorder level line
                if pd.notna(reorder_level_hist) and reorder_level_hist > 0:
                    fig.add_hline(y=reorder_level_hist, line_dash="dot", line_color="red",
                                  annotation_text=f"Reorder Level ({int(reorder_level_hist)})",
                                  annotation_position="bottom right")

                # Add max stock level line if defined
                if pd.notna(max_stock_level_hist) and max_stock_level_hist > 0:
                     fig.add_hline(y=max_stock_level_hist, line_dash="dot", line_color="orange",
                                   annotation_text=f"Max Stock Level ({int(max_stock_level_hist)})",
                                   annotation_position="top right")


                fig.update_layout(
                    xaxis_title='Timestamp',
                    yaxis_title='Quantity On Hand'
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- Detailed Log Table ---
                st.markdown("#### Detailed Change Log")
                # Format timestamp for display
                df_log_display = df_log.copy()
                df_log_display['timestamp'] = df_log_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(
                    df_log_display[['timestamp', 'change_type', 'quantity_change', 'new_quantity', 'notes']].sort_values(by='timestamp', ascending=False),
                    use_container_width=True,
                    hide_index=True # Hide the default index column
                )
            else:
                st.info(f"No tracking history found for '{selected_item_name}'.")
        else:
            st.info("Select an inventory item from the dropdown above to view its detailed tracking history.")
