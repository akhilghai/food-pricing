import streamlit as st
import data_collection_preprocessing
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import train_model
import pickle



def format_green(text:str):
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h2 style="color: #4caf50;">{text}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_data
def get_data():
    table_name = "commodity_data"

    # Check if data exists in MySQL
    if data_collection_preprocessing.table_exists(table_name):
        # st.write("Fetching data from MySQL...")
        return data_collection_preprocessing.fetch_from_mysql(table_name)
    else:
        # st.write("Running data preprocessing and saving to MySQL...")
        return data_collection_preprocessing.preprocess_and_save_to_mysql(table_name)

def main():
    if "user" not in st.session_state:
        st.session_state["user"] = None

    if st.session_state["user"]:
        st.sidebar.success(f"Welcome {st.session_state['user']}!")

        # Display the image in the sidebar
        
        st.sidebar.image('user.png')
        st.sidebar.markdown("---")

        # Initialize a placeholder for progress bar
        progress_placeholder = st.empty()

        with progress_placeholder.container():  # Use a placeholder for progress bar
            # Add a progress bar
            with st.spinner("Loading data and training models..."):
                # Progress initialization
                progress_bar = st.progress(0)

                # Load data
                combined_data = get_data()
                progress_bar.progress(33)  # Update progress to 33%

                # Directory to store models and reports
                MODEL_DIR = "models_and_reports"
                # Ensure the directory exists
                if not os.path.exists(MODEL_DIR):
                    os.makedirs(MODEL_DIR)
                    train_model.train_model_and_save(combined_data,MODEL_DIR)
                progress_bar.progress(100)  # Update progress to 100%
        # Remove progress bar after completion
        progress_placeholder.empty()

        
        with st.sidebar:
            selected_page = option_menu(
                menu_title="Menu",
                options=["Overview", "General Statistics","Trends","Model Building & Evaluation", "Forecasting", 
                        "About"],
                icons=["house", "bar-chart-line", "bi-graph-up-arrow","bi-gear","bi-rocket", "bi-info-square-fill"],
                menu_icon="cast",
                default_index=0,
                orientation="vertical",
                key="navigation_menu"
            )



        st.sidebar.markdown("---")
        
        if st.sidebar.button("Logout"):
            st.session_state["user"] = None
            st.rerun()



        # Display Body
        st.markdown(
            """
            <div style="text-align: center;">
                <h1 style="color: #4caf50;">U.S. Food Price Forecasting</h1>
                <h2>Predicting Future Trends for U.S. Food Commodities</h2>
            </div>
            <div>
                <p>This project aims to develop a statistical and forecasting tool to analyze the <b>closing prices</b> of food commodities in the United States. 
                By leveraging data from Yahoo and applying advanced statistical models and forecasting techniques, 
                the app provides insights into future price trends for a variety of food items. 
                This tool helps stakeholders make informed decisions regarding pricing, supply chain management, 
                and budgeting, ensuring the accuracy and reliability of price predictions in the food market.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if(selected_page=="Overview"):
            st.markdown('---')
            format_green("Overview")

            # Product Overview Section
            st.markdown(
                """
                The Commodity Configuration allows users to track and forecast prices of key agricultural commodities in the U.S. market. 
                Data for these commodities is fetched from Yahoo Finance for comprehensive analysis.
                
                - **Commodities Included**:- Corn, Wheat, Soybeans, Sugar and Coffee. These commodities are crucial indicators for agricultural markets, and the application enables detailed statistical and forecasting analysis.
                - **Data Source**: Yahoo Finance - The data is retrieved from Yahoo Finance, a trusted source for commodity and financial market data.
                - **Coverage**: Data spans five core agricultural commodities since January 2011.
                - **Data Organization**: Access is via programmatic queries to Yahoo Finance's API, allowing seamless integration for advanced forecasting.
                """
            )

            # Create DataFrame for Commodity Information
            commodity_data = pd.DataFrame(list(data_collection_preprocessing.commodity_tickers.items()), columns=["Commodity", "Ticker"])
            commodity_data['Units'] = commodity_data['Commodity'].map((data_collection_preprocessing.commodity_units))

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Commodity Configuration")
                st.dataframe(commodity_data,hide_index=True,use_container_width = True)
            with col2:
                st.markdown("### Data Specifications")
                st.dataframe(pd.DataFrame(list(data_collection_preprocessing.data_Specifications.items()), columns=["Data", "Specifications"]),hide_index=True,use_container_width = True)
                


            
            st.write("### Final Commodity Data after Data Collection , Pre-processing and Feature Eng.")
            st.dataframe(combined_data)
            st.write(f"rows: {combined_data.shape[0]} and columns: {combined_data.shape[1]}")

            # Display the Data Pipeline flow as markdown
            st.markdown("""
            ### Data Pipeline Overview

            The following steps represent the pipeline that extracts, transforms, and exports commodity data, followed by feature engineering and providing the data to the end user for analysis.


            - #### 1. **Data Extraction**
                - **Source**: Yahoo Finance
                - **Step**: Fetch commodity data for specific commodities: Corn, Wheat, Soybeans, Sugar, Coffee.
                - **Inputs**: 
                    - Commodity tickers (`'corn': 'ZC=F'`, `'wheat': 'ZW=F'`, etc.)
                    - Start and end dates for data extraction.


            - #### 2. **Data Transformation and Integration**
                - **Step**: Process and transform the fetched data to make it ready for analysis.
                - **Substeps**:
                    - **Unit Conversion**: Convert the commodity data units as needed.
                    - **Data Integration**: Combine different data sources into a cohesive dataset.
                    - **Monthly Re-Sampling**: Adjust the data to ensure consistency in time intervals.


            - #### 3. **Data Quality Handling**
                - **Step**: Ensure data quality by handling missing values.
                - **Substeps**:
                    - **Missing Data Detection**: Identifies gaps in the data range to find missing months.
                    - **KNN Imputation**: Uses K-Nearest Neighbors (KNN) to impute missing or NaN values.


            - #### 4. **Feature Engineering**
                - **Step**: Analyze the data for meaningful patterns.
                - **Substeps**:
                    - **Year-over-Year Percentage Change**: Computes the 12-month percentage change for each commodity to analyze annual price trends.


            - #### 5. **Data Export**
                - **Step**: Export the processed data for use.
                - **Substeps**:
                    - **Export Process**: Data is exported as a CSV file for further analysis or reporting.
                    - **Output**: A downloadable CSV file of the processed commodity data.


            - #### 6. **End User**
                - **Final Step**: The processed data is available for the end user to analyze, report, or further process.
                """)
            st.image('Pipeline.jpg', width=800)
 
            # Download Button for CSV Export
            st.markdown("### Data Export")
            st.markdown("""
            **Bulk Download**  
            You can download an entire table with one of the following links. The output format is always the same: a single CSV file containing the entire data table.
            """)

            # Create Download Buttons
            st.download_button(
                label="Download Commodity Data",
                data=combined_data.to_csv(index=False),
                file_name="commodity_data.csv",
                mime="text/csv",
                icon=":material/file_save:"
            )

            # Create Download Buttons
            st.download_button(
                label="Download Commodity Configuration",
                data=commodity_data.to_csv(index=False),
                file_name="commodity_configuration.csv",
                mime="text/csv",
                icon = ":material/file_save:"
            )
        if(selected_page=="General Statistics"):
            st.markdown('---')
            format_green("General Statistics")
            tab1, tab2 = st.tabs(["Data Summary", "Graphical Analysis"])

            with tab1:
                st.markdown("#### Basic statistical measures, including mean, median, and standard deviation, for key metrics")
                st.write(combined_data[["corn","wheat","soybeans","sugar","coffee"]].describe())
                # Plot correlation map
                st.markdown("### Correlation Map between food commodities")
                fig, ax = plt.subplots(figsize=(25, 4))   # Adjust the figure size as needed
                sns.heatmap( 
                    combined_data[["corn","wheat","soybeans","sugar","coffee"]].corr(),
                    annot=True,
                    fmt=".2f", 
                    cmap="Blues",  
                    cbar=True,
                    vmin=0.4,
                    vmax=1, 
                    square=True,
                    linewidths=0.5,
                    ax=ax,
                    )
                st.pyplot(fig,clear_figure = True,use_container_width=False) 

            with tab2:
                
                commodity_filter = st.selectbox(
                    f"#### Please select the food commodity for analysis",
                    options = data_collection_preprocessing.commodity_tickers.keys(),
                )
 
                
                st.markdown(f"### Distribution of {commodity_filter}")
                fig = ff.create_distplot(
                        [combined_data[commodity_filter].dropna()], [commodity_filter],[0.1, 0.25, 0.5])
                # Update axis labels
                fig.update_layout(
                    xaxis_title=f"Price in {data_collection_preprocessing.commodity_units[commodity_filter]}",  # Custom x-axis label
                    yaxis_title="Frequency"     # Custom y-axis label
                )
                # Plot!
                st.plotly_chart(fig, use_container_width=True)

                # Q-Q plot
                st.markdown(f"### Q-Q Plot of {commodity_filter} Price")
                data = combined_data[commodity_filter].dropna()  # Use raw data, cleaned of NaNs
                (quantiles, values), _ = stats.probplot(data, dist="norm")  # Compute quantiles and values
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=quantiles, y=values, mode="markers", name="Data"))  # Scatter plot of Q-Q points
                
                slope, intercept = np.polyfit(quantiles, values, 1)  # Fit a line to the data
                fig.add_trace(go.Scatter(x=quantiles, y=slope * quantiles + intercept, mode="lines", name="Fit Line"))  # Fit line
                
                fig.update_layout(
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles"
                )
                # Show the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)


                st.markdown(f"### Boxplot of {commodity_filter}")
                fig=px.box(data_frame=combined_data[commodity_filter],
                            # y=combined_data[["corn","wheat","soybeans","sugar","coffee"]].columns,  # Use the columns as the categories
                            # color=combined_data[["corn","wheat","soybeans","sugar","coffee"]].columns,  # Color each commodity differently
                            )
                # Update axis labels
                fig.update_layout(
                    xaxis_title="Commodity",  # Custom x-axis label
                    yaxis_title=f"Price in {data_collection_preprocessing.commodity_units[commodity_filter]}",        # Custom y-axis label
                )
                st.plotly_chart(fig) 
        if(selected_page=="Trends"):
            st.markdown('---')
            format_green("Trends")
            trend_filter = st.selectbox(
                    f"#### Please select the food commodity for analysis",
                    options = data_collection_preprocessing.commodity_tickers.keys(),
                )

            # Date slider (2011 to 2024)
            start_year, end_year = st.slider(
                'Select a Year Range',
                min_value=2000,
                max_value=2024,
                value=(2000, 2024),
                step=1
            )

            st.markdown(f'### Price Trend')
            # Trend plot for corn (using the cached data)
            fig = px.line(combined_data, x='Date', y=trend_filter)
            fig.update_layout(xaxis_title='Date', yaxis_title=f"{trend_filter} Price in {data_collection_preprocessing.commodity_units[trend_filter]}",
                              xaxis=dict(
                                range=[f"{start_year}-01-01", f"{end_year}-12-31"]  # Update x-axis range
                            ))
            # Display the plot in Streamlit
            st.plotly_chart(fig)

            st.markdown(f'### Percentage Changes (Month-over-Month) Over Time')
            # Trend plot for corn (using the cached data)
            fig = px.line(combined_data, x='Date', y=f'{trend_filter}_m/m-12')
            fig.update_layout(xaxis_title='Date', yaxis_title='Percentage Change (%)',
                              xaxis=dict(
                                range=[f"{start_year}-01-01", f"{end_year}-12-31"]  # Update x-axis range
                            ))
            # Display the plot in Streamlit
            st.plotly_chart(fig)

            st.markdown(f'### Seasonal Decomposition of {trend_filter}')
            result = seasonal_decompose(combined_data[trend_filter].dropna(), model='additive', period=12)
            # Plot observed, trend, seasonal, and residual
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=combined_data['Date'], y=result.observed, mode='lines', name='Observed'))
            fig.add_trace(go.Scatter(x=combined_data['Date'], y=result.trend, mode='lines', name='Trend'))
            fig.add_trace(go.Scatter(x=combined_data['Date'], y=result.seasonal, mode='lines', name='Seasonal'))
            fig.add_trace(go.Scatter(x=combined_data['Date'], y=result.resid, mode='lines', name='Residual'))
            fig.update_layout(xaxis_title='Date', yaxis_title='Value',
                              xaxis=dict(
                                range=[f"{start_year}-01-01", f"{end_year}-12-31"]  # Update x-axis range
                            ))
            st.plotly_chart(fig)


            st.markdown(f'### Rolling Statistics of {trend_filter}')
            window = 12
            rolling_mean = combined_data[trend_filter].rolling(window=window).mean()
            rolling_std = combined_data[trend_filter].rolling(window=window).std()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=combined_data['Date'], y=combined_data[trend_filter], mode='lines', name='Original'))
            fig.add_trace(go.Scatter(x=combined_data['Date'], y=rolling_mean, mode='lines', name=f'{window}-Month Rolling Mean', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=combined_data['Date'], y=rolling_std, mode='lines', name=f'{window}-Month Rolling Std Dev', line=dict(color='red')))
            fig.update_layout( xaxis_title='Date', yaxis_title='Value',
                              xaxis=dict(
                                range=[f"{start_year}-01-01", f"{end_year}-12-31"]  # Update x-axis range
                            ))
            st.plotly_chart(fig)
        if(selected_page=="Model Building & Evaluation"):
            # Placeholder for combined forecast data
            combined_forecast_ml = pd.DataFrame()


            # Define the layout for the plot
            layout_ml = go.Layout(
                title='Commodity Forecasting - Machine Learning Approach',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Commodity Price'),
                hovermode='closest',
            )

            # Forecasting and model training/testing loop
            for commodity in data_collection_preprocessing.commodity_tickers.keys():
                model_path = os.path.join(MODEL_DIR, f"{commodity}_rf_model.pkl")
                report_path = os.path.join(MODEL_DIR, "evaluation_metrics.json")

                # # Check if the model and report exist
                # if os.path.exists(model_path) and os.path.exists(report_path):
                print(f"Loading saved model for {commodity}...")
                # Load the model
                with open(model_path, 'rb') as f:
                    rf_model = pickle.load(f)
            # Load the report
            with open(report_path, 'r') as f:
                report = f.read()
            print(report)

            st.json(report)              

        # Footer
        st.markdown("""
            <div style="background-color:#f1f1f1;padding:20px;text-align:center;margin-top:30px;">
                <p style="color:#333;">U.S. Food Pricing Forecasting - All Rights Reserved</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.title("Please log in to access the dashboard.")


#_____________________________________

# import streamlit as st

# # Set page title and layout
# st.set_page_config(page_title="U.S. Food Pricing Forecasting", layout="wide")


# # Header
# st.markdown("""
#     <div style="background-color:#0073e6;padding:10px;text-align:center;">
#         <h1 style="color:white;">U.S. Food Price Forecasting</h1>
#     </div>
# """, unsafe_allow_html=True)

# # Sidebar (Right Sidebar)
# # Sidebar radio options
# page = st.sidebar.radio("Select a Page", ["Summary", "Trends", "Forecast", "About"])

# # Content based on the selected page
# if page == "Summary":

#     # How it's built section
#     st.markdown("""
#     ### 🏗️ How It's Built
#     Stockastic is built with these core frameworks and modules:

#     - **Streamlit** - To create the web app UI and interactivity
#     - **YFinance** - To fetch financial data from Yahoo Finance API
#     - **StatsModels** - To build the ARIMA time series forecasting model
#     - **Plotly** - To create interactive financial charts

#     The app workflow is:

#     1. User selects a stock ticker
#     2. Historical data is fetched with YFinance
#     3. ARIMA model is trained on the data
#     4. Model makes multi-day price forecasts
#     5. Results are plotted with Plotly
#     """)

#     # Key features section
#     st.markdown("""
#     ### 🎯 Key Features
#     - **Real-time data** - Fetch latest prices and fundamentals
#     - **Financial charts** - Interactive historical and forecast charts
#     - **ARIMA forecasting** - Make statistically robust predictions
#     - **Backtesting** - Evaluate model performance
#     - **Responsive design** - Works on all devices
#     """)

#     # Getting started section
#     st.markdown("""
#     ### 🚀 Getting Started

#     #### Local Installation

#     1. Clone the repo:
#     ```bash
#     git clone <To Be Filled> """)
            
# elif page == "Trends":
#     st.write("### Trends")
#     st.write("Here you can show the trends for U.S. food prices.")
# elif page == "Forecast":
#     st.write("### Forecast")
#     st.write("Here you can show the forecasting models for U.S. food prices.")
# elif page == "About":
#     st.write("### About")
#     st.write("This project aims to predict the trends in food pricing in the U.S. over time.")

# # Footer
# st.markdown("""
#     <div style="background-color:#f1f1f1;padding:20px;text-align:center;margin-top:30px;">
#         <p style="color:#333;">U.S. Food Pricing Forecasting - All Rights Reserved</p>
#     </div>
# """, unsafe_allow_html=True)
