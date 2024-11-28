import streamlit as st
import data_collection_preprocessing
from streamlit_option_menu import option_menu
import pandas as pd

def format_green(text:str):
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h2 style="color: #adcbbe;">{text}</h2>
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

         # Load data
        combined_data = get_data()

        with st.sidebar:
            selected_page = option_menu(
                menu_title="Menu",
                options=["Overview", "Statistics", "Forecasting", 
                        "About"],
                icons=["house", "bar-chart-line", "bi-graph-up-arrow", "bi-info-square-fill", ],
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
                <h1 style="color: #adcbbe;">U.S. Food Price Forecasting</h1>
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
                


            
            st.write("### Extraced and Pre-Processed Commodity Data")
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
        # if(selected_page=="Statistics"):

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
