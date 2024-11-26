import streamlit as st
import data_collection_preprocessing

@st.cache_data
def get_data():
    table_name = "commodity_data"

    # Check if data exists in MySQL
    if data_collection_preprocessing.table_exists(table_name):
        st.write("Fetching data from MySQL...")
        return data_collection_preprocessing.fetch_from_mysql(table_name)
    else:
        st.write("Running data preprocessing and saving to MySQL...")
        return data_collection_preprocessing.preprocess_and_save_to_mysql(table_name)

def main():
    if "user" not in st.session_state:
        st.session_state["user"] = None

    if st.session_state["user"]:
        st.title("Food-Related Commodity Data Viewer")
        st.sidebar.success(f"Welcome {st.session_state['user']}!")
        if st.sidebar.button("Logout"):
            st.session_state["user"] = None
            st.rerun()

         # Load data
        df = get_data()

        # Display data
        st.write("### Processed Commodity Data")
        st.dataframe(df)
    else:
        st.title("Please log in to access the dashboard.")

   

# if __name__ == "__main__":
#     main()

#________________________________________

# import streamlit as st
# from streamlit_login_auth_ui.widgets import __login__

# __login__obj = __login__(auth_token = "courier_auth_token", 
#                     company_name = "Shims",
#                     width = 200, height = 250, 
#                     logout_button_name = 'Logout', hide_menu_bool = False, 
#                     hide_footer_bool = False, 
#                     lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

# LOGGED_IN = __login__obj.build_login_ui()

# if LOGGED_IN == True:

#     st.markown("Your Streamlit Application Begins here!")
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
