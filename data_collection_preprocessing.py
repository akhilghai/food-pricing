from sqlalchemy import MetaData, Table, Column, DateTime, Float
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, inspect
from sklearn.impute import KNNImputer
import warnings
import os

warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv

# CA_CERT_PATH = 'ca.pem'
load_dotenv()  # Load the .env file


# Aiven MySQL Connection Details
DB_CONFIG = {
    "user": os.getenv("DB_USER"),  # Fetch user from the environment variable
    "password": os.getenv("DB_PASSWORD"),  # Fetch password from the environment variable
    "host": os.getenv("DB_HOST"),  # Fetch host from the environment variable
    "database": os.getenv("DB_NAME"),  # Fetch database from the environment variable
    "port": int(os.getenv("DB_PORT")),  # Fetch port from the environment variable
    "ssl_ca": os.getenv("SSL_CA_PATH"),  # Fetch SSL certificate path from the environment variable
    "ssl_disabled": False  # SSL is required, so it's False
}

# Commodity Configuration
commodity_tickers = {
    'corn': 'ZC=F',
    'wheat': 'ZW=F',
    'soybeans': 'ZS=F',
    'sugar': 'SB=F',
    'coffee': 'KC=F'
}

conversion_factors = {
    'corn': 0.01,
    'wheat': 0.01,
    'soybeans': 0.01,
    'sugar': 0.01,
    'coffee': 0.01
}

start_date = '2000-01-01'
end_date = '2024-12-31'

# Function to check if table exists in MySQL with SSL support
def table_exists(table_name):
    # Create a connection string with SSL enabled
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?ssl_ca={DB_CONFIG['ssl_ca']}&ssl_disabled={DB_CONFIG['ssl_disabled']}"
    )
    
    # Create an inspector object to inspect the database schema
    inspector = inspect(engine)
    
    # Check if the table exists in the database
    return table_name in inspector.get_table_names()

# Define the table schema
def create_commodity_table(engine, table_name):
    metadata = MetaData()
    commodity_table = Table(
        table_name, metadata,
        Column('Date', DateTime),
        *(Column(commodity, Float) for commodity in commodity_tickers.keys()),
        *(Column(f'{commodity}_m/m-12', Float) for commodity in commodity_tickers.keys())
    )
    metadata.create_all(engine)
    print(f"Table '{table_name}' ensured in database.")
    return commodity_table

# Preprocessing and saving to MySQL
def preprocess_and_save_to_mysql(table_name="commodity_data"):
    commodity_data = {}

    # Fetch data for each commodity
    for commodity, ticker in commodity_tickers.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            data = data[['Close']].rename(columns={'Close': commodity})
            data_monthly = data.resample('M').mean()
            data_monthly[commodity] = data_monthly[commodity] * conversion_factors[commodity]
            commodity_data[commodity] = data_monthly
        except Exception as e:
            print(f"Error fetching data for {commodity}: {e}")

    if not commodity_data:
        raise ValueError("No data was fetched.")

    # Combine and preprocess data
    combined_data = pd.concat(commodity_data.values(), axis=1)
    combined_data.index = pd.to_datetime(combined_data.index)  # Convert index to plain date
    combined_data.reset_index(inplace=True)
    combined_data.rename(columns={'index': 'Date'}, inplace=True)

    # Flatten the columns from MultiIndex
    combined_data.columns = [col[0] for col in combined_data.columns]

    # Interpolate missing months
    full_date_range = pd.date_range(start=combined_data["Date"].min(), end=combined_data["Date"].max(), freq='M')
    combined_data.set_index("Date", inplace=True)
    combined_data = combined_data.reindex(full_date_range).reset_index()
    combined_data.rename(columns={"index": "Date"}, inplace=True)
    combined_data.iloc[:, 1:] = combined_data.iloc[:, 1:].interpolate(method="linear", axis=0)

    # Add derived features
    for commodity in commodity_tickers.keys():
        combined_data[f'{commodity}_m/m-12'] = combined_data[commodity].pct_change(periods=12) * 100

    # Handle missing values using KNN Imputer
    knn_imputer = KNNImputer(n_neighbors=5)
    combined_data.iloc[:, 1:] = knn_imputer.fit_transform(combined_data.iloc[:, 1:])
    print(combined_data.head())

    # Save to Aiven MySQL
    try:
        engine = create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?ssl_ca={DB_CONFIG['ssl_ca']}&ssl_disabled={DB_CONFIG['ssl_disabled']}")
        connection = engine.connect()
        print("Connection to MySQL (Aiven) successful!")
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")

    create_commodity_table(engine, table_name)

    try:
        combined_data.to_sql(table_name, con=engine, if_exists="replace", index=False, chunksize=500)
        print(f"Data for '{table_name}' saved to MySQL.")
    except SQLAlchemyError as e:
        print(f"An error occurred while saving data: {e}")

    return combined_data

# Fetch data from MySQL
def fetch_from_mysql(table_name="commodity_data"):
    engine = create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?ssl_ca={DB_CONFIG['ssl_ca']}&ssl_disabled={DB_CONFIG['ssl_disabled']}")
    return pd.read_sql(f"SELECT * FROM {table_name}", con=engine)