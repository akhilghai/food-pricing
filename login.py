import streamlit as st
import mysql.connector
from mysql.connector import Error
from streamlit_option_menu import option_menu
import bcrypt
import secrets
import datetime
import os
from dotenv import load_dotenv

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

import streamlit_app


# Load environment variables
load_dotenv()

# Aiven MySQL Connection Details
DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT")),
    "ssl_ca": os.getenv("SSL_CA_PATH"),
    "ssl_disabled": False
}

st.set_page_config(page_title="Food Price Forecasting", layout="wide", initial_sidebar_state="expanded")


# MySQL Database Connection
def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

# Utility Functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_reset_token():
    return secrets.token_urlsafe(16)

# App Components
def login():
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
    
    if submit:
        try:
            connection = get_connection()
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            
            if user and check_password(password, user["password_hash"]):
                st.session_state["user"] = user["username"]
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        except Error as e:
            st.error(f"Database error: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

def signup():
    st.title("Sign Up")
    with st.form("signup_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign Up")
    
    if submit:
        try:
            connection = get_connection()
            cursor = connection.cursor()
            hashed_password = hash_password(password)
            default_role = "user"
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                (username, email, hashed_password, default_role)
            )
            connection.commit()
            st.success("Account created successfully!")
        except Error as e:
            st.error(f"Error: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

def forgot_password():
    st.title("Forgot Password")
    with st.form("forgot_form"):
        email = st.text_input("Enter your registered email")
        submit = st.form_submit_button("Request Reset")
    
    if submit:
        try:
            connection = get_connection()
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()
            if user:
                reset_token = generate_reset_token()
                expiry = datetime.datetime.now() + datetime.timedelta(hours=1)
                cursor.execute(
                    "UPDATE users SET reset_token = %s, token_expiry = %s WHERE email = %s",
                    (reset_token, expiry, email)
                )
                connection.commit()
                st.info(f"Reset token generated: {reset_token}")  # Replace with email functionality
            else:
                st.error("No account associated with this email.")
        except Error as e:
            st.error(f"Error: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

def reset_password():
    st.title("Reset Password")
    with st.form("reset_form"):
        reset_token = st.text_input("Reset Token")
        new_password = st.text_input("New Password", type="password")
        submit = st.form_submit_button("Reset Password")
    
    if submit:
        try:
            connection = get_connection()
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM users WHERE reset_token = %s AND token_expiry > NOW()",
                (reset_token,)
            )
            user = cursor.fetchone()
            if user:
                hashed_password = hash_password(new_password)
                cursor.execute(
                    "UPDATE users SET password_hash = %s, reset_token = NULL, token_expiry = NULL WHERE id = %s",
                    (hashed_password, user["id"])
                )
                connection.commit()
                st.success("Password reset successful!")
            else:
                st.error("Invalid or expired token.")
        except Error as e:
            st.error(f"Error: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Main Application
def main():
    if "user" not in st.session_state:
        st.session_state["user"] = None
    
    if st.session_state["user"]:
        streamlit_app.main()
        # st.sidebar.success(f"Welcome {st.session_state['user']}!")
        # if st.sidebar.button("Logout"):
        #     st.session_state["user"] = None
        #     st.rerun()
    else:
        with st.sidebar:
            choice = option_menu(
                "Menu",
                ["Login", "Sign Up", "Forgot Password", "Reset Password"],
                icons=["box-arrow-in-right", "person-plus", "key", "shield-lock"],
                menu_icon="menu",
                default_index=0
            )

        if choice == "Login":
            login()
        elif choice == "Sign Up":
            signup()
        elif choice == "Forgot Password":
            forgot_password()
        elif choice == "Reset Password":
            reset_password()

if __name__ == "__main__":
    # Inject custom CSS
    st.markdown(
        """
        <style>
        /* Make the sidebar fixed and prevent collapsing */
        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 250px;
            width: 250px;
            transition: none !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] {
            min-width: 250px;
            width: 250px;
            transition: none !important;
        }
        button[kind="expandMinimized"] {
            display: none !important;
        }
        button[kind="expand"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    main()
