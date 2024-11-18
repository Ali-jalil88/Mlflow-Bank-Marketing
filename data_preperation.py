from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import os



def load_and_split_data():
    file_path = r"C:\Users\HP\Desktop\Sache\JedenTag\Mahmoud\Session95_13.11.2024_Jypter_Mahmoud\Mlflow-Bank-Marketing\src\bank.csv"

    test_size=0.2
    random_state=42

    # Check if file exists
    if not os.path.exists(file_path):
        print("File not found!")
        return None, None, None, None

    # Load dataset
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")

    # Check column names
    print("Available columns:", df.columns)

    # Remove whitespace from column names (if needed)
    df.columns = df.columns.str.strip()

    # Check for the correct target column
    if 'deposit' not in df.columns:
        print("'deposit' column not found in the dataset.")
        return None, None, None, None

    # Split features and target
    x = df.drop(columns=['deposit'])
    y = df['deposit']
    print("Features and target split successfully.")

     # Encode categorical features
    label_encoder = LabelEncoder()
    for column in x.select_dtypes(include=['object']).columns:
        x[column] = label_encoder.fit_transform(x[column])

    # Encode target variable
    y = y.map({'yes': 1, 'no': 0})

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test
