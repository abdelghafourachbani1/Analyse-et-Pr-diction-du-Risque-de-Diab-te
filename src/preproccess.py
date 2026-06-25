import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df):
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    
    for col in cols_with_zeros:
        df[col] = df[col].fillna(df[col].median())
        
    return df

def cap_outliers(df):
    features_to_cap = ['Pregnancies', 'Insulin', 'SkinThickness', 'DiabetesPedigreeFunction', 'BMI', 'BloodPressure', 'Glucose']
    for col in features_to_cap:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[col] = np.clip(df[col], lower_bound, upper_bound)
        
    return df

def scale_features(df):
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    
    return df_scaled, scaler

def run_preprocessing_pipeline(file_path):
    print("1. Loading raw data...")
    df = pd.read_csv(file_path, index_col=0)
    
    print("2. Handling missing values...")
    df = handle_missing_values(df)

    print("3. Capping outliers...")
    df = cap_outliers(df)
    
    print("4. Scaling features...")
    df_scaled, scaler = scale_features(df)
    
    print("Pipeline Complete. Data is ready for Machine Learning.")
    return df_scaled, scaler