import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
 
 
def encode_features(df):
    df['Is_Canada'] = (df['Country'] == 'Canada').astype(int)
 
    encode_cols = ['Course Title', 'Province', 'Program', 'Job Family']
    df_encoded = pd.get_dummies(df, columns=encode_cols, drop_first=True)
 
    drop_cols = ['City', 'Country', 'Employer', 'Department',
                 'Job Title', 'Enrollment Status']
    df_model = df_encoded.drop(columns=drop_cols)
 
    return df_model
 
 
def split_data(df_model, test_size=0.2, random_state=42):
    X = df_model.drop(columns=['Enrolled_Binary'])
    y = df_model['Enrolled_Binary']
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
 
 
def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    return lr
 
 
def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf
 
 
def save_model(model, filename, model_dir='../outputs/models'):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)
    joblib.dump(model, path)
    print(f'Saved: {path}')