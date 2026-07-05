import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def run_preprocessing():
    print("Memulai proses otomatisasi preprocessing...")
    
    # 1. Memuat Dataset
    # Pastikan file CStrain.csv di-push juga ke repo GitHub Anda!
    dataset_path = 'CStrain.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: File {dataset_path} tidak ditemukan!")
        return
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset awal: {df.shape}")
    
    # 2. Hapus duplikat
    df_clean = df.drop_duplicates()
    
    # 3. Pilih Fitur
    target_col = 'Credit_Score'
    num_cols = [
        'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Interest_Rate', 'Delay_from_due_date', 'Num_Credit_Inquiries',
        'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Monthly_Balance'
    ]
    cat_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
    
    selected_cols = num_cols + cat_cols + [target_col]
    df_clean = df_clean[selected_cols]
    
    # 4. Pisahkan X dan y
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # 5. Pipeline Preprocessing
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # 6. Fit & Transform
    X_processed = preprocessor.fit_transform(X)
    cat_encoded_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    all_cols = num_cols + list(cat_encoded_cols)
    
    # 7. Simpan Hasil
    df_processed = pd.DataFrame(X_processed, columns=all_cols)
    df_processed[target_col] = y.reset_index(drop=True)
    
    joblib.dump(preprocessor, 'preprocessor.pkl')
    df_processed.to_csv('data_preprocessed.csv', index=False)
    
    print(f"Preprocessing Selesai! Data akhir: {df_processed.shape}")

if __name__ == "__main__":
    run_preprocessing()
