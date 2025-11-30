"""
Automate Preprocessing - Diabetes Dataset
Nama Siswa: Eidelwise Prily Safana
Kriteria: ADVANCE (4 pts)

File ini melakukan preprocessing data secara otomatis dan mengembalikan
data yang siap digunakan untuk pelatihan model machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import argparse
from datetime import datetime


class DiabetesPreprocessor:
    """
    Kelas untuk melakukan preprocessing dataset diabetes secara otomatis.
    """
    
    def __init__(self, raw_data_path: str):
        """
        Inisialisasi preprocessor dengan path ke data mentah.
        
        Args:
            raw_data_path: Path ke file CSV data mentah
        """
        self.raw_data_path = raw_data_path
        self.scaler = StandardScaler()
        self.zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        self.feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        self.target_col = 'Outcome'
        
    def load_data(self) -> pd.DataFrame:
        """
        Memuat data dari file CSV.
        
        Returns:
            DataFrame berisi data mentah
        """
        print(f"[INFO] Memuat data dari: {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        print(f"[INFO] Data berhasil dimuat. Shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menangani nilai 0 yang seharusnya missing dengan mengganti menggunakan median.
        
        Args:
            df: DataFrame input
            
        Returns:
            DataFrame dengan nilai missing yang sudah ditangani
        """
        print("\n[INFO] Menangani missing values (nilai 0)...")
        df_processed = df.copy()
        
        for col in self.zero_cols:
            zero_count = (df_processed[col] == 0).sum()
            if zero_count > 0:
                # Hitung median dari nilai non-zero
                median_val = df_processed[df_processed[col] != 0][col].median()
                df_processed[col] = df_processed[col].replace(0, median_val)
                print(f"  - {col}: {zero_count} nilai 0 diganti dengan median {median_val:.2f}")
        
        return df_processed
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menghapus data duplikat.
        
        Args:
            df: DataFrame input
            
        Returns:
            DataFrame tanpa duplikat
        """
        initial_count = len(df)
        df_processed = df.drop_duplicates()
        removed_count = initial_count - len(df_processed)
        print(f"\n[INFO] Menghapus duplikat: {removed_count} baris dihapus")
        return df_processed
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Melakukan standarisasi pada fitur numerik.
        
        Args:
            df: DataFrame input
            
        Returns:
            DataFrame dengan fitur yang sudah distandarisasi
        """
        print("\n[INFO] Melakukan standarisasi fitur...")
        df_processed = df.copy()
        
        # Pisahkan fitur dan target
        X = df_processed[self.feature_cols]
        y = df_processed[self.target_col]
        
        # Standarisasi fitur
        X_scaled = self.scaler.fit_transform(X)
        
        # Gabungkan kembali
        df_scaled = pd.DataFrame(X_scaled, columns=self.feature_cols)
        df_scaled[self.target_col] = y.values
        
        print(f"  - Fitur yang distandarisasi: {self.feature_cols}")
        
        return df_scaled
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   random_state: int = 42) -> tuple:
        """
        Membagi data menjadi training dan testing set.
        
        Args:
            df: DataFrame input
            test_size: Proporsi data testing
            random_state: Random seed untuk reprodusibilitas
            
        Returns:
            Tuple (train_df, test_df)
        """
        print(f"\n[INFO] Membagi data (test_size={test_size})...")
        
        X = df[self.feature_cols]
        y = df[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Gabungkan kembali untuk disimpan
        train_df = X_train.copy()
        train_df[self.target_col] = y_train.values
        
        test_df = X_test.copy()
        test_df[self.target_col] = y_test.values
        
        print(f"  - Training set: {len(train_df)} samples")
        print(f"  - Testing set: {len(test_df)} samples")
        
        return train_df, test_df
    
    def preprocess(self, output_dir: str = None) -> pd.DataFrame:
        """
        Menjalankan seluruh pipeline preprocessing.
        
        Args:
            output_dir: Directory untuk menyimpan hasil preprocessing
            
        Returns:
            DataFrame yang sudah dipreprocessing
        """
        print("=" * 60)
        print("AUTOMATE PREPROCESSING - DIABETES DATASET")
        print("Nama Siswa: Eidelwise Prily Safana")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Scale features
        df_processed = self.scale_features(df)
        
        # Split data
        train_df, test_df = self.split_data(df_processed)
        
        # Simpan hasil jika output_dir diberikan
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            train_path = os.path.join(output_dir, 'train_data.csv')
            test_path = os.path.join(output_dir, 'test_data.csv')
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            print(f"\n[INFO] Data training disimpan ke: {train_path}")
            print(f"[INFO] Data testing disimpan ke: {test_path}")
        
        print("\n" + "=" * 60)
        print("PREPROCESSING SELESAI")
        print("=" * 60)
        
        return df_processed


def main():
    """Fungsi utama untuk menjalankan preprocessing dari command line."""
    parser = argparse.ArgumentParser(
        description='Automate Preprocessing untuk Dataset Diabetes'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/raw/diabetes.csv',
        help='Path ke file data mentah'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/preprocessed',
        help='Directory untuk menyimpan hasil preprocessing'
    )
    
    args = parser.parse_args()
    
    # Jalankan preprocessing
    preprocessor = DiabetesPreprocessor(args.input)
    df_processed = preprocessor.preprocess(output_dir=args.output)
    
    # Tampilkan statistik akhir
    print("\n[INFO] Statistik Data Hasil Preprocessing:")
    print(df_processed.describe().T[['mean', 'std', 'min', 'max']])
    
    return df_processed


if __name__ == "__main__":
    main()
