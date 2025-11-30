# Eksperimen_SML_Eidelwise-Prily-Safana

## Nama Siswa: Eidelwise Prily Safana

## Deskripsi
Repository ini berisi eksperimen preprocessing data untuk dataset Diabetes menggunakan Machine Learning.

## Dataset
- **Nama Dataset:** Pima Indians Diabetes Database
- **Jumlah Data:** 768 baris
- **Jumlah Fitur:** 8 fitur + 1 target
- **Target:** Outcome (0 = No Diabetes, 1 = Diabetes)

### Fitur Dataset:
| No | Fitur | Deskripsi |
|----|-------|-----------|
| 1 | Pregnancies | Jumlah kehamilan |
| 2 | Glucose | Konsentrasi glukosa plasma |
| 3 | BloodPressure | Tekanan darah diastolik (mm Hg) |
| 4 | SkinThickness | Ketebalan kulit triceps (mm) |
| 5 | Insulin | Insulin serum 2 jam (mu U/ml) |
| 6 | BMI | Body mass index |
| 7 | DiabetesPedigreeFunction | Fungsi silsilah diabetes |
| 8 | Age | Usia (tahun) |
| 9 | Outcome | Target (0/1) |

## Struktur Folder
```
Eksperimen_SML_Eidelwise-Prily-Safana/
├── .github/
│   └── workflows/
│       └── simple_preprocessing.yml
├── preprocessing/
│   ├── automate_Eidelwise_Prily_Safana.py
│   └── data/
│       ├── raw/
│       │   └── diabetes.csv
│       └── preprocessed/
│           ├── train_data.csv
│           └── test_data.csv
└── README.md
```

## Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/labfisikaterapan/Eksperimen_SML_Eidelwise-Prily-Safana.git
cd Eksperimen_SML_Eidelwise-Prily-Safana
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn
```

### 3. Jalankan Preprocessing
```bash
cd preprocessing
python automate_Eidelwise_Prily_Safana.py --input data/raw/diabetes.csv --output data/preprocessed
```

## Preprocessing Pipeline
1. **Load Data** - Memuat data dari file CSV
2. **Handle Missing Values** - Mengganti nilai 0 dengan median (untuk kolom Glucose, BloodPressure, SkinThickness, Insulin, BMI)
3. **Remove Duplicates** - Menghapus data duplikat
4. **Feature Scaling** - Standarisasi fitur menggunakan StandardScaler
5. **Train-Test Split** - Membagi data menjadi 80% training dan 20% testing

## GitHub Actions
Workflow akan otomatis berjalan ketika ada push ke branch main atau perubahan pada folder preprocessing.

## Output
- `train_data.csv` - Data training (80%)
- `test_data.csv` - Data testing (20%)

---
**Nama:** Eidelwise Prily Safana
