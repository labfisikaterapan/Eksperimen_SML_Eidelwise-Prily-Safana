# Workflow Information

## GitHub Actions Workflow

### simple_preprocessing.yml

Workflow ini otomatis menjalankan preprocessing pipeline setiap kali ada push ke branch `main`.

### Workflow Steps:
1. **Checkout repository** - Mengambil kode dari repository
2. **Setup Python 3.10** - Menyiapkan environment Python
3. **Install dependencies** - Menginstall library yang diperlukan
4. **Run preprocessing** - Menjalankan script preprocessing
5. **Upload artifacts** - Menyimpan hasil preprocessing

### Trigger:
- Push ke branch `main` atau `master`
- Pull request ke branch `main` atau `master`
- Manual dispatch

### Artifacts:
- `preprocessed-data`: Berisi train_data.csv dan test_data.csv

## Cara Menjalankan Manual:

```bash
cd preprocessing
python automate_Eidelwise_Prily_Safana.py
```

## Output:
- `data/preprocessed/train_data.csv`
- `data/preprocessed/test_data.csv`
- `data/preprocessed/scaler.pkl`
