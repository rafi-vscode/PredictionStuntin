import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel("PredictionStudent/dataset.csv")

# Pilih fitur yang tersedia dalam dataset
feature_cols = ["BB", "TB", "Pekerjaan", "Pendidikan Terakhir", "Aktivitas Fisik", "IMT", "Status Gizi", "Konsumsi Obat Tambah Darah", "Konsumsi Suplemen Tambahan", "Konsumsi Nasi per Sajian", "Riwayat Anemia", "LILA", "KEK", "Konsumsi Tinggi Protein", "Konsumsi Buah dan Sayur", "Kebiasaan Makan"]
target_col = "Risiko Stunting"

# Pisahkan fitur dan target
X = df[feature_cols]
y = df[target_col]

# Bagi data menjadi training dan testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tangani missing values dengan imputasi (mean)
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inisialisasi model dengan hyperparameter yang lebih optimal
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=500)
tree = DecisionTreeClassifier(random_state=42)

# Training model
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)
tree.fit(X_train, y_train)

# Simpan model ke dalam file pickle
models = {'RandomForest': rf, 'LogisticRegression': lr, 'DecisionTree': tree}
with open("model_rf_smote.pkl", "wb") as file:
    pickle.dump(models, file)
