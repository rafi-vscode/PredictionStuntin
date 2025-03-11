from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pickle
import os

# Inisialisasi aplikasi Flask
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Cek apakah model tersedia
model_path = "model_rf_smote.pkl"
rf_model = None

if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as model_file:
            rf_model = pickle.load(model_file)
        print("✅ Model berhasil dimuat.")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
else:
    print("❌ Model tidak ditemukan!")

# Mapping label prediksi
label_mapping = {
    0: "Sehat (Tidak Berisiko / Ideal)",
    1: "Ringan (Ada sedikit risiko KEK)",
    2: "KEK (Beresiko Mengalami KEK, Butuh Intervensi Gizi)",
    3: "Berat (Kondisi KEK Parah)"
}

reverse_label_mapping = {v.upper(): k for k, v in label_mapping.items()}

@app.route("/")
def home():
    if "nama" in session:
        return redirect(url_for("prediction"))
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    session["nama"] = request.form.get("nama")
    session["nomor_hp"] = request.form.get("nomor_hp")
    return redirect(url_for("prediction"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route('/data')
def data():
    if "nama" not in session:
        return redirect(url_for("home"))
    return render_template("data.html", prediction_data=session.get("prediction_data"))


@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')

@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        try:
            # Ambil input dari form
            umur = request.form.get("umur", type=float)
            bb = request.form.get("bb", type=float)
            tb = request.form.get("tb", type=float)
            imt = request.form.get("imt", type=float)
            lila = request.form.get("lila", type=float)
            hb = request.form.get("hb", type=float)

            # Pastikan semua input sudah diisi
            if None in [umur, bb, tb, imt, lila, hb]:
                flash("❌ Semua input harus diisi dengan angka yang valid.", "danger")
                return redirect(url_for("prediction"))

            # Pastikan model tersedia
            if rf_model is None:
                flash("❌ Model tidak tersedia untuk prediksi.", "danger")
                return redirect(url_for("prediction"))

            # Konversi input ke numpy array
            features = np.array([[umur, bb, tb, imt, lila, hb]])

            # Lakukan prediksi
            if hasattr(rf_model, "predict"):
                prediction_raw = rf_model.predict(features)[0]

                # Pastikan output valid
                if isinstance(prediction_raw, str):
                    prediction_rf = reverse_label_mapping.get(prediction_raw.upper(), None)
                elif isinstance(prediction_raw, (int, float)):
                    prediction_rf = int(prediction_raw)
                else:
                    prediction_rf = None

                # Ambil label berdasarkan hasil prediksi
                if prediction_rf in label_mapping:
                    prediction_label = label_mapping.get(prediction_rf,)
                else:
                    prediction_label = "⚠️ Kategori Tidak Diketahui"
                    flash(f"❌ Hasil prediksi tidak valid: {prediction_raw}", "danger") 

                # Simpan hasil prediksi ke session
                session["prediction_data"] = {
                    "nama": session["nama"],
                    "nomor_hp": session["nomor_hp"],
                    "umur": umur,
                    "bb": bb,
                    "tb": tb,
                    "imt": imt,
                    "lila": lila,
                    "hb": hb,
                    "prediction_rf": prediction_rf,
                    "prediction_label": prediction_label,
                }

                # Redirect ke halaman hasil prediksi
                return redirect(url_for("data"))

            else:
                flash("❌ Model tidak memiliki metode predict(). Pastikan file model benar.", "danger")

        except Exception as e:
            print("❌ Terjadi kesalahan pada prediksi:", str(e))
            flash(f"❌ Error: {e}", "danger")

    return render_template("prediction.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
