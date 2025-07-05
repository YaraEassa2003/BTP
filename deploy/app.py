from flask import Flask, request, jsonify
import joblib, pandas as pd

# one artefact → the full preprocessing + model pipeline
PIPELINE = joblib.load("car_price_pipeline.pkl")

RAW_COLS = ['odometer', 'make', 'model', 'body', 'transmission', 'condition']

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # --- basic validation -------------------------------------------------
    missing = [c for c in RAW_COLS if c not in data]
    if missing:
        return jsonify({"error": f"Missing keys: {missing}"}), 400

    # --- build 1-row DataFrame in the right order -------------------------
    X = pd.DataFrame([{c: data[c] for c in RAW_COLS}])

    # cast numerics (silently converts bad strings to NaN, then to 0)
    X['odometer']   = pd.to_numeric(X['odometer'],   errors='coerce').fillna(0)
    X['condition']  = pd.to_numeric(X['condition'],  errors='coerce').fillna(0)

    # optional: lower-case text like you did in training
    for col in ['make', 'model', 'body', 'transmission']:
        X[col] = X[col].astype(str).str.lower().str.strip()

    # --- prediction -------------------------------------------------------
    pred = PIPELINE.predict(X)[0]          # returns a 1-element array
    return jsonify({"prediction": float(pred)})

if __name__ == "__main__":
    # when running locally:  http://127.0.0.1:5000/predict
    app.run(host="0.0.0.0", port=8080)
