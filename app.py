from flask import Flask, request, render_template
import numpy as np
import joblib   # better than pickle for sklearn

# ---------------------------------------------------------
# Load model and scalers (trained & saved earlier)
# ---------------------------------------------------------
model = joblib.load("model.joblib")
sc = joblib.load("standscaler.joblib")
mx = joblib.load("minmaxscaler.joblib")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Safely convert all inputs to floats
    try:
        N        = float(request.form['Nitrogen'])
        P        = float(request.form['Phosporus'])
        K        = float(request.form['Potassium'])
        temp     = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph       = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])
    except ValueError:
        return render_template("index.html",
            result="Invalid input: please enter numeric values.")

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred  = np.array([feature_list])

    # Apply the same scaling as training
    mx_features  = mx.transform(single_pred)
    sc_features  = sc.transform(mx_features)

    prediction   = model.predict(sc_features)

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
        7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
        12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
        16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
        20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    crop_name = crop_dict.get(prediction[0],
        "Sorry, we could not determine the best crop.")
    result = f"{crop_name} is the best crop to be cultivated right there."

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
