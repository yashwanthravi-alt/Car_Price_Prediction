import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "car_price_prediction_model.joblib"
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

st.title("🚗 Used Car Price Prediction")
st.write("Enter the vehicle details below to estimate its selling price using a machine learning model trained on the CarDekho dataset.")
try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("The trained model file is missing. First run the training script.")
    st.code("python train_model.py\nstreamlit run app.py", language="bash")
    st.stop()

model = bundle["model"]
features = bundle["features"]
car_options = bundle["categorical_options"]
num_defaults = bundle["numeric_defaults"]
num_bounds = bundle["numeric_bounds"]
st.caption(f"Selected model: {bundle['model_name']}")

cat_labels = {"brand": "Car brand", "fuel": "Fuel type", "seller_type": "Seller type", "transmission": "Transmission", "owner": "Owner history"}
num_labels = {"car_age": "Car age (years)", "km_driven": "Kilometers driven", "mileage_km_ltr_kg": "Mileage (km/l or kg)", "engine": "Engine capacity (cc)", "max_power": "Maximum power (bhp)", "seats": "Number of seats"}

with st.form("car_details"):
    st.subheader("Car details")
    input_values = {}
    cat_cols = [c for c in features if c in cat_options]
    cols = st.columns(2)
    for i, col in enumerate(cat_cols):
        with cols[i % 2]:
            opts = cat_options[col]
            input_values[col] = st.selectbox(cat_labels.get(col, col.replace("_", " ").title()), opts, key=f"cat_{col}")

    st.subheader("Vehicle specifications")
    num_cols = [c for c in features if c in num_defaults]
    cols = st.columns(2)
    for i, col in enumerate(num_cols):
        default = num_defaults[col]
        bounds = num_bounds[col]
        with cols[i % 2]:
            label = num_labels.get(col, col.replace("_", " ").title())
            if col == "car_age":
                max_age = max(40, bundle["current_year"] - 1980)
                input_values[col] = st.number_input(label, min_value=0, max_value=int(max_age), value=int(round(min(max(default, 0), max_age))), step=1)
            elif col == "seats":
                input_values[col] = st.number_input(label, min_value=1, max_value=20, value=int(round(min(max(default, 1), 20))), step=1)
            else:
                low, high = float(bounds["min"]), float(bounds["max"])
                if high <= low:
                    high = low + 1.0
                default = min(max(default, low), high)
                step = 1000.0 if col == "km_driven" else (0.1 if col in ["mileage_km_ltr_kg", "max_power"] else 1.0)
                input_values[col] = st.number_input(label, min_value=low, max_value=high, value=float(default), step=float(step), format="%.2f")

    submitted = st.form_submit_button("Predict Price", type="primary")

if submitted:
    input_df = pd.DataFrame([{c: input_values[c] for c in features}], columns=features)
    prediction = float(model.predict(input_df)[0])
    st.success("Prediction completed")
    st.metric("Estimated selling price", f"₹{prediction:,.0f}")
    st.caption("This is an estimate based on historical data, not a guaranteed market price.")
    with st.expander("Review entered car details"):
        st.dataframe(input_df, use_container_width=True)

with st.expander("Model evaluation"):
    results_df = pd.DataFrame(bundle.get("results", []))
    if not results_df.empty:
        results_df = results_df.rename(columns={"model": "Model", "mae": "MAE", "rmse": "RMSE", "r2": "R² Score"})
        st.dataframe(results_df.round(3), use_container_width=True)
        st.caption("MAE and RMSE use the dataset's price units. Lower errors are better; higher R² is generally better.")

st.divider()
st.caption("Built with Python, Scikit-learn, and Streamlit.")
