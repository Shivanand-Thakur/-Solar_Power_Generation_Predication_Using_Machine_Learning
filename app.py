import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load trained model and preprocessing
model = joblib.load("solar_model.pkl")
imputer = joblib.load("imputer.pkl")


# Feature columns 

FEATURE_COLUMNS = [
    "distance_to_solar_noon",
    "temperature",
    "wind_direction",
    "wind_speed",
    "sky_cover",
    "visibility",
    "humidity",
    "average_wind_speed",
    "average_pressure",
    "solar_exposure_index",
    "wind_energy_index",
    "air_density_proxy"
]


# Page configuration

st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="wide"
)


# Header

st.title("☀️ Solar Power Generation Prediction")
st.caption(
    "Gradient Boosting–based ML system for estimating solar energy production."
)

st.divider()


# Sidebar: Inputs

st.sidebar.header("🔧 Environmental Inputs")

distance_to_solar_noon = st.sidebar.slider(
    "Distance to Solar Noon (radians)", 0.0, 3.14, 0.5, 0.01
)
temperature = st.sidebar.slider(
    "Temperature (°C)", -10.0, 50.0, 25.0
)
wind_direction = st.sidebar.slider(
    "Wind Direction (degrees)", 0, 360, 180
)
wind_speed = st.sidebar.slider(
    "Wind Speed (m/s)", 0.0, 20.0, 3.0
)
sky_cover = st.sidebar.selectbox(
    "Sky Cover (0–4)",
    [0, 1, 2, 3, 4],
    format_func=lambda x: f"{x} – {'Clear' if x==0 else 'Cloudy' if x>=3 else 'Partial'}"
)
visibility = st.sidebar.slider(
    "Visibility (km)", 0.0, 20.0, 10.0
)
humidity = st.sidebar.slider(
    "Humidity (%)", 0, 100, 60
)
average_wind_speed = st.sidebar.slider(
    "Average Wind Speed (m/s)", 0.0, 20.0, 3.0
)
average_pressure = st.sidebar.slider(
    "Average Pressure (inHg)", 28.0, 31.0, 29.9
)

st.sidebar.divider()


# Sidebar: Analysis Controls

st.sidebar.header("📊 Analysis Options")

show_feature_importance = st.sidebar.checkbox(
    "Show Feature Importance", value=True
)

show_sensitivity = st.sidebar.checkbox(
    "Show Sensitivity Analysis", value=True
)

sensitivity_feature = st.sidebar.selectbox(
    "Sensitivity Feature",
    ["temperature", "wind_speed", "wind_direction"],
    disabled=not show_sensitivity
)


# Feature Engineering

solar_exposure_index = np.cos(distance_to_solar_noon) * (4 - sky_cover)
wind_energy_index = wind_speed * average_wind_speed
air_density_proxy = average_pressure / (temperature + 273.15)

input_df = pd.DataFrame([[
    distance_to_solar_noon,
    temperature,
    wind_direction,
    wind_speed,
    sky_cover,
    visibility,
    humidity,
    average_wind_speed,
    average_pressure,
    solar_exposure_index,
    wind_energy_index,
    air_density_proxy
]], columns=FEATURE_COLUMNS)


# Prediction Button

st.subheader("📈 Prediction")

predict_clicked = st.button("🚀 Predict Solar Power", use_container_width=True)

if predict_clicked:
    input_imputed = imputer.transform(input_df)
    prediction = model.predict(input_imputed)

    st.success(
        f"🔋 **Estimated Power Generated:** `{prediction[0]:,.2f} Joules`"
    )

    
    # Feature Importance
    
    if show_feature_importance:
        st.divider()
        st.subheader("📌 Feature Importance")

        fi_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(fi_df["Feature"], fi_df["Importance"])
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance (Gradient Boosting)")
        st.pyplot(fig)

    
    # Sensitivity Analysis
    
    if show_sensitivity:
        st.divider()
        st.subheader(f"🧪 Sensitivity Analysis: {sensitivity_feature}")

        ranges = {
            "temperature": np.linspace(-10, 50, 60),
            "wind_speed": np.linspace(0, 20, 60),
            "wind_direction": np.linspace(0, 360, 72)
        }

        predictions = []

        for val in ranges[sensitivity_feature]:
            vals = {
                "distance_to_solar_noon": distance_to_solar_noon,
                "temperature": temperature,
                "wind_direction": wind_direction,
                "wind_speed": wind_speed,
                "sky_cover": sky_cover,
                "visibility": visibility,
                "humidity": humidity,
                "average_wind_speed": average_wind_speed,
                "average_pressure": average_pressure
            }

            vals[sensitivity_feature] = val

            se = np.cos(vals["distance_to_solar_noon"]) * (4 - vals["sky_cover"])
            we = vals["wind_speed"] * vals["average_wind_speed"]
            ad = vals["average_pressure"] / (vals["temperature"] + 273.15)

            row_df = pd.DataFrame([[
                vals["distance_to_solar_noon"],
                vals["temperature"],
                vals["wind_direction"],
                vals["wind_speed"],
                vals["sky_cover"],
                vals["visibility"],
                vals["humidity"],
                vals["average_wind_speed"],
                vals["average_pressure"],
                se, we, ad
            ]], columns=FEATURE_COLUMNS)

            row_imp = imputer.transform(row_df)
            predictions.append(model.predict(row_imp)[0])

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(ranges[sensitivity_feature], predictions, linewidth=2)
        ax2.set_xlabel(sensitivity_feature)
        ax2.set_ylabel("Predicted Power Generated")
        ax2.set_title(f"Sensitivity of Power vs {sensitivity_feature}")
        st.pyplot(fig2)

else:
    st.info("👈 Adjust inputs and click **Predict Solar Power** to view results.")


# Footer
st.caption("© Solar Power Prediction")
