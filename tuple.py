
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="PCOS Risk Prediction System",
    page_icon="🩺",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🩺 AI Based PCOS Risk Prediction System")
st.subheader("Machine Learning Based PCOS Risk Prediction")

st.info(
    "This application is developed for academic project purposes. "
    "It is not a medical diagnosis."
)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

file_name = "PCOS_Risk_Prediction_Dataset_1000-24.csv"

try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    st.error(
        "Dataset file not found. Please keep the CSV file "
        "in the same folder as app.py."
    )
    st.stop()


# ---------------------------------------------------------
# DATASET COLUMNS
# ---------------------------------------------------------

features = [
    "Age",
    "BMI",
    "Cycle_Length_Days",
    "Irregular_Periods",
    "Weight_Gain",
    "Acne",
    "Excess_Hair_Growth",
    "Hair_Loss",
    "Skin_Darkening",
    "Pelvic_Pain",
    "Exercise_Days_Per_Week",
    "Family_History_PCOS"
]

target = "PCOS_Risk"


# ---------------------------------------------------------
# CHECK COLUMNS
# ---------------------------------------------------------

missing = [column for column in features + [target]
           if column not in df.columns]

if missing:
    st.error("These columns are missing from the dataset:")
    st.write(missing)
    st.stop()


# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------

df = df.dropna()

for column in features:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna()


# ---------------------------------------------------------
# INPUT AND TARGET
# ---------------------------------------------------------

X = df[features]

y = df[target].astype(str).str.strip()


# ---------------------------------------------------------
# ENCODE TARGET
# ---------------------------------------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)


# ---------------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# ---------------------------------------------------------
# RANDOM FOREST MODEL
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ---------------------------------------------------------
# MODEL ACCURACY
# ---------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🤖 Model Information")

st.sidebar.write(
    "Algorithm: Random Forest Classifier"
)

st.sidebar.write(
    "Dataset Records:",
    len(df)
)

st.sidebar.write(
    "Features:",
    len(features)
)

st.sidebar.write(
    "Accuracy:",
    f"{accuracy * 100:.2f}%"
)


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

st.header("👩 Enter Your Details")


col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=60,
        value=21
    )


with col2:

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=22.0,
        step=0.1
    )


with col3:

    cycle_length = st.number_input(
        "Cycle Length (Days)",
        min_value=15,
        max_value=60,
        value=28
    )


# ---------------------------------------------------------
# SYMPTOMS
# ---------------------------------------------------------

st.subheader("Symptoms")


col1, col2, col3 = st.columns(3)


with col1:

    irregular_periods = st.selectbox(
        "Irregular Periods",
        ["No", "Yes"]
    )

    weight_gain = st.selectbox(
        "Weight Gain",
        ["No", "Yes"]
    )

    acne = st.selectbox(
        "Acne",
        ["No", "Yes"]
    )


with col2:

    excess_hair = st.selectbox(
        "Excess Hair Growth",
        ["No", "Yes"]
    )

    hair_loss = st.selectbox(
        "Hair Loss",
        ["No", "Yes"]
    )

    skin_darkening = st.selectbox(
        "Skin Darkening",
        ["No", "Yes"]
    )


with col3:

    pelvic_pain = st.selectbox(
        "Pelvic Pain",
        ["No", "Yes"]
    )

    family_history = st.selectbox(
        "Family History of PCOS",
        ["No", "Yes"]
    )

    exercise_days = st.slider(
        "Exercise Days Per Week",
        0,
        7,
        3
    )


# ---------------------------------------------------------
# CONVERT YES / NO TO 0 / 1
# ---------------------------------------------------------

irregular_periods = 1 if irregular_periods == "Yes" else 0
weight_gain = 1 if weight_gain == "Yes" else 0
acne = 1 if acne == "Yes" else 0
excess_hair = 1 if excess_hair == "Yes" else 0
hair_loss = 1 if hair_loss == "Yes" else 0
skin_darkening = 1 if skin_darkening == "Yes" else 0
pelvic_pain = 1 if pelvic_pain == "Yes" else 0
family_history = 1 if family_history == "Yes" else 0


# ---------------------------------------------------------
# CREATE USER DATA
# ---------------------------------------------------------

user_data = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "Cycle_Length_Days": [cycle_length],
    "Irregular_Periods": [irregular_periods],
    "Weight_Gain": [weight_gain],
    "Acne": [acne],
    "Excess_Hair_Growth": [excess_hair],
    "Hair_Loss": [hair_loss],
    "Skin_Darkening": [skin_darkening],
    "Pelvic_Pain": [pelvic_pain],
    "Exercise_Days_Per_Week": [exercise_days],
    "Family_History_PCOS": [family_history]
})


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

st.divider()

if st.button(
    "🔍 Predict PCOS Risk",
    use_container_width=True
):

    prediction = model.predict(user_data)[0]

    prediction_label = encoder.inverse_transform(
        [prediction]
    )[0]

    probabilities = model.predict_proba(
        user_data
    )[0]


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.header("📋 Prediction Result")

    if prediction_label == "High":

        st.error("⚠️ HIGH PCOS RISK")

    elif prediction_label == "Medium":

        st.warning("⚠️ MEDIUM PCOS RISK")

    else:

        st.success("✅ LOW PCOS RISK")


    st.subheader(
        f"Predicted Risk Level: {prediction_label}"
    )


    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    st.subheader("📊 Prediction Probability")


    probability_df = pd.DataFrame({
        "Risk Level": encoder.classes_,
        "Probability (%)": probabilities * 100
    })


    probability_df["Probability (%)"] = (
        probability_df["Probability (%)"].round(2)
    )


    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )


    st.bar_chart(
        probability_df.set_index("Risk Level")
    )


# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------

st.divider()

st.header("🤖 Model Performance")

st.metric(
    "Random Forest Accuracy",
    f"{accuracy * 100:.2f}%"
)


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

st.subheader("📈 Feature Importance")


importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})


importance = importance.sort_values(
    "Importance",
    ascending=False
)


st.bar_chart(
    importance.set_index("Feature")
)


# ---------------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------------

with st.expander("📂 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "AI Based PCOS Risk Prediction System | "
    "Random Forest Classifier"
)

st.caption(
    "Academic Project | For educational purposes only"
)

