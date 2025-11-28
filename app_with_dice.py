import streamlit as st
import numpy as np
import pandas as pd
import joblib
import dice_ml

st.set_page_config(page_title="CL Slag Cu Classification", layout="wide")

# -------------------------
# Load model, scaler & training data / metadata
# -------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("cu_rf_model.pkl")
    scaler = joblib.load("cu_scaler.pkl")
    X_train = np.load("X_train.npy")      # unscaled training features (numpy array)
    y_train = np.load("y_train.npy")      # training labels (numpy array)
    feature_order = joblib.load("feature_order.pkl")  # list of feature names
    return model, scaler, X_train, y_train, feature_order

try:
    model, scaler, X_train, y_train, FEATURE_ORDER = load_artifacts()
except Exception as e:
    st.error("Error loading artifacts. Make sure cu_rf_model.pkl, cu_scaler.pkl, X_train.npy, y_train.npy, feature_order.pkl exist.")
    st.stop()

# -------------------------
# Build DiCE objects (sklearn backend)
# -------------------------
@st.cache_resource
def build_dice_objects(X_train, y_train, feature_order):
    # Build dataframe expected by DiCE
    df_train = pd.DataFrame(X_train, columns=feature_order)
    df_train["target"] = y_train

    # continuous_features = list of all feature names (assuming all are continuous)
    continuous_features = feature_order.copy()

    data_dice = dice_ml.Data(dataframe=df_train, continuous_features=continuous_features, outcome_name="target")
    model_dice = dice_ml.Model(model=model, backend="sklearn")
    # method "random" works robustly for tree models; "genetic" also possible
    dice_exp = dice_ml.Dice(data_dice, model_dice, method="genetic")
    return dice_exp, df_train

dice_exp, df_train = build_dice_objects(X_train, y_train, FEATURE_ORDER)

st.title("CL Slag Cu Classification")
st.markdown("Enter process inputs:")

# -------------------------
# UI: grouped inputs (you can rearrange UI freely)
# -------------------------
st.markdown("### Input parameters")

# Create columns/groups for appearance (example groups)
with st.expander("Blend Composition", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        Fe = st.number_input("Fe (%)", value=float(X_train[:, FEATURE_ORDER.index('Fe')].mean()))
        SiO2 = st.number_input("SiO₂ (%)", value=float(X_train[:, FEATURE_ORDER.index('SiO2')].mean()))
    with col2:
        CaO = st.number_input("CaO (%)", value=float(X_train[:, FEATURE_ORDER.index('CaO')].mean()))
        MgO = st.number_input("MgO (%)", value=float(X_train[:, FEATURE_ORDER.index('MgO')].mean()))
    with col3:
        Al2O3 = st.number_input("Al₂O₃ (%)", value=float(X_train[:, FEATURE_ORDER.index('Al2O3')].mean()))
        S_Cu = st.number_input("S/Cu Ratio", value=float(X_train[:, FEATURE_ORDER.index('S/Cu')].mean()))

with st.expander("S-Furnace Parameters", expanded=False):
    col4, col5 = st.columns(2)
    with col4:
        conc_feed = st.number_input("CONC. FEED RATE", value=float(X_train[:, FEATURE_ORDER.index('CONC. FEED RATE')].mean()))
        silica_feed = st.number_input("SILICA FEED RATE", value=float(X_train[:, FEATURE_ORDER.index('SILICA FEED RATE ')].mean()))
        cslag_feed = st.number_input("C-SLAG FEED RATE - S Furnace", value=float(X_train[:, FEATURE_ORDER.index('C-SLAG FEED RATE - S Furnace')].mean()))
    with col5:
        s_air = st.number_input("S-FURNACE AIR", value=float(X_train[:, FEATURE_ORDER.index('S-FURNACE AIR')].mean()))
        s_oxygen = st.number_input("S-FURNACE OXYGEN", value=float(X_train[:, FEATURE_ORDER.index('S-FURNACE OXYGEN')].mean()))

with st.expander("Fe/SiO2 & CLS", expanded=False):
    fe_sio2_ratio = st.number_input("Fe/SiO2", value=float(X_train[:, FEATURE_ORDER.index('Fe/SiO2')].mean()))
    fe3o4_cls = st.number_input("Fe3O4_Cls", value=float(X_train[:, FEATURE_ORDER.index('Fe3O4_Cls')].mean()))

with st.expander("Matte Grade", expanded=False):
    matte_grade = st.number_input("Matte Grade", value=float(X_train[:, FEATURE_ORDER.index('Matte Grade')].mean()))

with st.expander("C-Slag Analysis", expanded=False):
    col6, col7, col8, col9 = st.columns(4)
    with col6:
        cu_cslag = st.number_input("Cu_C_slag", value=float(X_train[:, FEATURE_ORDER.index('Cu_C_slag')].mean()))
    with col7:
        fe_cslag = st.number_input("Fe_C_slag", value=float(X_train[:, FEATURE_ORDER.index('Fe_C_slag')].mean()))
    with col8:
        cao_cslag = st.number_input("CaO_C_slag", value=float(X_train[:, FEATURE_ORDER.index('CaO_C_slag')].mean()))
    with col9:
        fe3o4_cslag = st.number_input("Fe3O4_C_slag", value=float(X_train[:, FEATURE_ORDER.index('Fe3O4_C_slag')].mean()))

# Assemble the inputs into the same order used for training
input_map = {
    'Fe': Fe, 'SiO2': SiO2, 'Al2O3': Al2O3, 'CaO': CaO, 'MgO': MgO, 'S/Cu': S_Cu,
    'CONC. FEED RATE': conc_feed, 'SILICA FEED RATE ': silica_feed, 'C-SLAG FEED RATE - S Furnace': cslag_feed,
    'S-FURNACE AIR': s_air, 'S-FURNACE OXYGEN': s_oxygen,
    'Fe/SiO2': fe_sio2_ratio, 'Fe3O4_Cls': fe3o4_cls, 'Matte Grade': matte_grade,
    'Cu_C_slag': cu_cslag, 'Fe_C_slag': fe_cslag, 'CaO_C_slag': cao_cslag, 'Fe3O4_C_slag': fe3o4_cslag
}

# Build feature vector in training order (FEATURE_ORDER)
try:
    input_vector = np.array([[ input_map[f] for f in FEATURE_ORDER ]], dtype=float)
except KeyError as ke:
    st.error(f"Feature {ke} not found in input_map — check FEATURE_ORDER and input_map keys.")
    st.stop()

# Show prediction on current input
st.markdown("---")
st.subheader(" Predict Cl Slag Cu Class")
scaled_in = scaler.transform(input_vector)
pred_class = model.predict(scaled_in)[0]
pred_proba = model.predict_proba(scaled_in)[0]

# ---- Modified display (your requested style) ----
if pred_class == 1:
    st.success(f"Predicted Class:[0.70–0.75 Cu%] — Probability: {pred_proba[1]:.2f}")
else:
    st.error(f"Predicted Class: [0.80–0.85 Cu%] — Probability: {pred_proba[0]:.2f}")

# -------------------------
# Counterfactual UI controls
# -------------------------
st.markdown("---")
st.subheader("Counterfactuals for the desired class")

# Multi-select to choose which features to lock (user can pick any number)
locked_features = st.multiselect(
    "Select features to vary:",
    options=FEATURE_ORDER,
    default=[]
)

# Allow user to choose desired target class (or keep same as current)
target_choice = st.radio("Target for counterfactuals:", options=["Same as current prediction", "[0.70–0.75 Cu%]", "[0.80–0.85 Cu%]"])
if target_choice == "Same as current prediction":
    desired_class = int(pred_class)
elif target_choice == "[0.70–0.75 Cu%]":
    desired_class = 1
else:
    desired_class = 0

total_cfs = st.slider("Number of counterfactuals to generate", min_value=1, max_value=5, value=2)
# Optional: permitted ranges input - left as None (DiCE will use training ranges)
use_permitted = st.checkbox("Restrict suggested values to training ranges", value=True)
permitted_range = None
if use_permitted:
    # Build permitted range from observed min/max in X_train
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    permitted_range = { FEATURE_ORDER[i]: [float(min_vals[i]), float(max_vals[i])] for i in range(len(FEATURE_ORDER)) }





# Generate button
if st.button("Generate counterfactuals"):
    with st.spinner("Generating counterfactuals..."):
        query_instance = pd.DataFrame(input_vector, columns=FEATURE_ORDER)

        # features_to_vary = all features except those locked
        features_to_vary = [f for f in FEATURE_ORDER if f not in locked_features]

        try:
            cf_obj = dice_exp.generate_counterfactuals(
                query_instances=query_instance,
                total_CFs=total_cfs,
                desired_class=desired_class,
                features_to_vary=features_to_vary,
                permitted_range=permitted_range,
                proximity_weight=1.5,      
                diversity_weight=1.0  
            )
        except Exception as e:
            st.error(f"DiCE generation error: {e}")
            st.stop()

        # Extract counterfactual dataframe(s)
        try:
            cf_df = cf_obj.cf_examples_list[0].final_cfs_df.reset_index(drop=True)
        except Exception as e:
            st.error(f"Unable to extract CF dataframe: {e}")
            st.stop()

        # Compute deltas (difference from original)
        original_series = query_instance.iloc[0]
        delta_df = cf_df.copy()
        for col in FEATURE_ORDER:
            delta_df[col] = cf_df[col] - original_series[col]

        # Show results side-by-side
        st.success("Counterfactuals generated")
        # Show ONLY changed parameters
        st.write("### Counterfactuals — Parameters Changed")
        changed = delta_df.loc[:, (delta_df != 0).any(axis=0)]
        st.dataframe(changed.style.format("{:.4f}"))

        # Full summary table
        st.write("### Summary table (original / counterfactuals / deltas):")
        combined = pd.DataFrame({"feature": FEATURE_ORDER, "original": original_series.values})
        for i in range(cf_df.shape[0]):
            combined[f"cf_{i+1}"] = cf_df.iloc[i].values
            combined[f"delta_{i+1}"] = cf_df.iloc[i].values - original_series.values
        st.dataframe(combined.style.format({c:"{:.4f}" for c in combined.columns if c!="feature"}))

        # Probabilities for CFs
        cf_scaled = scaler.transform(cf_df[FEATURE_ORDER].values)
        probs = model.predict_proba(cf_scaled)
        probs_df = pd.DataFrame(probs, columns=[f"prob_class_{i}" for i in range(probs.shape[1])])
        st.write("### Predicted probabilities for counterfactuals:")
        st.dataframe(probs_df.style.format("{:.4f}"))
