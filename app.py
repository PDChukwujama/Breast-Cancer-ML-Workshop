
# ─────────────────────────────────────────────────────────────────────────────
# Breast Cancer Risk Indicator — Educational Prototype
# Built during a Computer Society ML Workshop.
# NOT a medical device. Never use this for clinical decisions.
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ── 1. Page configuration ─────────────────────────────────────────────────────
# This must be the very first Streamlit call in the script
st.set_page_config(
    page_title='Breast Cancer Risk Indicator',
    page_icon='🏥',
    layout='centered'
)

# ── 2. Disclaimer banner — shown at the very top of every page ───────────────
st.error(
    '⚠️  EDUCATIONAL PROTOTYPE — NOT MEDICAL ADVICE  |  '
    'This tool was built during a university workshop and has not been '
    'validated for clinical use. Never use it to inform any medical decision.'
)

# ── 3. Title and subtitle ─────────────────────────────────────────────────────
st.title('🏥 Breast Cancer Risk Indicator')
st.caption(
    'Educational demonstration · '
    'Wisconsin Breast Cancer Dataset (UCI, 1993) · '
    'Decision Tree classifier'
)
st.markdown('---')
# ── 4. Train and cache the model ──────────────────────────────────────────────
# @st.cache_resource means this function runs only ONCE when the app first loads
# After that, Streamlit reuses the saved result instead of retraining every time
@st.cache_resource
def build_model():
    # Load the dataset — built into sklearn, no download needed
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target   # 0 = malignant, 1 = benign (sklearn's encoding)

    # Same 80/20 split and random seed used in Sessions 1 and 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train the decision tree model on the training data
    # max_depth=3 keeps the tree shallow to avoid overfitting (as in Session 2)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Calculate test accuracy to display in the sidebar
    test_acc = clf.score(X_test, y_test)

    # Store the training-set mean for each feature
    # We use these as the default values for features not shown as sliders
    train_means = X_train.mean()

    return clf, list(cancer.feature_names), train_means, test_acc

# Call the function — result is cached after the first load
model, feature_names, feature_means, test_accuracy = build_model()

# ── 5. Sidebar — model information and a second disclaimer ────────────────────
with st.sidebar:
    st.header('Model Information')
    st.metric(label='Test Accuracy', value=f'{test_accuracy*100:.1f}%')
    st.metric(label='Training patients', value='455')
    st.metric(label='Test patients', value='114')
    st.metric(label='Total features', value='30 (5 shown)')
    st.markdown('---')
    st.markdown('**What are these measurements?**')
    st.markdown(
        'Values come from digitised images of Fine Needle Aspirate (FNA) '
      )
    st.markdown('---')
    st.warning('⚠️  Not medical advice.')

# ── 6. Slider inputs ──────────────────────────────────────────────────────────
st.subheader('Cell Measurement Inputs')
st.markdown(
    'Adjust the 5 sliders below. The remaining 25 features from the dataset '
    'are held at their average values from the training data. '
    'The risk indicator updates automatically.'
)
st.markdown('')

# Two-column layout so the sliders are not stacked in a single long list
col1, col2 = st.columns(2)

with col1:
    # Worst radius: largest radius measurement in the sample (range: 7.9 – 36.0 mm)
    worst_radius = st.slider(
        'Worst Radius (mm)',
        min_value=7.0, max_value=37.0, value=16.0, step=0.1,
        help='Largest radius measurement from all cells in the biopsy sample'
    )
    # Worst texture: highest texture score (range: 12 – 50)
    worst_texture = st.slider(
        'Worst Texture',
        min_value=12.0, max_value=50.0, value=25.0, step=0.5,
        help='Texture irregularity score — higher means rougher, more irregular cells'
    )
    # Worst symmetry: how asymmetric the worst cell is (range: 0.15 – 0.66)
    worst_symmetry = st.slider(
        'Worst Symmetry',
        min_value=0.15, max_value=0.66, value=0.29, step=0.01,
        help='Higher value means more asymmetric cells — healthy cells tend to be symmetric'
    )

with col2:
    # Worst concave points: concave indentations in the worst cell boundary (range: 0 – 0.29)
    worst_concave_pts = st.slider(
        'Worst Concave Points',
        min_value=0.00, max_value=0.30, value=0.11, step=0.005,
        help='Number of concave indentations in the most irregular cell boundary (0 = smooth)'
    )
    mean_concave_pts = st.slider(
        'Mean Concave Points',
        min_value=0.00, max_value=0.20, value=0.05, step=0.005,
        help='Average number of concave indentations across all cells in the sample'
    )

# ── 7. Build the full 30-feature input vector ─────────────────────────────────
# Start from the training-set average values for all 30 features
input_row = feature_means.copy()

# Override just the 5 features the user has set with the slider
input_row['worst radius']         = worst_radius
input_row['worst texture']        = worst_texture
input_row['worst symmetry']       = worst_symmetry
input_row['worst concave points'] = worst_concave_pts
input_row['mean concave points']  = mean_concave_pts

# Reshape into a 2D array: sklearn's predict expects shape (n_samples, n_features)
input_array = np.array([input_row[feature_names].values])

# ── 8. Get prediction probabilities ───────────────────────────────────────────
# predict_proba returns [probability_of_class_0, probability_of_class_1]
# In this dataset: class 0 = malignant, class 1 = benign
probs          = model.predict_proba(input_array)[0]
prob_malignant = probs[0]    # probability the pattern matches malignant training examples
prob_benign    = probs[1]    # probability the pattern matches benign training examples

# ── 9. Map probability to a plain-language risk band ─────────────────────────
if prob_malignant < 0.25:
    risk_label = '🟢  LOW RISK PATTERN'
    risk_note  = (
        'The entered measurements are more consistent with patterns '
        'seen in benign samples in the training data.'
    )
elif prob_malignant < 0.60:
    risk_label = '🟡  MEDIUM RISK PATTERN'
    risk_note  = (
        'The entered measurements show mixed patterns. '
        'The model is uncertain — values fall between typical benign and malignant ranges.'
    )
else:
    risk_label = '🔴  HIGH RISK PATTERN'
    risk_note  = (
          'The entered measurements are more consistent with patterns '
        'seen in malignant samples in the training data.'
    )

# ── 10. Display the result ────────────────────────────────────────────────────
st.markdown('---')
st.subheader('Risk Indicator')
st.markdown(f'## {risk_label}')           # large coloured label
st.progress(float(prob_malignant))         # visual confidence bar (0 = benign, 1 = malignant)
st.markdown(
    f'Malignant pattern score: **{prob_malignant:.0%}** '
    f' |  '
    f'Benign pattern score: **{prob_benign:.0%}**'
)
st.info(risk_note)

# ── 11. Bottom disclaimer — repeated at the foot of every page ────────────────
st.markdown('---')
st.error(
    '🚫  This indicator does NOT diagnose cancer. '
    'It is a teaching demonstration only. '
    'Real cancer diagnosis requires clinical examination, imaging, biopsy analysis, '
    'and interpretation by qualified medical professionals. '
    'Please consult a doctor if you have any health concerns.'
)
st.caption(
    'Computer Society ML Workshop · Educational prototype · '
    'Not for clinical use · Wisconsin Breast Cancer Dataset'
)
