import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PMOS Intelligence Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

ARTIFACT_DIR  = 'notebook/artifacts'
HIDE_FROM_SHAP = ['Marraige Status (Yrs)']

@st.cache_resource
def load_models():
    with open(f'{ARTIFACT_DIR}/best_model.pkl', 'rb') as f:
        pmos_model = pickle.load(f)
    with open(f'{ARTIFACT_DIR}/final_features.pkl', 'rb') as f:
        final_features = pickle.load(f)
    with open(f'{ARTIFACT_DIR}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{ARTIFACT_DIR}/risk_features.pkl', 'rb') as f:
        risk_features = pickle.load(f)
    with open(f'{ARTIFACT_DIR}/repro_features.pkl', 'rb') as f:
        repro_features = pickle.load(f)
    risk_models = {}
    for target in ['Metabolic_Risk', 'CVD_Risk', 'Reproductive_Risk', 'Psych_Risk']:
        with open(f'{ARTIFACT_DIR}/risk_model_{target}.pkl', 'rb') as f:
            risk_models[target] = pickle.load(f)
    with open(f'{ARTIFACT_DIR}/xgb_shap_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open(f'{ARTIFACT_DIR}/xgb_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    return (pmos_model, final_features, scaler, risk_features,
            repro_features, risk_models, xgb_model, explainer)

(pmos_model, final_features, scaler, risk_features,
 repro_features, risk_models, xgb_model, explainer) = load_models()

@st.cache_resource
def load_cnn():
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(f'{ARTIFACT_DIR}/pmos_cnn_final.keras')
    except Exception:
        return None

cnn_model   = load_cnn()
CNN_CLASSES = ['Dominant Follicle', 'Normal', 'PCO']

def predict_pmos(features_dict):
    X = pd.DataFrame([features_dict])[final_features].fillna(0)
    return pmos_model.predict_proba(X)[0][1]

def predict_risks(features_dict):
    results = {}
    feature_sets = {
        'Metabolic_Risk': risk_features, 'CVD_Risk': risk_features,
        'Reproductive_Risk': repro_features, 'Psych_Risk': risk_features,
    }
    for target, feats in feature_sets.items():
        X     = pd.DataFrame([features_dict])[feats].fillna(0)
        prob  = risk_models[target].predict_proba(X)[0][1]
        label = 'High' if prob > 0.66 else 'Moderate' if prob > 0.33 else 'Low'
        results[target] = {'prob': prob, 'label': label}
    return results

def get_shap_values(features_dict):
    X  = pd.DataFrame([features_dict])[final_features].fillna(0)
    sv = explainer.shap_values(X)
    return sv[0], X.iloc[0]

def predict_cnn(image_bytes):
    if cnn_model is None:
        return None, None
    try:
        import tensorflow as tf
        from PIL import Image
        import io
        img  = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
        arr  = np.expand_dims(np.array(img) / 255.0, 0)
        prob = cnn_model.predict(arr, verbose=0)[0]
        idx  = np.argmax(prob)
        return CNN_CLASSES[idx], float(prob[idx])
    except Exception:
        return None, None

def get_recommendations(risks):
    action_map = {
        'Metabolic_Risk': {
            'High':     {'title': 'Metabolic Risk — Take Action',        'color': '#c0392b',
                         'actions': ['Book a consultation — discuss metabolic risk with an endocrinologist or physician',
                                     'Ask about fasting glucose, HbA1c and insulin resistance tests',
                                     'Diet — prioritise protein, vegetables, whole grains; limit sugary drinks and refined carbohydrates',
                                     'Activity — aim for 150 minutes per week of moderate physical activity',
                                     'Weight — discuss a sustainable weight-management plan with your doctor']},
            'Moderate': {'title': 'Metabolic Risk — Focus on Prevention', 'color': '#d68910',
                         'actions': ['Schedule annual metabolic screening with your doctor',
                                     'Increase whole foods, reduce processed snacks and sugary drinks',
                                     '30 minutes of walking daily is a great starting point',
                                     'Monitor weight monthly, aim to maintain a healthy range']},
            'Low':      {'title': 'Metabolic Risk — Maintain',            'color': '#1e8449',
                         'actions': ['Keep up your current healthy habits',
                                     'Annual metabolic check as routine preventive care']},
        },
        'CVD_Risk': {
            'High':     {'title': 'Cardiovascular Risk — Take Action',        'color': '#c0392b',
                         'actions': ['Book a consultation — discuss cardiovascular risk with a physician',
                                     'Ask about blood pressure and lipid profile (LDL, HDL, triglycerides)',
                                     'Diet — reduce saturated and trans fats and highly processed foods',
                                     'Maintain regular aerobic and strength activity as appropriate',
                                     'Avoid or quit smoking if applicable',
                                     'Prioritise sleep and stress management']},
            'Moderate': {'title': 'Cardiovascular Risk — Focus on Prevention', 'color': '#d68910',
                         'actions': ['Annual lipid panel and blood pressure check',
                                     'Increase omega-3 rich foods — fish, walnuts, flaxseeds',
                                     'Regular aerobic exercise for heart health',
                                     'Aim for 7-8 hours of quality sleep']},
            'Low':      {'title': 'Cardiovascular Risk — Maintain',            'color': '#1e8449',
                         'actions': ['Continue heart-healthy diet and activity habits',
                                     'Routine annual blood pressure check']},
        },
        'Reproductive_Risk': {
            'High':     {'title': 'Reproductive Risk — Take Action',        'color': '#c0392b',
                         'actions': ['Book a consultation — speak with a gynaecologist or reproductive specialist',
                                     'Ask about AMH levels, ovulation assessment and cycle regulation',
                                     'Track your menstrual cycle using an app or basal body temperature',
                                     'Lifestyle — stress reduction and healthy weight support hormonal balance',
                                     'Family planning — if pregnancy is a goal, discuss timing and options early']},
            'Moderate': {'title': 'Reproductive Risk — Focus on Prevention', 'color': '#d68910',
                         'actions': ['Track your cycle monthly, note any irregularities',
                                     'Annual AMH test to monitor ovarian reserve',
                                     'Consult gynaecologist if planning pregnancy in the next 1-2 years']},
            'Low':      {'title': 'Reproductive Risk — Maintain',            'color': '#1e8449',
                         'actions': ['Continue routine gynaecological care',
                                     'Annual cycle tracking and checkup']},
        },
        'Psych_Risk': {
            'High':     {'title': 'Psychological Risk — Take Action',        'color': '#c0392b',
                         'actions': ['Talk to someone — consider speaking with a counsellor or trusted person',
                                     'Ask about mental health screening; your doctor can guide next steps',
                                     'Connect with PMOS support groups — shared experiences help',
                                     'Mindfulness — breathing exercises, meditation or gentle yoga',
                                     'Regular physical activity is one of the strongest mood regulators']},
            'Moderate': {'title': 'Psychological Risk — Focus on Prevention', 'color': '#d68910',
                         'actions': ['Mood tracking — journaling or a mood app can highlight patterns',
                                     'Regular movement for natural mood regulation',
                                     'Talk openly with your doctor about how PMOS symptoms affect you']},
            'Low':      {'title': 'Psychological Risk — Maintain',            'color': '#1e8449',
                         'actions': ['Maintain social connections and regular self-care',
                                     'Continue stress management practices that work for you']},
        },
    }
    recs = []
    for target, info in risks.items():
        label = info['label']
        rec   = action_map[target][label]
        recs.append({'title': rec['title'], 'color': rec['color'],
                     'actions': rec['actions'], 'prob': info['prob'], 'label': label})
    return recs

def gauge_chart(prob, label, color):
    fig, ax = plt.subplots(figsize=(3, 2.2), subplot_kw=dict(polar=False))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    theta      = np.linspace(np.pi, 0, 100)
    theta_fill = np.linspace(np.pi, np.pi - prob * np.pi, 100)
    ax.fill_between(0.5 + 0.4*np.cos(theta),      0.1 + 0.4*np.sin(theta),      0.1, alpha=0.1, color='grey')
    ax.fill_between(0.5 + 0.4*np.cos(theta_fill), 0.1 + 0.4*np.sin(theta_fill), 0.1, alpha=0.8, color=color)
    ax.text(0.5, 0.38, f'{prob*100:.0f}%', ha='center', va='center', fontsize=17, fontweight='bold', color=color)
    ax.text(0.5, 0.18, label,              ha='center', va='center', fontsize=9,  color='grey')
    fig.patch.set_alpha(0); plt.tight_layout(pad=0)
    return fig

# ── Sidebar ──
st.sidebar.title('PMOS Intelligence')
st.sidebar.markdown('---')
page = st.sidebar.radio('Navigate', ['Patient Input', 'Diagnosis Result',
                                      'Risk Dashboard', 'Recommendations',
                                      'SHAP Explanation'])
st.sidebar.markdown('---')
st.sidebar.caption('This tool is for informational purposes only and does not '
                   'constitute medical advice. Always consult a qualified healthcare professional.')

for key in ['patient_data', 'pmos_prob', 'risks', 'cnn_result', 'cnn_conf']:
    if key not in st.session_state:
        st.session_state[key] = None

# ══ PAGE 1 ══
if page == 'Patient Input':
    st.title('Patient Information')
    st.markdown('Enter your clinical measurements below.')
    st.markdown('---')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Ovarian Morphology')
        follicle_r = st.number_input('Follicle No. (Right Ovary)', 0, 30, 5)
        follicle_l = st.number_input('Follicle No. (Left Ovary)',  0, 30, 5)
        st.subheader('Hormonal Markers')
        lh  = st.number_input('LH (mIU/mL)',         0.0, 200.0, 2.5)
        amh = st.number_input('AMH (ng/mL)',          0.0,  50.0, 2.0)
        hcg = st.number_input('I beta-HCG (mIU/mL)', 0.0, 500.0, 2.0)
        prg = st.number_input('PRG (ng/mL)',          0.0,  30.0, 0.3)
    with col2:
        st.subheader('Clinical Symptoms')
        cycle       = st.selectbox('Menstrual Cycle', options=[2, 4],
                                   format_func=lambda x: 'Regular' if x == 2 else 'Irregular')
        weight_gain = st.checkbox('Weight Gain')
        hair_growth = st.checkbox('Excess Hair Growth (Hirsutism)')
        skin_dark   = st.checkbox('Skin Darkening')
        pimples     = st.checkbox('Pimples / Acne')
        fast_food   = st.checkbox('Regular Fast Food Consumption')
    with col3:
        st.subheader('Ultrasound Image (Optional)')
        st.caption('Upload your ovarian ultrasound image for CNN-based morphology classification.')
        uploaded = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])
        if uploaded:
            st.image(uploaded, caption='Uploaded Image', use_column_width=True)
    st.markdown('---')
    if st.button('Analyse', type='primary', use_container_width=True):
        patient = {
            'Follicle No. (R)': follicle_r, 'Follicle No. (L)': follicle_l,
            'LH(mIU/mL)': lh, 'AMH(ng/mL)': amh,
            'I   beta-HCG(mIU/mL)': hcg, 'PRG(ng/mL)': prg,
            'Cycle(R/I)': cycle, 'Weight gain(Y/N)': int(weight_gain),
            'hair growth(Y/N)': int(hair_growth), 'Skin darkening (Y/N)': int(skin_dark),
            'Pimples(Y/N)': int(pimples), 'Hair loss(Y/N)': 0,
            'Fast food (Y/N)': int(fast_food), 'Marraige Status (Yrs)': 5,
        }
        with st.spinner('Analysing...'):
            st.session_state.patient_data = patient
            st.session_state.pmos_prob    = predict_pmos(patient)
            st.session_state.risks        = predict_risks(patient)
            if uploaded and cnn_model is not None:
                cls, conf = predict_cnn(uploaded.read())
                st.session_state.cnn_result = cls
                st.session_state.cnn_conf   = conf
        st.success('Analysis complete. Navigate to Diagnosis Result.')

# ══ PAGE 2 ══
elif page == 'Diagnosis Result':
    st.title('Diagnosis Result')
    if st.session_state.pmos_prob is None:
        st.warning('Please complete Patient Input first.'); st.stop()
    prob  = st.session_state.pmos_prob
    color = '#c0392b' if prob > 0.66 else '#d68910' if prob > 0.33 else '#1e8449'
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader('PMOS Probability')
        st.markdown(f'<h1 style="color:{color}; font-size:60px;">{prob*100:.1f}%</h1>', unsafe_allow_html=True)
        if prob > 0.66:   st.error('High probability of PMOS detected.')
        elif prob > 0.33: st.warning('Moderate probability of PMOS detected.')
        else:             st.success('Low probability of PMOS detected.')
        st.caption('Model-based estimate only. Consult a healthcare professional.')
    with col2:
        st.pyplot(gauge_chart(prob, 'PMOS Risk', color), use_container_width=False)
    st.markdown('---')
    if st.session_state.cnn_result:
        st.subheader('Ultrasound Morphology Classification')
        c3, c4 = st.columns(2)
        with c3:
            st.metric('Classification', st.session_state.cnn_result)
            st.metric('Model Confidence', f'{st.session_state.cnn_conf*100:.1f}%')
        with c4:
            desc = {'PCO': 'Multiple small follicles — consistent with polycystic ovarian morphology.',
                    'Dominant Follicle': 'Single dominant follicle — associated with ovulatory cycle.',
                    'Normal': 'Normal ovarian morphology — no polycystic features identified.'}
            st.info(desc.get(st.session_state.cnn_result, ''))
    else:
        st.info('No ultrasound image uploaded. Upload on the Input page for CNN classification.')
    st.markdown('---')
    st.subheader('Patient Summary')
    p = st.session_state.patient_data
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Follicle No. R', p['Follicle No. (R)'])
    c2.metric('Follicle No. L', p['Follicle No. (L)'])
    c3.metric('AMH (ng/mL)',    p['AMH(ng/mL)'])
    c4.metric('LH (mIU/mL)',   p['LH(mIU/mL)'])

# ══ PAGE 3 ══
elif page == 'Risk Dashboard':
    st.title('Risk Dashboard')
    if st.session_state.risks is None:
        st.warning('Please complete Patient Input first.'); st.stop()
    risks = st.session_state.risks
    risk_labels = {'Metabolic_Risk': 'Metabolic', 'CVD_Risk': 'Cardiovascular',
                   'Reproductive_Risk': 'Reproductive', 'Psych_Risk': 'Psychological'}
    cols = st.columns(4)
    for i, (target, label) in enumerate(risk_labels.items()):
        info   = risks[target]
        prob   = info['prob']
        rlabel = info['label']
        color  = '#c0392b' if rlabel == 'High' else '#d68910' if rlabel == 'Moderate' else '#1e8449'
        with cols[i]:
            st.subheader(label)
            st.pyplot(gauge_chart(prob, rlabel, color), use_container_width=True)
            st.markdown(f'<p style="text-align:center; color:{color}; font-weight:bold;">{rlabel}</p>',
                        unsafe_allow_html=True)
    st.markdown('---')
    st.subheader('Risk Score Summary')
    st.dataframe(pd.DataFrame([{'Risk Dimension': risk_labels[t],
                                 'Probability': f'{risks[t]["prob"]*100:.1f}%',
                                 'Category': risks[t]['label']} for t in risk_labels]),
                 use_container_width=True, hide_index=True)
    st.caption('Risk scores are model-based estimates. Not clinical diagnoses.')

# ══ PAGE 4 ══
elif page == 'Recommendations':
    st.title('Personalised Recommendations')
    st.caption('Based on your risk profile, here are practical steps commonly recommended for people with similar risk patterns.')
    st.markdown('---')
    if st.session_state.risks is None:
        st.warning('Please complete Patient Input first.'); st.stop()
    for rec in get_recommendations(st.session_state.risks):
        color = rec['color']
        st.markdown(f'<h3 style="color:{color}; margin-bottom:4px;">{rec["title"]}</h3>',
                    unsafe_allow_html=True)
        st.markdown(f'*Risk probability: {rec["prob"]*100:.1f}%*')
        for action in rec['actions']:
            st.markdown(f'- {action}')
        st.markdown('---')
    st.caption('These recommendations do not replace professional medical advice. Always consult a qualified healthcare provider.')

# ══ PAGE 5 ══
elif page == 'SHAP Explanation':
    st.title('Why This Prediction?')
    st.markdown('SHAP values show which features pushed the prediction toward or away from PMOS.')
    if st.session_state.patient_data is None:
        st.warning('Please complete Patient Input first.'); st.stop()
    sv, patient_vals = get_shap_values(st.session_state.patient_data)
    shap_df = pd.DataFrame({'Feature': final_features, 'SHAP Value': sv, 'Patient Val': patient_vals.values})
    shap_df = shap_df[~shap_df['Feature'].isin(HIDE_FROM_SHAP)].sort_values('SHAP Value', key=abs, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = ['#c0392b' if v > 0 else '#2980b9' for v in shap_df['SHAP Value']]
    ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('SHAP Value  (positive = increases PMOS probability, negative = decreases PMOS probability)')
    ax.set_title('Feature Contributions to This Prediction', fontsize=13, fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color='#c0392b', label='Increases PMOS probability'),
                        mpatches.Patch(color='#2980b9', label='Decreases PMOS probability')], loc='lower right')
    ax.grid(axis='x', alpha=0.3); plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.markdown('---')
    st.subheader('Clinical Interpretation')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**Factors increasing PMOS risk:**')
        for _, row in shap_df[shap_df['SHAP Value'] > 0].sort_values('SHAP Value', ascending=False).head(3).iterrows():
            st.markdown(f'- **{row["Feature"]}** = {row["Patient Val"]:.2f} (SHAP: +{row["SHAP Value"]:.3f})')
    with col2:
        st.markdown('**Factors reducing PMOS risk:**')
        for _, row in shap_df[shap_df['SHAP Value'] < 0].sort_values('SHAP Value').head(3).iterrows():
            st.markdown(f'- **{row["Feature"]}** = {row["Patient Val"]:.2f} (SHAP: {row["SHAP Value"]:.3f})')
    st.caption('SHAP values explain the XGBoost model. Discuss results with a healthcare professional.')