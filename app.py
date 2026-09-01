"""PMOS Intelligence Platform — Streamlit front end.

This file is presentation only. Every model call goes through
``src.pipeline.predict_pipeline``, and every piece of recommendation copy comes
from ``src.recommendations``, so the same logic could back a different UI
without changes here.

Run from the repo root:

    venv/bin/streamlit run app.py
"""

import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline.predict_pipeline import PatientData, PredictPipeline
from src.recommendations import (
    DAILY_FOUNDATION,
    PLAN_FOOTER,
    PLAN_INTRO,
    get_recommendations,
    overall_urgency,
)
from src.utils import COLOR_HIGH, COLOR_LOW, COLOR_MODERATE, RISK_HIGH, RISK_MODERATE

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='PMOS Intelligence Platform',
    page_icon='🧬',
    layout='wide',
    initial_sidebar_state='expanded',
)

DISCLAIMER = (
    'This tool is for informational purposes only and does not constitute '
    'medical advice. Always consult a qualified healthcare professional.'
)

RISK_LABELS = {
    'Metabolic_Risk': 'Metabolic',
    'CVD_Risk': 'Cardiovascular',
    'Reproductive_Risk': 'Reproductive',
    'Psych_Risk': 'Psychological',
}

MORPHOLOGY_NOTES = {
    'PCO': 'Multiple small follicles — consistent with polycystic ovarian morphology.',
    'Dominant Follicle': 'Single dominant follicle — associated with ovulatory cycle.',
    'Normal': 'Normal ovarian morphology — no polycystic features identified.',
}


@st.cache_resource
def get_pipeline() -> PredictPipeline:
    return PredictPipeline()


pipeline = get_pipeline()


def gauge_chart(prob, label, color):
    fig, ax = plt.subplots(figsize=(3, 2.2), subplot_kw=dict(polar=False))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    theta = np.linspace(np.pi, 0, 100)
    theta_fill = np.linspace(np.pi, np.pi - prob * np.pi, 100)
    ax.fill_between(
        0.5 + 0.4 * np.cos(theta), 0.1 + 0.4 * np.sin(theta), 0.1,
        alpha=0.1, color='grey',
    )
    ax.fill_between(
        0.5 + 0.4 * np.cos(theta_fill), 0.1 + 0.4 * np.sin(theta_fill), 0.1,
        alpha=0.8, color=color,
    )
    ax.text(0.5, 0.38, f'{prob * 100:.0f}%', ha='center', va='center',
            fontsize=17, fontweight='bold', color=color)
    ax.text(0.5, 0.18, label, ha='center', va='center', fontsize=9, color='grey')
    fig.patch.set_alpha(0)
    plt.tight_layout(pad=0)
    return fig


def shap_chart(shap_df):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#c0392b' if v > 0 else '#2980b9' for v in shap_df['SHAP Value']]
    ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors,
            edgecolor='white', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('SHAP Value  (positive = increases PMOS probability, '
                  'negative = decreases PMOS probability)')
    ax.set_title('Feature Contributions to This Prediction',
                 fontsize=13, fontweight='bold')
    ax.legend(
        handles=[
            mpatches.Patch(color='#c0392b', label='Increases PMOS probability'),
            mpatches.Patch(color='#2980b9', label='Decreases PMOS probability'),
        ],
        loc='lower right',
    )
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


# ── Sidebar ──
st.sidebar.title('PMOS Intelligence')
st.sidebar.markdown('---')
page = st.sidebar.radio(
    'Navigate',
    ['Patient Input', 'Diagnosis Result', 'Risk Dashboard',
     'Recommendations', 'SHAP Explanation'],
)
st.sidebar.markdown('---')
st.sidebar.caption(DISCLAIMER)

for key in ['patient_data', 'pmos_prob', 'risks', 'cnn_result', 'cnn_conf']:
    if key not in st.session_state:
        st.session_state[key] = None


def require_input() -> None:
    """Pages after the first depend on session state from Patient Input."""
    if st.session_state.patient_data is None:
        st.warning('Please complete Patient Input first.')
        st.stop()


# ══ PAGE 1 ══
if page == 'Patient Input':
    st.title('Patient Information')
    st.markdown('Enter your clinical measurements below.')
    st.markdown('---')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Ovarian Morphology')
        follicle_r = st.number_input('Follicle No. (Right Ovary)', 0, 30, 5)
        follicle_l = st.number_input('Follicle No. (Left Ovary)', 0, 30, 5)
        st.subheader('Hormonal Markers')
        lh = st.number_input('LH (mIU/mL)', 0.0, 200.0, 2.5)
        amh = st.number_input('AMH (ng/mL)', 0.0, 50.0, 2.0)
        hcg = st.number_input('I beta-HCG (mIU/mL)', 0.0, 500.0, 2.0)
        prg = st.number_input('PRG (ng/mL)', 0.0, 30.0, 0.3)
    with col2:
        st.subheader('Clinical Symptoms')
        cycle = st.selectbox(
            'Menstrual Cycle', options=[2, 4],
            format_func=lambda x: 'Regular' if x == 2 else 'Irregular',
        )
        weight_gain = st.checkbox('Weight Gain')
        hair_growth = st.checkbox('Excess Hair Growth (Hirsutism)')
        skin_dark = st.checkbox('Skin Darkening')
        pimples = st.checkbox('Pimples / Acne')
        fast_food = st.checkbox('Regular Fast Food Consumption')
    with col3:
        st.subheader('Ultrasound Image (Optional)')
        st.caption('Upload your ovarian ultrasound image for CNN-based '
                   'morphology classification.')
        uploaded = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])
        if uploaded:
            st.image(uploaded, caption='Uploaded Image', width='stretch')

    st.markdown('---')
    if st.button('Analyse', type='primary', width='stretch'):
        patient = PatientData(
            follicle_r=follicle_r,
            follicle_l=follicle_l,
            lh=lh,
            amh=amh,
            hcg=hcg,
            prg=prg,
            cycle=cycle,
            weight_gain=weight_gain,
            hair_growth=hair_growth,
            skin_darkening=skin_dark,
            pimples=pimples,
            fast_food=fast_food,
        ).to_dict()

        with st.spinner('Analysing...'):
            st.session_state.patient_data = patient
            st.session_state.pmos_prob = pipeline.predict_pmos(patient)
            st.session_state.risks = pipeline.predict_risks(patient)
            if uploaded:
                cls, conf = pipeline.predict_ultrasound(uploaded.read())
                st.session_state.cnn_result = cls
                st.session_state.cnn_conf = conf
        st.success('Analysis complete. Navigate to Diagnosis Result.')

# ══ PAGE 2 ══
elif page == 'Diagnosis Result':
    st.title('Diagnosis Result')
    require_input()

    prob = st.session_state.pmos_prob
    color = (COLOR_HIGH if prob > RISK_HIGH
             else COLOR_MODERATE if prob > RISK_MODERATE
             else COLOR_LOW)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader('PMOS Probability')
        st.markdown(
            f'<h1 style="color:{color}; font-size:60px;">{prob * 100:.1f}%</h1>',
            unsafe_allow_html=True,
        )
        if prob > RISK_HIGH:
            st.error('High probability of PMOS detected.')
        elif prob > RISK_MODERATE:
            st.warning('Moderate probability of PMOS detected.')
        else:
            st.success('Low probability of PMOS detected.')
        st.caption('Model-based estimate only. Consult a healthcare professional.')
    with col2:
        st.pyplot(gauge_chart(prob, 'PMOS Risk', color), width='content')

    st.markdown('---')
    if st.session_state.cnn_result:
        st.subheader('Ultrasound Morphology Classification')
        c3, c4 = st.columns(2)
        with c3:
            st.metric('Classification', st.session_state.cnn_result)
            st.metric('Model Confidence', f'{st.session_state.cnn_conf * 100:.1f}%')
        with c4:
            st.info(MORPHOLOGY_NOTES.get(st.session_state.cnn_result, ''))
    else:
        st.info('No ultrasound image uploaded. Upload on the Input page for '
                'CNN classification.')

    st.markdown('---')
    st.subheader('Patient Summary')
    p = st.session_state.patient_data
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Follicle No. R', p['Follicle No. (R)'])
    c2.metric('Follicle No. L', p['Follicle No. (L)'])
    c3.metric('AMH (ng/mL)', p['AMH(ng/mL)'])
    c4.metric('LH (mIU/mL)', p['LH(mIU/mL)'])

# ══ PAGE 3 ══
elif page == 'Risk Dashboard':
    st.title('Risk Dashboard')
    require_input()

    risks = st.session_state.risks
    cols = st.columns(4)
    for col, (target, label) in zip(cols, RISK_LABELS.items()):
        info = risks[target]
        with col:
            st.subheader(label)
            st.pyplot(
                gauge_chart(info['prob'], info['label'], info['color']),
                width='stretch',
            )
            st.markdown(
                f'<p style="text-align:center; color:{info["color"]}; '
                f'font-weight:bold;">{info["label"]}</p>',
                unsafe_allow_html=True,
            )

    st.markdown('---')
    st.subheader('Risk Score Summary')
    st.dataframe(
        pd.DataFrame(
            [
                {
                    'Risk Dimension': label,
                    'Probability': f'{risks[target]["prob"] * 100:.1f}%',
                    'Category': risks[target]['label'],
                }
                for target, label in RISK_LABELS.items()
            ]
        ),
        width='stretch',
        hide_index=True,
    )
    st.caption('Risk scores are model-based estimates. Not clinical diagnoses.')

# ══ PAGE 4 ══
elif page == 'Recommendations':
    st.title('Your Personalised PMOS Management Plan')
    st.info(PLAN_INTRO)
    require_input()

    st.markdown('## Your Daily Foundation')
    st.markdown('These habits support metabolic, hormonal and overall wellbeing '
                'across all risk domains:')
    for item in DAILY_FOUNDATION:
        st.markdown(f'- {item}')
    st.markdown('---')

    headline, headline_color, headline_sub = overall_urgency(st.session_state.risks)
    st.markdown(
        f'<h3 style="color:{headline_color}; margin-bottom:4px;">{headline}</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(headline_sub)
    st.markdown('---')

    sections = [
        ('driver', 'What may be driving this?'),
        ('actions', 'What you can do now'),
    ]
    for rec in get_recommendations(st.session_state.risks):
        color = rec['color']
        with st.container(border=True):
            st.markdown(
                f'<h3 style="color:{color}; margin-bottom:2px;">{rec["title"]}</h3>'
                f'<p style="color:{color}; font-weight:bold; margin-top:0;">'
                f'{rec["label"]} &middot; risk probability {rec["prob"] * 100:.1f}%</p>',
                unsafe_allow_html=True,
            )
            for key, header in sections:
                body = rec.get(key)
                if not body:
                    continue
                st.markdown(f'**{header}**')
                if isinstance(body, list):
                    for item in body:
                        st.markdown(f'- {item}')
                else:
                    st.markdown(body)
            if rec.get('goal'):
                st.markdown(f'**Main goal:** {rec["goal"]}')
        st.markdown('')

    st.divider()
    st.caption(PLAN_FOOTER)

# ══ PAGE 5 ══
elif page == 'SHAP Explanation':
    st.title('Why This Prediction?')
    st.markdown('SHAP values show which features pushed the prediction toward '
                'or away from PMOS.')
    require_input()

    shap_df = pipeline.shap_contributions(st.session_state.patient_data)
    st.pyplot(shap_chart(shap_df), width='stretch')

    st.markdown('---')
    st.subheader('Clinical Interpretation')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**Factors increasing PMOS risk:**')
        top_up = (shap_df[shap_df['SHAP Value'] > 0]
                  .sort_values('SHAP Value', ascending=False).head(3))
        for _, row in top_up.iterrows():
            st.markdown(f'- **{row["Feature"]}** = {row["Patient Val"]:.2f} '
                        f'(SHAP: +{row["SHAP Value"]:.3f})')
    with col2:
        st.markdown('**Factors reducing PMOS risk:**')
        top_down = (shap_df[shap_df['SHAP Value'] < 0]
                    .sort_values('SHAP Value').head(3))
        for _, row in top_down.iterrows():
            st.markdown(f'- **{row["Feature"]}** = {row["Patient Val"]:.2f} '
                        f'(SHAP: {row["SHAP Value"]:.3f})')
    st.caption(DISCLAIMER)
