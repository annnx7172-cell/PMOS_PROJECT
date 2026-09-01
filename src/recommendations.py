"""Recommendation content keyed by risk dimension and band.

This is the copy the Recommendations page renders. It is the same content the
Block 8 engine produces; ``artifacts/recommendation_engine.pkl`` is kept for
the notebook's own use and is not read at serving time, so if you change the
wording in one place change it in the other too.

Every ``ACTION_MAP`` entry has the same shape:

``title``    heading for the card
``color``    band colour, matching ``src.utils.risk_band``
``driver``   what in the profile the model responded to, and the mechanism
             behind it where there is one worth stating
``actions``  what the patient can discuss with a doctor or do now, as a
             single ordered list — doctor-facing checks come first
``goal``     one line on what the actions are aiming at

The wording is deliberately non-prescriptive — it describes what people with a
similar profile are commonly advised to discuss, never what the patient
should do. Keep it that way, and keep the disclaimer on any page that renders
it.
"""

from src.utils import COLOR_HIGH, COLOR_LOW, COLOR_MODERATE

PLAN_INTRO = (
    'These recommendations are educational and are designed to help you '
    'understand practical next steps. They do not replace diagnosis or '
    'treatment from a healthcare professional.'
)

# Applies across all four risk dimensions regardless of band — the same
# circadian-rhythm and meal-timing habits support metabolic, hormonal and
# psychological wellbeing at once.
DAILY_FOUNDATION = [
    'Follow your body clock: wake around sunrise when practical, get morning '
    'daylight, and keep your wake-up and sleep times consistent every day.',
    'Structured eating: aim for approximately 3-4 hours between meals and '
    'avoid continuous grazing or frequent snacking throughout the day.',
    'Finish eating early: aim to have dinner by 8 PM at the latest and avoid '
    'late-night eating.',
    'Leave roughly 3 hours between dinner and sleep whenever practical.',
    'Build balanced meals: prioritise protein, vegetables, fibre-rich '
    'carbohydrates and minimally processed foods.',
    'Stay active: aim for around 150-300 minutes of moderate activity per '
    'week, plus strength training as appropriate.',
]

PLAN_FOOTER = (
    'Important: Risk scores are model-generated estimates, not medical '
    'diagnoses. Testing, supplements, medications and treatment decisions '
    'should be discussed with a qualified healthcare professional.'
)

ACTION_MAP = {
    'Metabolic_Risk': {
        'High': {
            'title'  : 'Metabolic Risk — Take Action',
            'color'  : COLOR_HIGH,
            'driver' : 'Insulin resistance can cause the body to produce more insulin than usual. Persistently elevated insulin can contribute to abnormal glucose metabolism, weight-management difficulties and increased androgen activity in PMOS.',
            'actions': ['Discuss with your doctor: fasting glucose, HbA1c and other metabolic testing your clinician considers appropriate',
                        'Improve insulin sensitivity: prioritise protein, vegetables, fibre-rich foods and minimally processed carbohydrates; limit sugary drinks and refined carbohydrates',
                        'Avoid constant snacking: keep meals structured with roughly 3-4 hours between meals',
                        'Eat earlier: aim for dinner by 8 PM and avoid late-night eating',
                        'Protect sleep: maintain a consistent sleep/wake schedule and aim for approximately 3 hours between dinner and bedtime',
                        'Exercise regularly: combine aerobic activity with resistance training',
                        'Inositol: myo-inositol/inositol supplementation may be discussed with your healthcare professional — do not assume a specific 40:1 ratio is medically required, as evidence does not establish one standard formulation'],
            'goal'   : 'Improve insulin sensitivity and reduce long-term metabolic risk.',
        },
        'Moderate': {
            'title'  : 'Metabolic Risk — Focus on Prevention',
            'color'  : COLOR_MODERATE,
            'driver' : 'Some features in your profile are associated with metabolic risk, such as weight changes or mild insulin-resistance markers, but the overall pattern is not yet in the high range. This is the stage where prevention has the most leverage.',
            'actions': ['Ask your doctor whether routine metabolic screening — fasting glucose, HbA1c — is appropriate at your next check-up',
                        'Shift towards whole foods and reduce processed snacks and sugary drinks',
                        'Keep meals reasonably structured, with a consistent overnight fasting window',
                        'Aim for roughly 150 minutes of moderate activity a week, building towards the fuller range as it suits you',
                        'Track weight and symptoms monthly rather than daily, and look at the trend rather than single readings'],
            'goal'   : 'Keep metabolic risk from progressing, using changes you can sustain.',
        },
        'Low': {
            'title'  : 'Metabolic Risk — Maintain',
            'color'  : COLOR_LOW,
            'driver' : 'Your profile does not show the pattern this model associates with elevated metabolic risk.',
            'actions': ['Keep up the eating and activity habits you already have',
                        'Attend routine preventive check-ups, including periodic metabolic screening as your doctor advises'],
            'goal'   : 'Maintain current metabolic health and re-check periodically.',
        },
    },
    'CVD_Risk': {
        'High': {
            'title'  : 'Cardiovascular Risk — Take Action',
            'color'  : COLOR_HIGH,
            'driver' : 'Insulin resistance and associated metabolic abnormalities can contribute to unfavourable cholesterol levels, blood-pressure changes and increased cardiometabolic risk.',
            'actions': ['Monitor: discuss blood pressure and a lipid profile with your doctor',
                        'Choose heart-healthy foods: vegetables, fibre-rich foods, protein, whole foods and unsaturated fats',
                        'Reduce highly processed foods, excess saturated/trans fats and sugary drinks',
                        'Stay active: combine regular walking or aerobic activity with strength training',
                        'Avoid smoking, or seek help to quit if applicable',
                        'Keep a regular rhythm: consistent sleep/wake times and avoiding routine late-night eating can support healthier daily habits',
                        'Avoid continuous grazing: use structured meals rather than eating small amounts throughout the day'],
            'goal'   : 'Reduce the metabolic factors that contribute to cardiovascular risk.',
        },
        'Moderate': {
            'title'  : 'Cardiovascular Risk — Focus on Prevention',
            'color'  : COLOR_MODERATE,
            'driver' : 'A few cardiovascular risk factors show up in your profile, but not the full high-risk pattern.',
            'actions': ['Have blood pressure checked at your next routine visit, and ask about a lipid panel if it has been a while',
                        'Add omega-3 rich foods such as fish, walnuts and flaxseed where suitable',
                        'Keep regular aerobic exercise in your week',
                        'Aim for seven to eight hours of sleep on a consistent schedule'],
            'goal'   : 'Hold cardiovascular risk factors steady while they are still easy to influence.',
        },
        'Low': {
            'title'  : 'Cardiovascular Risk — Maintain',
            'color'  : COLOR_LOW,
            'driver' : 'This model does not find the cardiovascular risk pattern in your profile.',
            'actions': ['Continue heart-healthy eating and regular activity',
                        'Have blood pressure checked at routine visits'],
            'goal'   : 'Maintain cardiovascular health with routine monitoring.',
        },
    },
    'Reproductive_Risk': {
        'High': {
            'title'  : 'Reproductive Risk — Take Action',
            'color'  : COLOR_HIGH,
            'driver' : 'In PMOS, insulin resistance and higher insulin levels can increase androgen activity. This may interfere with normal follicle development and ovulation, contributing to irregular or prolonged menstrual cycles.',
            'actions': ['Track your cycle: record cycle length, bleeding and unusual changes',
                        'Discuss persistent irregularity: speak with a gynaecologist if cycles remain irregular or ovulation is a concern',
                        'Support insulin sensitivity: prioritise balanced meals, fibre and protein while limiting refined carbohydrates and sugary drinks',
                        'Keep meals structured: aim for 3-4 hours between meals and avoid constant snacking',
                        'Support your circadian rhythm: get morning daylight, maintain consistent wake/sleep times and avoid routinely eating late at night',
                        'Spearmint tea may be included as a dietary option, particularly for androgen-related symptoms, but it should be considered supportive rather than a replacement for medical treatment',
                        'Inositol: discuss myo-inositol/inositol supplementation with your clinician if cycle or metabolic concerns are present',
                        'Planning pregnancy? Discuss ovulation and fertility planning early with your gynaecologist'],
            'goal'   : 'Support insulin sensitivity and hormonal balance while protecting regular ovulation and reproductive health.',
        },
        'Moderate': {
            'title'  : 'Reproductive Risk — Focus on Prevention',
            'color'  : COLOR_MODERATE,
            'driver' : 'Your profile shows some features linked to cycle or ovulatory irregularity, without the full high-risk pattern.',
            'actions': ['Track your cycle monthly and note anything unusual',
                        'Speak to a gynaecologist if you are planning pregnancy in the next one to two years, or sooner if cycles are consistently irregular',
                        'Keep meals structured and limit refined carbohydrates, which can support hormonal balance'],
            'goal'   : 'Catch cycle changes early and keep reproductive options open.',
        },
        'Low': {
            'title'  : 'Reproductive Risk — Maintain',
            'color'  : COLOR_LOW,
            'driver' : 'Your cycle and hormonal features do not show the pattern this model links to reproductive risk.',
            'actions': ['Continue routine gynaecological care',
                        'Keep an eye on cycle regularity year to year'],
            'goal'   : 'Maintain reproductive health with routine follow-up.',
        },
    },
    'Psych_Risk': {
        'High': {
            'title'  : 'Psychological Risk — Take Action',
            'color'  : COLOR_HIGH,
            'driver' : 'A high overall symptom burden — such as acne, excess hair growth, weight changes and menstrual irregularity occurring together — is present in your profile. These visible and physical symptoms can meaningfully affect confidence, stress and day-to-day emotional wellbeing for some people.',
            'actions': ['Consider a mental-health screening if low mood, anxiety or distress has been persistent',
                        'Talk to a counsellor or psychologist — a reasonable first step, not a last resort',
                        'Address the physical symptoms with your healthcare provider; relief there often helps mood as well',
                        'Keep sleep, activity and social contact regular, since all three affect mood',
                        'Maintain your daily rhythm: morning daylight and consistent sleep/wake times can help stabilise mood',
                        'Connect with PMOS support groups if shared experience would help'],
            'goal'   : 'Address both the physical symptoms and their emotional impact.',
        },
        'Moderate': {
            'title'  : 'Psychological Risk — Focus on Recovery & Prevention',
            'color'  : COLOR_MODERATE,
            'driver' : 'PMOS symptoms such as acne, hair growth, weight changes and menstrual irregularity can affect confidence, stress levels and emotional wellbeing. Psychological risk is influenced by the overall symptom burden and should not be attributed solely to insulin resistance.',
            'actions': ['Monitor your wellbeing: pay attention to persistent low mood, anxiety, sleep difficulties or loss of interest in normal activities',
                        'Talk to someone: consider a counsellor, psychologist or trusted person if symptoms are affecting daily life',
                        'Stay physically active: regular movement can support mood and wellbeing',
                        'Maintain your daily rhythm: morning daylight and consistent sleep/wake times can help establish a stable routine',
                        'Prioritise sleep: maintain a consistent bedtime and wake-up time',
                        'Keep eating structured: avoid constant snacking and late-night eating',
                        'Stay connected: maintain supportive relationships and seek help when needed'],
            'goal'   : 'Protect mental wellbeing while addressing the physical symptoms that may be contributing to emotional stress.',
        },
        'Low': {
            'title'  : 'Psychological Risk — Maintain',
            'color'  : COLOR_LOW,
            'driver' : 'Your profile does not show the level of symptom burden this model associates with psychological risk.',
            'actions': ['Keep up social connections and regular self-care',
                        'Continue whatever stress management already works for you'],
            'goal'   : 'Maintain wellbeing alongside routine PMOS care.',
        },
    },
}


def get_recommendations(risks: dict) -> list:
    """Turn the risk dict from ``PredictPipeline.predict_risks`` into cards."""
    recs = []
    for target, info in risks.items():
        rec = dict(ACTION_MAP[target][info['label']])
        rec['prob'] = info['prob']
        rec['label'] = info['label']
        recs.append(rec)
    return recs


def overall_urgency(risks: dict):
    """Headline for the Recommendations page: ``(text, colour, subtitle)``.

    Driven by the worst band across the four dimensions, so a single High
    dimension is enough to set the headline.
    """
    labels = [info['label'] for info in risks.values()]
    if 'High' in labels:
        return (
            'High Risk — Take Action',
            COLOR_HIGH,
            'One or more risk areas are in the high range. The steps below are '
            'what people with this kind of profile are commonly advised to '
            'discuss with a doctor.',
        )
    if 'Moderate' in labels:
        return (
            'Moderate Risk — Focus on Prevention',
            COLOR_MODERATE,
            'Nothing is in the high range, but there is room to act. Small '
            'consistent changes have the most effect at this stage.',
        )
    return (
        'Low Risk — Maintain and Monitor',
        COLOR_LOW,
        'No risk area is elevated. Keep up your current habits and routine '
        'check-ups.',
    )
