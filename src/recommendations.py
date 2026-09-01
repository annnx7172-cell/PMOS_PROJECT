"""Recommendation content keyed by risk dimension and band.

This is the copy the Recommendations page renders. It is the same content the
Block 8 engine produces; ``artifacts/recommendation_engine.pkl`` is kept for
the notebook's own use and is not read at serving time, so if you change the
wording in one place change it in the other too.

Every entry has the same shape:

``title``    heading for the card
``color``    band colour, matching ``src.utils.risk_band``
``driver``   what in the profile the model responded to
``pathway``  the mechanism, where there is one worth stating
``checks``   things to raise with a doctor
``actions``  what the patient can do now
``goal``     one line on what the actions are aiming at

The wording is deliberately non-prescriptive — it describes what people with a
similar profile are commonly advised to discuss, never what the patient should
do. Keep it that way, and keep the disclaimer on any page that renders it.
"""

from src.utils import COLOR_HIGH, COLOR_LOW, COLOR_MODERATE

ACTION_MAP = {
    'Metabolic_Risk': {
        'High': {
            'title'  : 'Metabolic Risk — Take Action',
            'color'  : COLOR_HIGH,
            'driver' : 'This model responds to patterns such as weight gain, menstrual irregularity and hormonal markers. In PMOS, insulin resistance is often an underlying contributor to metabolic problems.',
            'pathway': 'Insulin resistance means the cells of the body respond less effectively to insulin. The pancreas may compensate by producing more insulin, and that pattern can make weight management and glucose regulation harder over time.',
            'checks' : ['HbA1c and fasting glucose — ask your physician whether these are appropriate for you',
                        'Whether any further assessment of insulin resistance is warranted',
                        'Weight, blood pressure and metabolic markers, monitored as your doctor advises'],
            'actions': ['Build meals around protein, vegetables and high-fibre carbohydrates',
                        'Reduce sugary drinks and highly refined carbohydrates',
                        'Aim for around 150 minutes per week of moderate activity, if appropriate for you',
                        'Include resistance training two to three times a week if medically suitable',
                        'If weight management is relevant, discuss a sustainable plan with your doctor rather than a crash diet'],
            'goal'   : 'Improve insulin sensitivity and reduce long-term metabolic risk.',
        },
        'Moderate': {
            'title'  : 'Metabolic Risk — Focus on Prevention',
            'color'  : COLOR_MODERATE,
            'driver' : 'Some features in your profile are associated with metabolic risk, but the overall pattern is not in the high range. This is the stage where prevention has the most leverage.',
            'pathway': '',
            'checks' : ['Annual metabolic screening — fasting glucose and HbA1c as your doctor advises',
                        'Weight and blood pressure at routine check-ups'],
            'actions': ['Shift towards whole foods and reduce processed snacks and sugary drinks',
                        'Thirty minutes of brisk walking most days is a solid starting point',
                        'Track weight monthly rather than daily, and look at the trend'],
            'goal'   : 'Keep metabolic risk from progressing, using changes you can sustain.',
        },
        'Low': {
            'title'  : 'Metabolic Risk — Maintain',
            'color'  : COLOR_LOW,
            'driver' : 'Your profile does not show the pattern this model associates with elevated metabolic risk.',
            'pathway': '',
            'checks' : [],
            'actions': ['Keep up the eating and activity habits you already have',
                        'Attend routine preventive check-ups'],
            'goal'   : 'Maintain current metabolic health and re-check periodically.',
        },
    },
    'CVD_Risk': {
        'High': {
            'title'  : 'Cardiovascular Risk — Take Action',
            'color'  : COLOR_HIGH,
            'driver' : 'Cardiovascular risk in PMOS tends to travel with metabolic features such as weight gain and hormonal imbalance, and your profile shows that pattern.',
            'pathway': 'A possible pathway is PMOS leading to insulin resistance and metabolic changes, which over time can affect blood lipids and blood pressure, and in turn raise cardiovascular risk.',
            'checks' : ['Blood pressure',
                        'Lipid profile — LDL, HDL and triglycerides',
                        'Glucose and metabolic markers, as clinically appropriate'],
            'actions': ['Combine regular aerobic activity with strength work, as your fitness allows',
                        'Reduce saturated and trans fats and highly processed foods',
                        'Avoid smoking, or ask your doctor for help stopping',
                        'Protect sleep and manage stress — both affect blood pressure',
                        'If weight management is relevant, address it alongside the rest rather than on its own'],
            'goal'   : 'Act on the specific cardiovascular risk factors that can be changed, rather than following general healthy-eating advice.',
        },
        'Moderate': {
            'title'  : 'Cardiovascular Risk — Focus on Prevention',
            'color'  : COLOR_MODERATE,
            'driver' : 'A few cardiovascular risk factors show up in your profile, but not the full high-risk pattern.',
            'pathway': '',
            'checks' : ['Annual blood pressure check',
                        'Lipid panel at your next routine visit'],
            'actions': ['Add omega-3 rich foods such as fish, walnuts and flaxseed',
                        'Keep regular aerobic exercise in your week',
                        'Aim for seven to eight hours of sleep'],
            'goal'   : 'Hold cardiovascular risk factors steady while they are still easy to influence.',
        },
        'Low': {
            'title'  : 'Cardiovascular Risk — Maintain',
            'color'  : COLOR_LOW,
            'driver' : 'This model does not find the cardiovascular risk pattern in your profile.',
            'pathway': '',
            'checks' : [],
            'actions': ['Continue heart-healthy eating and regular activity',
                        'Have blood pressure checked at routine visits'],
            'goal'   : 'Maintain cardiovascular health with routine monitoring.',
        },
    },
    'Reproductive_Risk': {
        'High': {
            'title'  : 'Reproductive Risk — Take Action',
            'color'  : COLOR_HIGH,
            'driver' : 'Features associated with ovulatory difficulty — such as irregular cycles and markers linked to follicle development — are present in your profile.',
            'pathway': 'A possible pathway is hormonal imbalance disrupting follicular development and ovulation. That shows up as irregular cycles and can make conception harder.',
            'checks' : ['Cycle pattern — length, regularity and any missed periods',
                        'Ovulation status',
                        'Relevant hormones including AMH, as advised by a gynaecologist'],
            'actions': ['Track your cycles — an app or a simple calendar is enough to show the pattern',
                        'Raise irregular periods or suspected missed ovulation with your doctor rather than waiting',
                        'If pregnancy is a goal, discuss timing and options early rather than after months of trying',
                        'Stress management and a stable weight both support hormonal balance'],
            'goal'   : 'Understand and manage ovulatory and cycle dysfunction.',
        },
        'Moderate': {
            'title'  : 'Reproductive Risk — Focus on Prevention',
            'color'  : COLOR_MODERATE,
            'driver' : 'Your profile shows some features linked to cycle or ovulatory irregularity, without the full high-risk pattern.',
            'pathway': '',
            'checks' : ['Cycle regularity over the next few months',
                        'AMH and ovarian reserve, if family planning is on the horizon'],
            'actions': ['Track your cycle monthly and note anything unusual',
                        'Speak to a gynaecologist if you are planning pregnancy in the next one to two years'],
            'goal'   : 'Catch cycle changes early and keep reproductive options open.',
        },
        'Low': {
            'title'  : 'Reproductive Risk — Maintain',
            'color'  : COLOR_LOW,
            'driver' : 'Your cycle and hormonal features do not show the pattern this model links to reproductive risk.',
            'pathway': '',
            'checks' : [],
            'actions': ['Continue routine gynaecological care',
                        'Keep an eye on cycle regularity year to year'],
            'goal'   : 'Maintain reproductive health with routine follow-up.',
        },
    },
    'Psych_Risk': {
        'High': {
            'title'  : 'Psychological Risk — Take Action',
            'color'  : COLOR_HIGH,
            'driver' : 'Symptoms such as acne, excess hair growth, weight concerns or menstrual difficulty may affect emotional wellbeing in some people. Several of these appear together in your profile.',
            'pathway': 'This model counts visible and physical symptoms; it does not measure mood and cannot establish cause. What it flags is a symptom burden of the kind many people find hard to live with.',
            'checks' : ['Mental-health screening, if low mood, anxiety or distress has been persistent',
                        'Whether the physical symptoms themselves can be treated more effectively'],
            'actions': ['Talk to a counsellor or psychologist — a reasonable first step, not a last resort',
                        'Address the physical symptoms with your healthcare provider; relief there often helps',
                        'Keep sleep, activity and social contact regular, since all three affect mood',
                        'Consider mindfulness, breathing practice or gentle movement',
                        'Connect with PMOS support groups if shared experience would help'],
            'goal'   : 'Address both the physical symptoms and their emotional impact.',
        },
        'Moderate': {
            'title'  : 'Psychological Risk — Focus on Prevention',
            'color'  : COLOR_MODERATE,
            'driver' : 'A moderate symptom burden shows in your profile. Some people find symptoms like these affect how they feel day to day.',
            'pathway': '',
            'checks' : ['Whether persistent low mood or anxiety is present and worth screening for'],
            'actions': ['Track mood alongside symptoms — a journal or app can reveal patterns',
                        'Keep regular movement in your week; it is one of the more reliable mood supports',
                        'Tell your doctor how the PMOS symptoms are affecting you, not only what they are'],
            'goal'   : 'Notice early if symptoms begin affecting wellbeing, and act then.',
        },
        'Low': {
            'title'  : 'Psychological Risk — Maintain',
            'color'  : COLOR_LOW,
            'driver' : 'Your profile does not show the level of symptom burden this model associates with psychological risk.',
            'pathway': '',
            'checks' : [],
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
