decisionTree = {
    "root": {
        "question": "Are you experiencing any of the following symptoms: fever, cough, shortness of breath, or fatigue?",
        "fever": "fever",
        "cough": "cough",
        "shortness_of_breath": "shortness_of_breath",
        "fatigue": "fatigue",
        "none": "age",
    },
    "fever": {
        "question": "Do you have a fever above 38°C (100.4°F)?",
        "yes": "high_fever",
        "no": "low_fever",
    },
    "cough": {
        "question": "Do you have a persistent cough?",
        "yes": "persistent_cough",
        "no": "mild_cough",
    },
    "high_fever": {
        "question": "Have you experienced a rapid heart rate or chills?",
        "yes": "flu_like_symptoms",
        "no": "common_cold",
    },
    "low_fever": {
        "question": "Do you have a sore throat or mild cough?",
        "yes": "upper_respiratory_infection",
        "no": "mild_viral_infection",
    },
    "persistent_cough": {
        "question": "Is your cough accompanied by chest pain or wheezing?",
        "yes": "serious_respiratory_condition",
        "no": "upper_respiratory_infection",
    },
    "mild_cough": {
        "question": "Is your cough improving over time?",
        "yes": "recovering_mild_cough",
        "no": "monitor_for_changes",
    },
    "flu_like_symptoms": {
        "question": "Do you have muscle aches or body pain?",
        "yes": "influenza",
        "no": "viral_infection",
    },
    "common_cold": {
        "question": "Do you have a runny nose or congestion?",
        "yes": "common_cold_diagnosis",
        "no": "mild_infection",
    },
    "shortness_of_breath": {
        "question": "Do you have chest pain or tightness?",
        "yes": "serious_respiratory_issue",
        "no": "mild_breathing_issue",
    },
    "serious_respiratory_issue": {
        "question": "Have you been exposed to any respiratory irritants or allergens recently?",
        "yes": "allergic_reaction_or_asthma",
        "no": "potential_respiratory_infection",
    },
    "fatigue": {
        "question": "Is your fatigue accompanied by dizziness or fainting?",
        "yes": "serious_cardiovascular_issue",
        "no": "general_fatigue",
    },
    "serious_cardiovascular_issue": {
        "question": "Do you have a history of heart disease or high blood pressure?",
        "yes": "cardiovascular_condition",
        "no": "investigate_further",
    },
    "age": {
        "question": "What is your age?",
        "<10": "weight_child",
        "10-18": "weight_teen",
        "18-50": "gender",
        ">50": "older_adult",
    },
    "weight_child": {
        "question": "What is your weight (in kg or lbs)?",
        "low_weight": "low_weight_child",
        "normal_weight": "normal_weight_child",
        "high_weight": "high_weight_child",
    },
    "weight_teen": {
        "question": "What is your weight (in kg or lbs)?",
        "low_weight": "low_weight_teen",
        "normal_weight": "normal_weight_teen",
        "high_weight": "high_weight_teen",
    },
    "gender": {
        "question": "What is your gender?",
        "male": "adult_male",
        "female": "pregnancy",
    },
    "pregnancy": {
        "question": "Are you currently pregnant or menstruating?",
        "pregnant": "pregnancy",
        "menstruating": "menstruation",
        "no": "adult_female",
    },
    "pregnancy": {
        "question": "Do you have any abdominal pain or unusual symptoms?",
        "yes": "potential_pregnancy_complication",
        "no": "normal_pregnancy",
    },
    "menstruation": {
        "question": "Are you experiencing heavy bleeding or severe cramps?",
        "yes": "severe_menstruation",
        "no": "normal_menstruation",
    },
    "older_adult": {
        "question": "Do you have any chronic conditions like diabetes, heart disease, or hypertension?",
        "yes": "chronic_condition",
        "no": "healthy_older_adult",
    },
    # Add more branches and symptoms as needed
}