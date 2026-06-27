"""
Dataset d'évaluation pour ClinicalRAG (RAGAS).

Chaque entrée contient une question et une réponse de référence ("ground truth"),
rédigée manuellement à partir du contenu connu du corpus (guidelines OMS + PubMed).
Ces réponses de référence servent à calculer le context_recall (le retrieval a-t-il
manqué une info importante présente dans le corpus ?).
"""

EVAL_SET = [
    {
        "question": "What are the first-line drug classes recommended for treating hypertension in adults?",
        "ground_truth": (
            "WHO recommends three first-line drug classes for pharmacological treatment of "
            "hypertension in adults: thiazide and thiazide-like diuretics, ACE inhibitors or "
            "angiotensin receptor blockers (ARBs), and long-acting dihydropyridine calcium "
            "channel blockers (CCBs)."
        ),
    },
    {
        "question": "What are the main risk factors for type 2 diabetes?",
        "ground_truth": (
            "Main risk factors for type 2 diabetes include overweight and obesity, physical "
            "inactivity, family history of diabetes, history of gestational diabetes, "
            "cardiovascular disease, certain ethnic backgrounds (South Asian, African-Caribbean, "
            "Hispanic), and smoking."
        ),
    },
    {
        "question": "What is the global action plan on antimicrobial resistance aiming to achieve?",
        "ground_truth": (
            "The WHO global action plan on antimicrobial resistance aims to ensure, for as long "
            "as possible, the continuity of successful treatment and prevention of infectious "
            "diseases with effective and safe medicines, through improved awareness, surveillance, "
            "infection control, and sustainable investment in new tools."
        ),
    },
    {
        "question": "What lifestyle modifications are recommended for managing obesity-related diabetes risk?",
        "ground_truth": (
            "Recommended lifestyle modifications include weight loss, dietary changes, increased "
            "physical activity, and smoking cessation counselling for people at risk of developing "
            "type 2 diabetes."
        ),
    },
    {
        "question": "What recommendation does WHO give regarding combination therapy for hypertension?",
        "ground_truth": (
            "WHO suggests combination therapy, preferably as a single-pill combination to improve "
            "adherence and persistence, as initial treatment for hypertension, particularly when "
            "blood pressure is significantly elevated. The combination should draw from diuretics, "
            "ACE inhibitors/ARBs, and calcium channel blockers."
        ),
    },
    {
        "question": "What is the target blood pressure for adults with hypertension without comorbidities?",
        "ground_truth": (
            "WHO recommends a target blood pressure below 140/90 mmHg for adults with hypertension "
            "who do not have other comorbidities."
        ),
    },
]
