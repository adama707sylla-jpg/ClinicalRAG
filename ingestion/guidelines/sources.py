"""
Sources des guidelines cliniques (OMS) pour ClinicalRAG.

Note: iris.who.int a migré vers DSpace 7 — seules les URLs au format
/server/api/core/bitstreams/{UUID}/content servent le PDF brut.
Les anciennes URLs /bitstream/handle/... renvoient l'app JS (HTML).
"""

GUIDELINES = [
    {
        "name": "who_diabetes_diagnosis_management",
        "topic": "diabete_type2",
        "url": "https://iris.who.int/server/api/core/bitstreams/2a0b4f68-7155-4ad1-b543-945791e31830/content",
        "title": "Diagnosis and management of type 2 diabetes",
    },
    {
        "name": "who_global_report_diabetes",
        "topic": "diabete_type2",
        "url": "https://iris.who.int/server/api/core/bitstreams/d2997184-51c3-4b0b-aa2c-0cc840424b6c/content",
        "title": "Global report on diabetes",
    },
    {
        "name": "who_hypertension_guideline",
        "topic": "hypertension",
        "url": "https://iris.who.int/server/api/core/bitstreams/f062769d-f075-4a00-87af-0a2106e0bd04/content",
        "title": "Guideline for the pharmacological treatment of hypertension in adults",
    },
    {
        "name": "who_chronic_respiratory_diseases",
        "topic": "asthme",
        "url": "https://iris.who.int/server/api/core/bitstreams/e5dfafb7-6109-48eb-8aa0-59b919a6ddc8/content",
        "title": "Global scientific panel on chronic respiratory diseases (asthma & COPD)",
    },
    {
        "name": "who_cvd_prevention",
        "topic": "maladies_cardiovasculaires",
        "url": "https://iris.who.int/server/api/core/bitstreams/f106282d-01ad-4978-9a61-27fb1a8c305d/content",
        "title": "Prevention of cardiovascular disease: guidelines for assessment and management of total cardiovascular risk",
    },
    {
        "name": "who_amr_global_action_plan",
        "topic": "antibioresistance",
        "url": "https://iris.who.int/server/api/core/bitstreams/1a487887-e162-46a0-8aef-802907c66070/content",
        "title": "Global action plan on antimicrobial resistance",
    },
]
