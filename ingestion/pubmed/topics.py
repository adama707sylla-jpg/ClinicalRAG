"""
Configuration des thématiques de recherche PubMed pour ClinicalRAG.
Chaque entrée définit une requête de recherche PubMed (syntaxe Entrez)
et le nombre cible d'articles à récupérer.
"""

TOPICS = [
    {
        "name": "diabete_type2",
        "query": "type 2 diabetes mellitus[MeSH Terms] AND treatment",
        "max_results": 130,
    },
    {
        "name": "hypertension",
        "query": "hypertension[MeSH Terms] AND management",
        "max_results": 130,
    },
    {
        "name": "asthme",
        "query": "asthma[MeSH Terms] AND treatment",
        "max_results": 100,
    },
    {
        "name": "depression",
        "query": "depressive disorder[MeSH Terms] AND therapy",
        "max_results": 100,
    },
    {
        "name": "maladies_cardiovasculaires",
        "query": "cardiovascular diseases[MeSH Terms] AND prevention",
        "max_results": 130,
    },
    {
        "name": "obesite",
        "query": "obesity[MeSH Terms] AND management",
        "max_results": 100,
    },
    {
        "name": "antibioresistance",
        "query": "drug resistance, bacterial[MeSH Terms]",
        "max_results": 100,
    },
    {
        "name": "cancer_general",
        "query": "neoplasms[MeSH Terms] AND early detection",
        "max_results": 110,
    },
]

TOTAL_TARGET = sum(t["max_results"] for t in TOPICS)
