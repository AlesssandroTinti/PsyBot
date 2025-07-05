# Canonical questionnaires mapped to their alias variants
ALIAS_GROUPS = {
    "PANAS": [
        "PANAS PA", "PANAS NA", "PANAS-PA", "PANAS-NA",
        "PANAS_PA", "PANAS_NA", "PA", "NA"
    ],
    "BIS/BAS": [
        "BIS", "BAS Drive", "BAS Fun Seeking", "BAS Reward Resp.",
        "BAS-Drive", "BAS-Fun-Seeking", "BAS-Reward-Resp.",
        "BAS_Drive", "BAS_Fun_Seeking", "BAS_Reward_Resp.",
        "Drive", "Fun Seeking", "Fun-Seeking", "Fun_Seeking",
        "Reward Resp.", "Reward-Resp.", "Reward_Resp."
    ],
    "STAI-Y": ["STAI-Y"],
    "PSQI": ["PSQI"],
    "MEQ": ["MEQ"]
}

# Dynamically generated aliases and questionnaire keys
ALIASES = {alias: canonical for canonical, aliases in ALIAS_GROUPS.items() for alias in aliases}
QUESTIONNAIRES_KEYWORDS = set(ALIASES.keys())  # List of supported questionnaire names

# Keywords for detecting ICD-11 disease labels in user input
DISEASE_KEYWORDS = {
    "disorder", "syndrome", "disease", "condition", "distress",
    "psychosis", "depression", "anxiety", "phobia", "deficit", "nervosa",
    "symptomatic", "symptom", "induced", "mild", "moderate", "severe", "profound",
    "impairment", "type", "unspecified", "with", "due", "episode", "acute"
}

# Keyword triggers for comparison queries
COMPARISON_KEYWORDS = {
    "differences", "difference", "compare", "comparison", "vs", "versus"
}

# Keywords signaling diagnosis/score-related queries
DIAGNOSIS_KEYWORDS = {
    "score", "scores", "scale", "scales", "evaluation", "evaluations", "valuation", 
    "valuations", "questionnaire", "questionnaires", "evaluate", "diagnose", "diagnosis"
}

# Keywords signaling interpretation/score-related queries
INTERPRETATION_KEYWORDS = {
    "interpret", "interpretation", "transcription", "transcriptions", "transcript", "transcripts",
    "dialogue", "session", "psychological", "therapy", "therapeutic"
}

# Keywords signaling clinical/score-related queries
CLINICAL_KEYWORDS = {
    "three", "3", "suppose", "suppose", "suggest", "suggested", "propose", "proposed",
    "recommend", "recommended", "hypothesize", "hypothesized"
}

# Canonical prompts names mapped to their keywords
KEYWORD_GROUPS = {
    "interpretation": INTERPRETATION_KEYWORDS,  
    "comparison": COMPARISON_KEYWORDS,          
    "diagnosis": DIAGNOSIS_KEYWORDS,            
    "clinical": CLINICAL_KEYWORDS               
}