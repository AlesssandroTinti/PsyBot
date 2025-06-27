# Questionnaires aliases for questionnaire subscales
QUESTIONNAIRE_ALIASES = {
    "PANAS PA": "PANAS",
    "PANAS NA": "PANAS",
    "BIS": "BIS/BAS",
    "BAS Drive": "BIS/BAS", 
    "BAS Fun Seeking": "BIS/BAS", 
    "BAS Reward Resp.": "BIS/BAS"
}

# Keywords for detecting ICD-11 disease labels in text
DISEASE_KEYWORDS = {
    "disorder", "syndrome", "disease", "condition", "distress", "disorders"
    "psychosis", "depression", "anxiety", "phobia", "deficit", "nervosa",
    "symptomatic", "symptom", "induced", "mild", "moderate", "severe", "profound",
    "impairment", "type", "unspecified", "with", "due", "episode", "acute"
}

# List of supported questionnaire names
QUESTIONNAIRES = {
    "STAI-Y", "PSQI", "BIS", "BAS Drive", "BAS Fun Seeking", "BAS Reward Resp.", "MEQ", "PANAS PA", "PANAS NA"
}

# Keywords related to scoring and diagnosis questions
COMPARISON_KEYWORDS = {"differences", "difference", "compare", "comparison", "vs", "versus"}
QUESTIONNAIRE_KEYWORDS = {
    "score", "scores", "scale", "scales", "evaluation", "evaluations", "valuation", 
    "valuations", "questionnaire", "questionnaires", "evaluate", "diagnose", "diagnosis", 
    "transcription", "transcriptions", "transcript", "transcripts", "disorder", "disorders"
}