
from pathlib import Path

#Directory layout 
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
GT_DIR       = BASE_DIR / "ground_truth"
RESULTS_DIR  = BASE_DIR / "results"
LOGS_DIR     = BASE_DIR / "logs"

for _d in [DATA_DIR, GT_DIR, RESULTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

#Mediated schema 
MEDIATED_SCHEMA = ["title", "authors", "venue", "year", "abstract", "doi"]

#Per-source column names 
SOURCE_COLS: dict[str, list[str]] = {
    "dblp":    ["paper_title", "author_list",  "journal_or_conf",   "pub_year",
                "abstract",    "doi_link"],
    "acm":     ["Title",       "Authors",       "Publication_Venue", "Year",
                "Abstract",    "DOI"],
    "scholar": ["ArticleTitle","Contributor",   "Source",            "PublicationYear",
                "Summary",     "Identifier"],
}

#Ground-truth schema mappings
SCHEMA_GT: dict[str, dict[str, str]] = {
    src: dict(zip(cols, MEDIATED_SCHEMA))
    for src, cols in SOURCE_COLS.items()
}

#Dataset generation parameters 
DATA_CFG = {
    "n_entities":   500,    
    "dblp_frac":    0.90,   
    "acm_frac":     0.75,  
    "scholar_frac": 0.65,   
    "noise_rate":   0.15,   
    "seed":         42,
}

#Blocking 
BLOCK_CFG = {
    "ngram_range": (2, 3),   
    "top_k":       8,         
    "min_sim":     0.20,     
}

#Record-linkage thresholds 
MATCH_CFG = {
    "hard_threshold":      0.80,  
    "borderline_low":      0.50,  
    "borderline_high":     0.80,
    "non_match_threshold": 0.40,  
}

#LLM
LLM_CFG = {
    "base_url":    "http://localhost:11434",
    "model":       "llama3",   
    "temperature": 0.0,
    "max_tokens":  1024,
    "timeout":     60,         
}

#Data fusion
FUSION_CFG = {
    
    "source_weights": {"dblp": 1.2, "acm": 1.0, "scholar": 0.9},
}

SEED = DATA_CFG["seed"]
