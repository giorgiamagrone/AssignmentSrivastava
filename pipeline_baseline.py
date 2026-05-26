import pandas as pd
import numpy as np
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DATA_DIR, GT_DIR, RESULTS_DIR, BLOCK_CFG, MATCH_CFG, SOURCE_COLS, MEDIATED_SCHEMA, FUSION_CFG

def baseline_schema_alignment():
    print("Esecuzione Schema Alignment (Baseline)...")
    predictions = []
    
    def tokenize(s):
        return set(re.sub(r'[_\-]', ' ', s.lower()).split())

    for source, cols in SOURCE_COLS.items():
        for col in cols:
            best_match = None
            best_score = -1
            col_tokens = tokenize(col)
            
            for target in MEDIATED_SCHEMA:
                tgt_tokens = tokenize(target)
                union = col_tokens | tgt_tokens
                score = len(col_tokens & tgt_tokens) / len(union) if union else 0
                
                if score > best_score:
                    best_score = score
                    best_match = target
                    
            predictions.append({
                'source': source,
                'source_column': col,
                'predicted_target': best_match
            })
            
    pd.DataFrame(predictions).to_csv(RESULTS_DIR / "schema_alignment_predictions.csv", index=False)

def run_baseline_linkage():
    print("Esecuzione Record Linkage (Baseline)...")
    dblp = pd.read_csv(DATA_DIR / "dblp.csv")
    acm = pd.read_csv(DATA_DIR / "acm.csv")
    scholar = pd.read_csv(DATA_DIR / "scholar.csv")
    
    dblp['title_norm'] = dblp['paper_title'].fillna('')
    acm['title_norm'] = acm['Title'].fillna('')
    scholar['title_norm'] = scholar['ArticleTitle'].fillna('')
    
    all_records = pd.concat([
        dblp[['record_id', 'title_norm']].assign(source='dblp'),
        acm[['record_id', 'title_norm']].assign(source='acm'),
        scholar[['record_id', 'title_norm']].assign(source='scholar')
    ]).reset_index(drop=True)
    
    vectorizer = TfidfVectorizer(ngram_range=BLOCK_CFG["ngram_range"])
    tfidf_matrix = vectorizer.fit_transform(all_records['title_norm'])
    
    predicted_matches = []
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    for i in range(len(all_records)):
        for j in range(i + 1, len(all_records)):
            if all_records.iloc[i]['source'] != all_records.iloc[j]['source']:
                score = sim_matrix[i, j]
                if score >= MATCH_CFG["non_match_threshold"]:
                    predicted_matches.append({
                        'left_id': all_records.iloc[i]['record_id'],
                        'right_id': all_records.iloc[j]['record_id'],
                        'score': score,
                        'prediction': 1 if score >= MATCH_CFG["hard_threshold"] else 0
                    })
                    
    df_matches = pd.DataFrame(predicted_matches)
    df_matches.to_csv(RESULTS_DIR / "baseline_predictions.csv", index=False)
    
    n_candidates = len(df_matches)
    n_dblp = len(dblp)
    n_acm = len(acm)
    n_scholar = len(scholar)
    total_cross_source_pairs = (n_dblp * n_acm) + (n_dblp * n_scholar) + (n_acm * n_scholar)

    stats = {
        "total_pairs_evaluated": n_candidates,
        "total_cross_source_pairs": total_cross_source_pairs,
        "reduction_ratio": round(1 - n_candidates / total_cross_source_pairs, 4) if total_cross_source_pairs > 0 else 0
    }
    with open(RESULTS_DIR / "blocking_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Candidate pairs dopo blocking salvati: {n_candidates} / {total_cross_source_pairs} totali")
    print(f"Reduction ratio: {stats['reduction_ratio']:.4f}")

def baseline_fusion():
    print("Esecuzione Data Fusion (Baseline)...")
    baseline_preds = pd.read_csv(RESULTS_DIR / "baseline_predictions.csv")
    
    dblp    = pd.read_csv(DATA_DIR / "dblp.csv")
    acm     = pd.read_csv(DATA_DIR / "acm.csv")
    scholar = pd.read_csv(DATA_DIR / "scholar.csv")

    record_map = {}
    for _, r in dblp.iterrows(): record_map[r['record_id']] = ('dblp', r)
    for _, r in acm.iterrows(): record_map[r['record_id']] = ('acm', r)
    for _, r in scholar.iterrows(): record_map[r['record_id']] = ('scholar', r)

    col_to_med = {
        'paper_title': 'title',    'Title': 'title',    'ArticleTitle': 'title',
        'author_list': 'authors',  'Authors': 'authors', 'Contributor': 'authors',
        'journal_or_conf': 'venue','Publication_Venue': 'venue', 'Source': 'venue',
        'pub_year': 'year',        'Year': 'year',       'PublicationYear': 'year',
        'abstract': 'abstract',    'Abstract': 'abstract', 'Summary': 'abstract',
        'doi_link': 'doi',         'DOI': 'doi',          'Identifier': 'doi',
    }
    weights = FUSION_CFG["source_weights"]
    gt_fusion = pd.read_csv(GT_DIR / "fusion_gt.csv")
    predictions = []

    for _, conf_row in gt_fusion.iterrows():
        eid     = conf_row['entity_id']
        med_fld = conf_row['med_field']
        src     = conf_row['source']
        rec_id  = conf_row['record_id']

        if rec_id not in record_map:
            continue

        src_name, rec = record_map[rec_id]
        candidate_values = [] 

        for col, med in col_to_med.items():
            if med == med_fld and col in rec.index:
                val = rec.get(col)
                if pd.notna(val) and str(val).strip():
                    w = weights.get(src_name, 1.0)
                    candidate_values.append((str(val).strip(), w))

        if not candidate_values:
            selected = ""
        else:
            vote = {}
            for val, w in candidate_values:
                vote[val] = vote.get(val, 0) + w
            selected = max(vote, key=vote.__getitem__)

        predictions.append({
            'entity_id': eid,
            'med_field': med_fld,
            'selected_value': selected
        })

    pd.DataFrame(predictions).to_csv(RESULTS_DIR / "baseline_fusion.csv", index=False)
    print(f"Baseline fusion completata: {len(predictions)} valori prodotti.")

if __name__ == "__main__":
    print("--- Avvio Pipeline A: Baseline Tradizionale ---")
    baseline_schema_alignment()
    run_baseline_linkage()
    baseline_fusion()
    print("Completato!")