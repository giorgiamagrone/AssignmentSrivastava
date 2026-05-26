import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from config import GT_DIR, RESULTS_DIR, SCHEMA_GT

def evaluate_schema_alignment(predictions_file, pipeline_name):
    print(f"\n--- Valutazione Schema Alignment ({pipeline_name}) ---")
    try:
        preds = pd.read_csv(RESULTS_DIR / predictions_file)
    except FileNotFoundError:
        print(f"File {predictions_file} non trovato. Salto la valutazione.")
        return

    correct, total = 0, 0
    for _, row in preds.iterrows():
        gt_mapping = SCHEMA_GT.get(row['source'], {})
        expected = gt_mapping.get(row['source_column'])
        predicted = row['predicted_target'] if pd.notna(row['predicted_target']) else None
        
        if expected == predicted:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"Accuracy: {acc:.4f} ({correct}/{total} corretti)")

def evaluate_linkage(predictions_file, pipeline_name):
    print(f"\n--- Valutazione Record Linkage ({pipeline_name}) ---")
    
    gt_pairs = pd.read_csv(GT_DIR / "match_pairs.csv")
    try:
        preds = pd.read_csv(RESULTS_DIR / predictions_file)
    except FileNotFoundError:
        print(f"File {predictions_file} non trovato.")
        return

    def make_key(row, col1, col2):
        return "_".join(sorted([str(row[col1]), str(row[col2])]))
        
    gt_pairs['pair_key'] = gt_pairs.apply(lambda r: make_key(r, 'left_id', 'right_id'), axis=1)
    preds['pair_key'] = preds.apply(lambda r: make_key(r, 'left_id', 'right_id'), axis=1)
    
    eval_df = gt_pairs.merge(preds[['pair_key', 'prediction']], on='pair_key', how='outer')
    
    eval_df['label'] = eval_df['label'].fillna(0) 
    eval_df['prediction'] = eval_df['prediction'].fillna(0)
    
    y_true = eval_df['label'].astype(int)
    y_pred = eval_df['prediction'].astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

def evaluate_fusion(predictions_file, pipeline_name, sample_file=None):
    print(f"\n--- Valutazione Data Fusion ({pipeline_name}) ---")
    gt = pd.read_csv(GT_DIR / "fusion_gt.csv")
    try:
        preds = pd.read_csv(RESULTS_DIR / predictions_file)
    except FileNotFoundError:
        print(f"File {predictions_file} non trovato. Salto la valutazione.")
        return

    if sample_file:
        try:
            samples = pd.read_csv(RESULTS_DIR / sample_file)
            gt_eval = gt.merge(samples, on=['entity_id', 'med_field'], how='inner')
        except FileNotFoundError:
            gt_eval = gt
    else:
        gt_eval = gt.merge(preds[['entity_id', 'med_field']], on=['entity_id', 'med_field'], how='inner')
        
    merged  = gt_eval.merge(preds, on=['entity_id', 'med_field'], how='left')
    merged['selected_value'] = merged['selected_value'].fillna('')
    
    correct = (merged['canonical'].astype(str).str.strip() == 
               merged['selected_value'].astype(str).str.strip()).sum()
    total = len(merged)
    acc = correct / total if total > 0 else 0
    print(f"Accuracy: {acc:.4f} ({correct}/{total} corretti) [su {total} testati]")

if __name__ == "__main__":
    evaluate_schema_alignment("schema_alignment_predictions.csv", "Baseline Tradizionale")
    evaluate_schema_alignment("schema_alignment_llm.csv", "LLM-Assisted Pipeline")
    evaluate_linkage("baseline_predictions.csv", "Baseline Tradizionale")
    evaluate_linkage("llm_predictions.csv", "LLM-Assisted Pipeline")
    evaluate_fusion("baseline_fusion.csv", "Baseline Tradizionale")
    evaluate_fusion("llm_fusion.csv", "LLM-Assisted Pipeline", "llm_fusion_sample_ids.csv")