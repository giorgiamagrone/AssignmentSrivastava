"""
prompts.py – prompt templates for Pipeline B.
Includes prompts for schema alignment, record linkage, and data fusion.
"""

PROMPT_VERSION = "v1.0"

#Stage 1: Schema Alignment
#Input: source table name, column names, sample values
#Output: JSON object with a "mappings" list
SCHEMA_ALIGNMENT_PROMPT = """\
You are a data integration expert. Map the columns of a source table to a \
canonical (mediated) schema.

Mediated schema columns: {mediated_cols}

Source table: "{source_name}"
Columns with sample values (up to 5 samples per column):
{col_samples}

Return ONLY a JSON object that strictly follows this structure — no prose, \
no markdown fences:
{{
  "mappings": [
    {{
      "source_column": "<exact source column name>",
      "target_column": "<mediated column name, or null if no good match>",
      "confidence": <float between 0.0 and 1.0>,
      "reason": "<one concise sentence explaining the mapping>"
    }}
  ]
}}

Rules:
1. Include one entry per source column — do not add or omit any.
2. "target_column" must be one of {mediated_cols} or null.
3. confidence = 1.0 → certain mapping; 0.0 → total uncertainty.
4. If two source columns could map to the same target, pick the better one \
   and set the other to null.
5. Output ONLY valid JSON. No text before or after the JSON object.
"""

#Stage 2: Record Linkage (borderline pairs)
#Input: two records with four fields each + source names
#Output: JSON object with decision, confidence, explanation
RECORD_LINKAGE_PROMPT = """\
You are a data integration expert. Decide whether the two bibliographic records \
below describe the SAME real-world paper.

Record A  (source: {source_a})
  Title  : {title_a}
  Authors: {authors_a}
  Venue  : {venue_a}
  Year   : {year_a}

Record B  (source: {source_b})
  Title  : {title_b}
  Authors: {authors_b}
  Venue  : {venue_b}
  Year   : {year_b}

Return ONLY a JSON object — no prose, no markdown fences:
{{
  "decision": "match" or "non_match",
  "confidence": <float between 0.0 and 1.0>,
  "explanation": "<one or two sentences focusing on the key similarities or \
differences that drove your decision>"
}}

Rules:
1. "match" means both records refer to the SAME paper (minor formatting \
   differences are acceptable).
2. "non_match" means they describe different papers.
3. Different titles AND different authors → strong evidence for non_match.
4. Year difference of more than 1 → strong evidence for non_match.
5. confidence ≥ 0.85 means you are highly certain.
6. Output ONLY valid JSON. No text before or after the JSON object.
"""

#Stage 3: Data Fusion
#Input: field name, entity id, source values
#Output: selected value, source, confidence, abstention flag
DATA_FUSION_PROMPT = """\
You are a data curator. Multiple bibliographic sources provide conflicting \
values for the same field of the same paper. Select the most reliable value.

Paper entity ID: {entity_id}
Field          : {field_name}

Source values:
{source_values}

Return ONLY a JSON object — no prose, no markdown fences:
{{
  "selected_value"  : "<the most reliable / canonical value>",
  "selected_source" : "<source name>",
  "confidence"      : <float between 0.0 and 1.0>,
  "reason"          : "<one sentence explanation>",
  "abstain"         : <true if too uncertain to decide, false otherwise>
}}

Rules:
1. For publication year  : prefer 4-digit integers. If sources disagree by \
   more than 1 year, set abstain = true.
2. For title             : prefer properly capitalised, complete titles.
3. For authors           : prefer last-name + initials format, all authors \
   listed.
4. For venue             : prefer full official venue name.
5. Source reliability: DBLP > ACM > Scholar for academic papers.
6. If abstain = true, still provide your best guess in "selected_value".
7. Output ONLY valid JSON. No text before or after the JSON object.
"""