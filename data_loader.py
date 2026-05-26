"""
data_loader.py - generates three heterogeneous bibliographic sources
(DBLP, ACM, Scholar) for the data integration pipeline.
"""

import random
import numpy as np
import pandas as pd

from config import DATA_CFG, SOURCE_COLS, SCHEMA_GT, MEDIATED_SCHEMA, \
                   DATA_DIR, GT_DIR, SEED

# ocabulary tables

TECHNIQUES = [
    "Query Processing", "Entity Resolution", "Data Integration", "Schema Matching",
    "Record Linkage", "Knowledge Graph Embedding", "Graph Neural Networks",
    "Information Extraction", "Named Entity Recognition", "Relation Extraction",
    "Text Classification", "Sentiment Analysis", "Topic Modelling", "Dense Retrieval",
    "Approximate Nearest Neighbour", "Collaborative Filtering", "Matrix Factorisation",
    "Anomaly Detection", "Time Series Forecasting", "Differential Privacy",
    "Federated Learning", "Transfer Learning", "Contrastive Learning",
    "Self-Supervised Learning", "Active Learning", "Weak Supervision",
    "Data Augmentation", "Scalable Indexing", "Bloom Filters", "LSH Hashing",
    "Stream Processing", "Column Stores", "Data Fusion", "Truth Discovery",
    "Causal Inference", "Explainable AI", "Data Cleaning", "Duplicate Detection",
    "Data Profiling", "Data Wrangling",
]

DOMAINS = [
    "Large-Scale Databases", "Social Networks", "Biomedical Data", "E-Commerce",
    "Web Data", "Knowledge Bases", "Heterogeneous Sources", "Scientific Literature",
    "Enterprise Data", "Open Government Data", "Healthcare Records",
    "Financial Transactions", "Product Catalogues", "News Articles", "IoT Sensor Data",
    "Graph Databases", "Spatial Data", "Temporal Data", "Multilingual Corpora",
    "Streaming Data",
]

ADJECTIVES = [
    "Scalable", "Efficient", "Robust", "Unified", "Probabilistic", "Adaptive",
    "Joint", "End-to-End", "Weakly-Supervised", "Semi-Supervised", "Generative",
    "Discriminative", "Hierarchical", "Iterative", "Parallel", "Context-Aware",
    "Multi-Source", "Neural", "Learned", "Hybrid",
]

TITLE_TEMPLATES = [
    "{tech}: A {adj} Approach",
    "{tech} for {domain}",
    "Towards {adj} {tech}",
    "{adj} {tech} at Scale",
    "{tech}: Challenges and Opportunities",
    "A {adj} Framework for {tech}",
    "{tech} via {domain}",
    "Improving {tech} with {adj} Models",
    "{adj} {tech}: A Survey",
    "{tech} in the Wild",
    "{tech} with {adj} Constraints",
    "Robust {tech} for {domain}",
    "{tech}: From Theory to Practice",
    "Learning-Based {tech}",
    "{tech} Over {domain}",
]

VENUES_SHORT = [
    "VLDB", "SIGMOD", "ICDE", "EDBT", "PODS", "ICDM", "KDD", "SDM",
    "NeurIPS", "ICML", "ICLR", "CVPR", "ACL", "EMNLP", "NAACL",
    "AAAI", "IJCAI", "CIKM", "SIGIR", "WWW", "WSDM",
    "TKDE", "TODS", "TKDD", "VLDBJ", "IS", "JMLR", "AIJ", "DMKD", "KAIS",
]

VENUE_FULL = {
    "VLDB": "Proc. VLDB Endowment",
    "SIGMOD": "ACM SIGMOD Int. Conf. Management of Data",
    "ICDE": "IEEE Int. Conf. on Data Engineering",
    "EDBT": "Int. Conf. on Extending Database Technology",
    "PODS": "ACM Symposium on Principles of Database Systems",
    "ICDM": "IEEE Int. Conf. on Data Mining",
    "KDD": "ACM SIGKDD Conf. on Knowledge Discovery and Data Mining",
    "SDM": "SIAM Int. Conf. on Data Mining",
    "NeurIPS": "Advances in Neural Information Processing Systems",
    "ICML": "Int. Conf. on Machine Learning",
    "ICLR": "Int. Conf. on Learning Representations",
    "CVPR": "IEEE/CVF Conf. Computer Vision and Pattern Recognition",
    "ACL": "Annual Meeting of the Association for Computational Linguistics",
    "EMNLP": "Conf. on Empirical Methods in Natural Language Processing",
    "NAACL": "North American Chapter of the ACL",
    "AAAI": "AAAI Conf. on Artificial Intelligence",
    "IJCAI": "Int. Joint Conf. on Artificial Intelligence",
    "CIKM": "Int. Conf. on Information and Knowledge Management",
    "SIGIR": "ACM SIGIR Conf. on Research and Development in IR",
    "WWW": "Int. World Wide Web Conference",
    "WSDM": "ACM Int. Conf. on Web Search and Data Mining",
    "TKDE": "IEEE Transactions on Knowledge and Data Engineering",
    "TODS": "ACM Transactions on Database Systems",
    "TKDD": "ACM Transactions on Knowledge Discovery from Data",
    "VLDBJ": "The VLDB Journal",
    "IS": "Information Systems",
    "JMLR": "Journal of Machine Learning Research",
    "AIJ": "Artificial Intelligence",
    "DMKD": "Data Mining and Knowledge Discovery",
    "KAIS": "Knowledge and Information Systems",
}

FIRST_NAMES = [
    "James", "Maria", "Wei", "Anna", "David", "Elena", "Ahmed", "Laura",
    "Chen", "Sara", "Michael", "Julia", "Peng", "Alice", "Robert", "Emma",
    "Diego", "Fatima", "Yuki", "Nadia", "Tom", "Lisa", "Marco", "Grace",
    "Carlos", "Priya", "Eric", "Sophie", "Jin", "Rachel", "Alex", "Olga",
    "Kenji", "Ines", "Pedro", "Mei", "Sven", "Alicia", "Reza", "Natasha",
]

LAST_NAMES = [
    "Smith", "Wang", "Müller", "Garcia", "Chen", "Kumar",
    "Rossi", "Lee", "Johnson", "Li", "Brown", "Zhang",
    "Martinez", "Patel", "Wilson", "Nguyen", "Tanaka", "Kovacs",
    "Silva", "O'Brien", "Dubois", "Fischer", "Yamamoto", "Cohen",
    "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Thompson", "Johansson", "Deng", "Sharma", "Ferreira", "Kim",
    "Nakamura", "Okonkwo", "Petrov", "Hernandez",
]


# Helper functions

def _make_title(idx: int) -> str:
    tpl = TITLE_TEMPLATES[idx % len(TITLE_TEMPLATES)]
    tech = TECHNIQUES[(idx * 3) % len(TECHNIQUES)]
    adj = ADJECTIVES[(idx * 7) % len(ADJECTIVES)]
    dom = DOMAINS[(idx * 11) % len(DOMAINS)]
    return tpl.format(tech=tech, adj=adj, domain=dom)


def _make_abstract(title: str) -> str:
    starters = [
        "In this paper, we propose",
        "We present a novel approach for",
        "This work addresses the problem of",
        "We introduce a new method for",
        "We study the problem of",
    ]
    s = starters[abs(hash(title)) % len(starters)]
    return (f"{s} {title.lower()}. "
            "Extensive experiments on real-world benchmarks demonstrate "
            "the effectiveness of the proposed approach.")


def _make_doi(entity_id: int, venue: str, year: int) -> str:
    return f"10.{1000 + (entity_id % 9000)}/{venue.lower()}.{year}.{entity_id:05d}"


def _make_authors(entity_id: int) -> list[tuple[str, str]]:
    n = 1 + (entity_id % 4)
    return [
        (FIRST_NAMES[(entity_id * (i + 1) * 3) % len(FIRST_NAMES)],
         LAST_NAMES[(entity_id * (i + 1) * 7) % len(LAST_NAMES)])
        for i in range(n)
    ]


# source-specific formatting

def _fmt_authors_dblp(authors: list) -> str:
    return " and ".join(f"{last}, {first[0]}." for first, last in authors)

def _fmt_authors_acm(authors: list) -> str:
    return ", ".join(f"{first[0]}. {last}" for first, last in authors)

def _fmt_authors_scholar(authors: list) -> str:
    return "; ".join(f"{first} {last}" for first, last in authors)

def _venue_scholar(venue: str, year: int) -> str:
    return VENUE_FULL.get(venue, venue) if (hash(venue + str(year)) % 2 == 0) else venue


#entity generation

def generate_entities(n: int = 500) -> pd.DataFrame:
    rows = []
    for i in range(n):
        venue = VENUES_SHORT[i % len(VENUES_SHORT)]
        year = 2000 + (i % 24)
        authors = _make_authors(i)
        title = _make_title(i)
        rows.append({
            "entity_id": i,
            "title": title,
            "authors": _fmt_authors_dblp(authors),
            "venue": venue,
            "year": year,
            "abstract": _make_abstract(title),
            "doi": _make_doi(i, venue, year),
            "_raw_authors": authors,
        })
    return pd.DataFrame(rows)


# Source creation

def create_source(entities: pd.DataFrame, source: str, frac: float, seed: int = SEED) -> pd.DataFrame:
    n = int(len(entities) * frac)
    selected = entities.sample(n=n, random_state=seed).reset_index(drop=True)
    rows = []

    for _, e in selected.iterrows():
        eid = int(e["entity_id"])
        authors = e["_raw_authors"]

        if source == "dblp":
            row = {
                "record_id": f"dblp_{eid:05d}",
                "entity_id": eid,
                "paper_title": e["title"],
                "author_list": _fmt_authors_dblp(authors),
                "journal_or_conf": e["venue"],
                "pub_year": int(e["year"]),
                "abstract": e["abstract"],
                "doi_link": e["doi"],
            }
        elif source == "acm":
            row = {
                "record_id": f"acm_{eid:05d}",
                "entity_id": eid,
                "Title": e["title"],
                "Authors": _fmt_authors_acm(authors),
                "Publication_Venue": VENUE_FULL.get(e["venue"], e["venue"]),
                "Year": int(e["year"]),
                "Abstract": e["abstract"],
                "DOI": e["doi"],
            }
        else:
            row = {
                "record_id": f"scholar_{eid:05d}",
                "entity_id": eid,
                "ArticleTitle": e["title"],
                "Contributor": _fmt_authors_scholar(authors),
                "Source": _venue_scholar(e["venue"], e["year"]),
                "PublicationYear": int(e["year"]),
                "Summary": e["abstract"],
                "Identifier": e["doi"],
            }
        rows.append(row)

    return pd.DataFrame(rows)


#Noise injection

def _perturb_title(title: str, rng_local: random.Random) -> str:
    words = title.split()
    if len(words) < 2:
        return title
    idx = rng_local.randint(0, len(words) - 1)
    w = words[idx]
    if len(w) > 3:
        pos = rng_local.randint(0, len(w) - 2)
        w = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
    words[idx] = w
    return " ".join(words)


def _perturb_year(year, rng_local: random.Random):
    return int(year) + rng_local.choice([-1, 1])


def _perturb_venue(venue: str, rng_local: random.Random) -> str:
    if venue in VENUE_FULL:
        return VENUE_FULL[venue]
    for short, full in VENUE_FULL.items():
        if full == venue:
            return short
    return venue + " Workshop"


def _perturb_authors(authors: str, rng_local: random.Random) -> str:
    if " and " in authors:
        sep = " and "
    elif "; " in authors:
        sep = "; "
    else:
        sep = ", "
    parts = authors.split(sep)
    if len(parts) > 1:
        rng_local.shuffle(parts)
    return sep.join(parts)


_PERTURB_FIELDS = {
    "dblp": [("paper_title", _perturb_title),
             ("pub_year", _perturb_year),
             ("journal_or_conf", _perturb_venue),
             ("author_list", _perturb_authors)],
    "acm": [("Title", _perturb_title),
            ("Year", _perturb_year),
            ("Publication_Venue", _perturb_venue),
            ("Authors", _perturb_authors)],
    "scholar": [("ArticleTitle", _perturb_title),
                ("PublicationYear", _perturb_year),
                ("Source", _perturb_venue),
                ("Contributor", _perturb_authors)],
}


def inject_noise(df: pd.DataFrame, source: str, rate: float = 0.09, seed: int = SEED) -> tuple[pd.DataFrame, list[dict]]:
    rng_local = random.Random(seed)
    n_noisy = max(1, int(len(df) * rate))
    noisy_idxs = rng_local.sample(range(len(df)), n_noisy)
    field_fns = _PERTURB_FIELDS[source]

    df = df.copy().reset_index(drop=True)
    conflict_log = []

    for idx in noisy_idxs:
        field, fn = rng_local.choice(field_fns)
        orig = df.at[idx, field]
        try:
            noisy = fn(orig, rng_local)
        except Exception:
            continue
        if str(noisy) == str(orig):
            continue
        df.at[idx, field] = noisy
        conflict_log.append({
            "record_id": df.at[idx, "record_id"],
            "source": source,
            "field": field,
            "original": str(orig),
            "noisy": str(noisy),
            "entity_id": int(df.at[idx, "entity_id"]),
        })

    return df, conflict_log


#Ground-truth construction 

def build_match_pairs(dblp: pd.DataFrame, acm: pd.DataFrame, scholar: pd.DataFrame) -> pd.DataFrame:
    dblp_map = dblp.set_index("entity_id")["record_id"].to_dict()
    acm_map = acm.set_index("entity_id")["record_id"].to_dict()
    scholar_map = scholar.set_index("entity_id")["record_id"].to_dict()

    all_eids = set(dblp_map) | set(acm_map) | set(scholar_map)
    rows = []

    for eid in all_eids:
        present = []
        if eid in dblp_map: present.append(("dblp", dblp_map[eid]))
        if eid in acm_map: present.append(("acm", acm_map[eid]))
        if eid in scholar_map: present.append(("scholar", scholar_map[eid]))

        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                src_a, rid_a = present[i]
                src_b, rid_b = present[j]
                rows.append({"left_source": src_a, "left_id": rid_a,
                             "right_source": src_b, "right_id": rid_b,
                             "entity_id": eid, "label": 1})

    n_pos = sum(1 for r in rows if r["label"] == 1)

    all_records = (
        [(r["record_id"], r["entity_id"], "dblp") for _, r in dblp.iterrows()] +
        [(r["record_id"], r["entity_id"], "acm") for _, r in acm.iterrows()] +
        [(r["record_id"], r["entity_id"], "scholar") for _, r in scholar.iterrows()]
    )
    rng_local = random.Random(SEED + 999)
    neg_count = 0
    attempts = 0

    while neg_count < n_pos and attempts < n_pos * 30:
        attempts += 1
        rid_a, eid_a, src_a = rng_local.choice(all_records)
        rid_b, eid_b, src_b = rng_local.choice(all_records)
        if eid_a != eid_b and src_a != src_b:
            rows.append({"left_source": src_a, "left_id": rid_a,
                         "right_source": src_b, "right_id": rid_b,
                         "entity_id": -1, "label": 0})
            neg_count += 1

    return pd.DataFrame(rows)


_SRC_TO_MED = {
    "dblp": {"paper_title": "title", "author_list": "authors", "journal_or_conf": "venue", "pub_year": "year", "abstract": "abstract", "doi_link": "doi"},
    "acm": {"Title": "title", "Authors": "authors", "Publication_Venue": "venue", "Year": "year", "Abstract": "abstract", "DOI": "doi"},
    "scholar": {"ArticleTitle": "title", "Contributor": "authors", "Source": "venue", "PublicationYear": "year", "Summary": "abstract", "Identifier": "doi"},
}


def build_fusion_gt(conflicts: list[dict], entities: pd.DataFrame) -> pd.DataFrame:
    ent_idx = entities.set_index("entity_id")
    rows = []

    for c in conflicts:
        src = c["source"]
        field = c["field"]
        med_col = _SRC_TO_MED.get(src, {}).get(field)
        if med_col is None:
            continue
        eid = c["entity_id"]
        if eid not in ent_idx.index:
            continue
        canonical = ent_idx.at[eid, med_col]
        if isinstance(canonical, list):
            canonical = str(canonical)
        rows.append({
            "entity_id": eid,
            "record_id": c["record_id"],
            "source": src,
            "source_field": field,
            "med_field": med_col,
            "noisy_value": c["noisy"],
            "canonical": str(canonical),
        })

    return pd.DataFrame(rows)


# main entry point

def load_all(force_regenerate: bool = False):
    paths = {
        "dblp": DATA_DIR / "dblp.csv",
        "acm": DATA_DIR / "acm.csv",
        "scholar": DATA_DIR / "scholar.csv",
        "pairs": GT_DIR / "match_pairs.csv",
        "fusion": GT_DIR / "fusion_gt.csv",
        "entities": GT_DIR / "entities.csv",
    }

    if not force_regenerate and all(p.exists() for p in paths.values()):
        print("[data_loader] Loading cached data from CSV...")
        dblp = pd.read_csv(paths["dblp"])
        acm = pd.read_csv(paths["acm"])
        scholar = pd.read_csv(paths["scholar"])
        gt_pairs = pd.read_csv(paths["pairs"])
        gt_fusion = pd.read_csv(paths["fusion"])
        entities = pd.read_csv(paths["entities"])
        _print_stats(dblp, acm, scholar, gt_pairs, gt_fusion)
        return dblp, acm, scholar, gt_pairs, gt_fusion, entities

    print("[data_loader] Generating synthetic dataset...")
    cfg = DATA_CFG

    entities = generate_entities(n=cfg["n_entities"])

    dblp = create_source(entities, "dblp", cfg["dblp_frac"], seed=cfg["seed"])
    acm = create_source(entities, "acm", cfg["acm_frac"], seed=cfg["seed"] + 1)
    scholar = create_source(entities, "scholar", cfg["scholar_frac"], seed=cfg["seed"] + 2)

    dblp, conf_d = inject_noise(dblp, "dblp", cfg["noise_rate"], seed=cfg["seed"])
    acm, conf_a = inject_noise(acm, "acm", cfg["noise_rate"], seed=cfg["seed"] + 100)
    scholar, conf_s = inject_noise(scholar, "scholar", cfg["noise_rate"], seed=cfg["seed"] + 200)
    all_conflicts = conf_d + conf_a + conf_s

    gt_pairs = build_match_pairs(dblp, acm, scholar)
    gt_fusion = build_fusion_gt(all_conflicts, entities)

    for df in (dblp, acm, scholar):
        df.drop(columns=["entity_id"], inplace=True, errors="ignore")
    entities.drop(columns=["_raw_authors"], inplace=True, errors="ignore")

    dblp.to_csv(paths["dblp"], index=False)
    acm.to_csv(paths["acm"], index=False)
    scholar.to_csv(paths["scholar"], index=False)
    gt_pairs.to_csv(paths["pairs"], index=False)
    gt_fusion.to_csv(paths["fusion"], index=False)
    entities.to_csv(paths["entities"], index=False)

    _print_stats(dblp, acm, scholar, gt_pairs, gt_fusion)
    return dblp, acm, scholar, gt_pairs, gt_fusion, entities


def _print_stats(dblp, acm, scholar, gt_pairs, gt_fusion):
    print(f"  DBLP   : {len(dblp)} records")
    print(f"  ACM    : {len(acm)} records")
    print(f"  Scholar: {len(scholar)} records")
    print(f"  Total  : {len(dblp) + len(acm) + len(scholar)} records")
    print(f"  GT pos : {(gt_pairs['label'] == 1).sum()} positive pairs")
    print(f"  GT neg : {(gt_pairs['label'] == 0).sum()} negative pairs")
    print(f"  GT fus : {len(gt_fusion)} fusion conflicts")


if __name__ == "__main__":
    load_all(force_regenerate=True)