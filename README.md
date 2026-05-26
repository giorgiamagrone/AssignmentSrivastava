# LLM-Assisted Big Data Integration

This repository contains the source code, data, and evaluation scripts for the final project of the **Advanced Topics in Computer Science** course. 

The project compares a traditional deterministic Big Data Integration pipeline with a hybrid architecture that selectively utilizes Large Language Models (LLMs) to resolve complex semantic conflicts across three simulated bibliographic sources (DBLP, ACM, Google Scholar).

## Repository Structure

* `data/`: Contains the synthetic dataset sources (`acm.csv`, `dblp.csv`, `scholar.csv`).
* `ground_truth/`: Contains the reference files (`entities.csv`, `match_pairs.csv`, `fusion_gt.csv`) used to evaluate the pipelines.
* `logs/`: Contains the raw JSONL logs of the LLM responses (`llm_responses.jsonl`).
* `results/`: Directory where the pipeline output predictions and statistics are saved.
* `Technical_Report.pdf`: The final academic report detailing methodology, metrics, and error analysis.

## Requirements & Setup

This project uses Python. To install the required dependencies, run:

```
pip install -r requirements.txt
```
Note: The LLM pipeline requires a local instance of Ollama running on localhost:11434 with the llama3 model installed.
## How to Run the Pipelines
To reproduce the experiments, execute the scripts in the following order from the root directory:
1. Generate the Dataset (Optional, data is already provided in /data):
```
python data_loader.py
```
2. Run the Traditional Baseline (Pipeline A):
```
   python pipeline_baseline.py
```
3. Run the LLM-Assisted Integration (Pipeline B):
```
python pipeline_llm.py
```
4.Evaluate Results and Compute Metrics:
```
python evaluation.py
```



