# CommonWhyDataset

This repository contains the resources for the paper submitted to SIGIR 2026:  
**“CommonWhy: A Dataset for Evaluating Entity-Based Causal Commonsense Reasoning in Large Language Models.”**


# Introduction

This repository provides the CommonWhy dataset and implementations of the baseline methods. CommonWhy consists of 15,000 *why* questions with ground-truth answers that explain the underlying cause of the event described in each question.
The dataset is designed to evaluate the capability of large language models (LLMs) in abductive, entity-based causal commonsense reasoning, as well as the commonsense reasoning ability of knowledge graph question answering (KGQA) methods.


## Data Downloading Instructions

You can access the dataset by clicking on the [**dataset**](https://github.com/faezemoradik/CommonWhyDataset/blob/main/data_files/inference_rule_entities_with_grounded_qa.json) link and downloading the file.

The dataset is provided in `.json` format and can be loaded and viewed using standard software tools that support JSON files, such as VS Code, Sublime Text, Notepad++, or any other compatible editor.

In order to download the relevant KG subgarphs run Retrieve_rel_KG.py.


CommonWhy includes different reasoning skills, illustrated [**here**](https://github.com/faezemoradik/CommonWhyDataset/blob/main/data_statistics/label_distribution_bar_plot.pdf).




The dataset is divided into two subsets based on the popularity of the entities appearing in the questions:

- [**Long-tail subset**](https://github.com/faezemoradik/CommonWhyDataset/blob/main/data_files/Longtail.json): contains half of the queries involving less popular entities. 
- [**Head subset**](https://github.com/faezemoradik/CommonWhyDataset/blob/main/data_files/Head.json): contains half of the queries involving more popular entities.  

The distribution of these two subsets is illustrated [**here**](https://github.com/faezemoradik/CommonWhyDataset/blob/main/data_statistics/popularity_split_histogram.pdf).





## Data Format

The dataset is provided in JSON format. Each entry consists of a commonsense axiom as the key, while the corresponding value contains the associated questions and answers.
Specifically, the value is itself a dictionary with multiple keys that store different pieces of information used to generate the question–answer pairs. The `"grounded_qa"` key contains a list of all question–answer pairs grounded in the given commonsense axiom.

An exemplar entry of the dataset:
```json
{
"If a person A died before 2002, then person A could not have attended El Último Tour Del Mundo.": {
    "sparql": "\nSELECT DISTINCT ?item ?itemLabel WHERE {\n  SERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],mul,en\". }\n  {\n    SELECT DISTINCT ?item WHERE {\n      ?item p:P570 ?statement.\n      ?statement psv:P570 ?value.\n      ?value wikibase:timePrecision ?precision.\n      hint:Prior hint:rangeSafe \"true\"^^xsd:boolean.\n      FILTER(?precision >= 9)\n      ?value wikibase:timeValue ?death.\n      hint:Prior hint:rangeSafe \"true\"^^xsd:boolean.\n      FILTER(?death < \"+1999-06-25T00:00:00Z\"^^xsd:dateTime)\n        ?item p:P27 ?statement1.\n      ?statement1 (ps:P27/(wdt:P279*)) wd:Q211.\n    }\n    LIMIT 100\n  }\n}",
    "entities": [
      {
        "qid": "Q108011",
        "label": "Harald Kalniņš"
      },
      {
        "qid": "Q1069984",
        "label": "Kārlis Skalbe"
      }, 
       ...
    ],
    "question": "Why couldn't person A have attended El Último Tour Del Mundo?",
    "answer": "Because person A died before 2002, while El Último Tour Del Mundo occurred in 2002.",
    "grounded_qa": [
      {
        "qid": "Q108011",
        "label": "Harald Kalniņš",
        "grounded_question": "Why couldn't Harald Kalniņš have attended El Último Tour Del Mundo?",
        "grounded_answer": "Because Harald Kalniņš died before 2002, while El Último Tour Del Mundo occurred in 2002."
      },
      {
        "qid": "Q1069984",
        "label": "Kārlis Skalbe",
        "grounded_question": "Why couldn't Kārlis Skalbe have attended El Último Tour Del Mundo?",
        "grounded_answer": "Because Kārlis Skalbe died before 2002, while El Último Tour Del Mundo occurred in 2002."
      },  
      ...
    ]

  }


  }
```





## Baseline Methods
To run the LLM inference on the dataset, use the following command.
```
python baselines/evaluation.py
```
 
To run the KGQA method on the dataset, use the following command.
```
python -m baselines.<KBBinder|KGR> datafiles/ --dataset_name commonwhy
```
