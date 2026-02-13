import json
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
import os
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
MAX_ENTITIES = 20
OUTPUT_FILE = "inference_rule_entities_2.json"
HEADERS = {
    "Accept": "application/sparql+json",
    "User-Agent": "Commonsense-Reasoning-Generator/1.0"
}
with open("rules2sparql_with_qa_2.json", "r", encoding="utf-8") as f:
    # open the csv file with \n as line endings and add to a list
    inference_rules_data = json.load(f)

# -----------------------------
# Execute SPARQL
# -----------------------------

def run_sparql(query):
    response = requests.get(
        WIKIDATA_SPARQL_URL,
        headers=HEADERS,
        params={"query": query, "format": "json"}
    )
    response.raise_for_status()
    return response.json()["results"]["bindings"]

# -----------------------------
# Extract QIDs
# -----------------------------

def extract_entities(bindings):
    entities = []
    for b in bindings:
        uri = b["item"]["value"]
        qid = uri.split("/")[-1]
        label = b["itemLabel"]["value"]
        entities.append({"qid": qid, "label": label})
    return entities

# -----------------------------
# Main pipeline
# -----------------------------

def main():
    output = {}

    for rule in inference_rules_data:

        output[rule['rule']] = {"sparql": rule['sparql']}

        print(f"Generating SPARQL for {rule['rule']}...")

        sparql = rule["sparql"]


        try:
            bindings = run_sparql(sparql)
            entities = extract_entities(bindings)
        except Exception as e:
            print(f"Error querying Wikidata for {rule['rule']}: {e}")
            entities = []

        output[rule['rule']]['entities'] = entities
        output[rule['rule']]['question'] = rule.get('question', '')
        output[rule['rule']]['answer'] = rule.get('answer', '')

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {OUTPUT_FILE}")

# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":
    main()
