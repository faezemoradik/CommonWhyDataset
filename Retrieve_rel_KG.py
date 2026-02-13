import json
import requests
import time

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

HEADERS = {
    "User-Agent": "EntityKGExtractor/1.0 (your_email@example.com)"
}


def get_wikidata_facts(qid):
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "format": "json",
        "props": "claims"
    }

    response = requests.get(
        WIKIDATA_API,
        params=params,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()
    data = response.json()

    facts = []
    entity_data = data.get("entities", {}).get(qid, {})
    claims = entity_data.get("claims", {})

    for prop_id, statements in claims.items():
        for statement in statements:
            mainsnak = statement.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue")

            if datavalue:
                value = datavalue.get("value")

                if isinstance(value, dict):
                    if "id" in value:
                        value = value["id"]
                    elif "time" in value:
                        value = value["time"]
                    else:
                        value = str(value)

                facts.append({
                    "property": prop_id,
                    "value": value
                })

    return facts


def main():
    input_file = "inference_rule_entities_with_grounded_qa.json"
    output_results = []

    # 1. Load JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Iterate over each key
    for key, value in data.items():
        entities = value.get("entities", [])

        for entity in entities:
            time.sleep(2)  # required to avoid hitting API rate limits
            qid = entity.get("qid")
            label = entity.get("label")

            if not qid:
                continue

            print(f"Retrieving facts for {qid} ({label})...")

            try:
                facts = get_wikidata_facts(qid)

                output_results.append({
                    "qid": qid,
                    "label": label,
                    "Relevant KG": facts
                })

                # Be polite to API
                time.sleep(0.1)

            except Exception as e:
                print(f"Error retrieving {qid}: {e}")

    # Optional: save results
    with open("wikidata_facts_output.json", "w", encoding="utf-8") as f:
        json.dump(output_results, f, indent=2, ensure_ascii=False)

    print("Done. Results saved to wikidata_facts_output.json")


if __name__ == "__main__":
    main()
