import os, requests, json
import matplotlib.pyplot as plt




WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

HEADERS = {
    "User-Agent": "CommonWhy/1.0 (research use)"
}

def plot_popularity_distribution(popularities, bins=30, log_scale=False):
    """
    Plot the distribution of entity popularities using a seaborn theme.

    Parameters
    ----------
    popularities : list[int]
        List of popularity values (e.g., triple counts).
    bins : int
        Number of histogram bins.
    log_scale : bool
        Whether to use a logarithmic x-axis.
    """
    if not popularities:
        raise ValueError("Popularity list is empty.")

    # Set seaborn theme
    plt.style.use("seaborn-v0_8")

    plt.figure()
    plt.hist(popularities, bins=bins)
    plt.xlabel("Number of Wikidata Triples")
    plt.ylabel("Frequency")
    plt.title("Distribution of Entity Popularities")

    if log_scale:
        plt.xscale("log")

    plt.tight_layout()
    plt.savefig("popularity_distribution.png")

def count_wikidata_triples(qid: str) -> int:
    if not qid.startswith("Q"):
        raise ValueError(f"Invalid QID: {qid}")

    query = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>

    SELECT (COUNT(*) AS ?count) WHERE {{
      wd:{qid} ?p ?o .
    }}
    """

    response = requests.get(
        WIKIDATA_SPARQL_URL,
        params={
            "query": query,
            "format": "json" 
        },
        headers=HEADERS,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Wikidata error {response.status_code}: {response.text[:300]}"
        )

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(
            f"Expected JSON but got:\n{response.text[:300]}"
        )

    return int(data["results"]["bindings"][0]["count"]["value"])



with open("inference_rule_entities_with_grounded_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)
popularities = []

i = 0
output_file = "inference_rule_entities_with_grounded_qa_and_popularity.json"
temp_file = output_file + ".tmp"
for rule, entry in data.items():
    i += 1
    # if i < 58:
    #     continue
    
    print(f"Processing rule {i}/{len(data)}")
    grounded_qas = entry.get("grounded_qa", [])
    for qa in grounded_qas:
        qid = qa["qid"]
        try:
            popularity = count_wikidata_triples(qid)
            popularities.append(popularity)
        except Exception as e:
            print(f"Error counting triples for {qid}: {e}")
            popularity = -1
        qa["popularity"] = popularity
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, output_file)
plot_popularity_distribution(popularities, bins=50, log_scale=True)
with open("inference_rule_entities_with_grounded_qa_and_popularity.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)


