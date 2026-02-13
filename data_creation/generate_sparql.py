import json
output = []

with open("inference_rules2.csv", "r") as file:
    lines = file.readlines()

for line in lines:  # Skip header
    rule = line.strip().split("\n")[0]
    output.append({"rule": rule})

sparqls = ["""SELECT DISTINCT ?item ?itemLabel WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en". }
  {
    SELECT DISTINCT ?item WHERE {
      ?item p:P27 ?statement0.
      ?statement0 (ps:P27/(wdt:P279*)) wd:Q183.
      ?item p:P27 ?statement1.
      ?statement1 (ps:P27/(wdt:P279*)) wd:Q145.
    }
    LIMIT 100
  }
}"""
]

for i in range(len(output)):
    output[i]["sparql"] = sparqls[i]

with open("rules2sparql_2.json", "w", encoding="utf-8") as json_file:
    json.dump(output, json_file, ensure_ascii=False, indent=4)