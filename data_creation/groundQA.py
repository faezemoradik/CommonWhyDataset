import json
import re

INPUT_FILE = "inference_rule_entities_2.json"
OUTPUT_FILE = "inference_rule_entities_with_grounded_qa_2.json"

# Regex to find placeholders like "person A", "town B", "building C"
PLACEHOLDER_PATTERN = re.compile(
    r"\b(person|town|enterprise|building|device|product|object)\s+[A-Z]\b"
)

def ground_text(text: str, entity_label: str) -> str:
    """
    Replace the first placeholder (e.g., 'person A') with the entity label.
    """
    return PLACEHOLDER_PATTERN.sub(entity_label, text, count=1)

def main():
    question_answer_pairs = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for rule, entry in data.items():
        question = entry.get("question")
        answer = entry.get("answer")
        entities = entry.get("entities", [])

        if not question or not answer:
            print(f"Skipping rule without QA: {rule}")
            continue

        grounded_qas = []

        for ent in entities:
            label = ent["label"]
            qid = ent["qid"]


            grounded_question = ground_text(question, label)
            grounded_answer = ground_text(answer, label)
            if qid != label:
                grounded_qas.append({
                "qid": qid,
                "label": label,
                "grounded_question": grounded_question,
                "grounded_answer": grounded_answer
                })
                question_answer_pairs.append({
                    "question": grounded_question,
                    "answer": grounded_answer
                })

        entry["grounded_qa"] = grounded_qas

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open("grounded_qas.json", "w", encoding="utf-8") as f:
        json.dump(question_answer_pairs, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
