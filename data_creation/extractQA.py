import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)



INPUT_FILE = "rules2sparql.json"
OUTPUT_FILE = "rules2sparql_with_qa.json"

SYSTEM_PROMPT = (
    "You convert logical rules into a question–answer pair. The question must begin with \"Why\" and the answer must begin with \"Because\".\n\n"
    "You MUST output ONLY valid JSON with exactly two keys:\n"
    "  - \"question\"\n"
    "  - \"answer\"\n\n"
    "No extra text, no markdown, no explanations.\n\n"
    "Closely follow the style and structure shown in the example."
)

USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE = """
EXAMPLE

Rule:
"If a person A died before 2002, then person A could not have attended El Último Tour Del Mundo."

Output:
{{
  "question": "Why couldn't person A have attended El Último Tour Del Mundo?",
  "answer": "Because person A died before 2002, while El Último Tour Del Mundo occurred in 2002."
}}

---

NOW CONVERT THIS RULE

Rule:
"{rule}"

Output (JSON only):
{{
  "question": "...",
  "answer": "..."
}}
"""

def generate_qa(rule: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(rule=rule)
            },
        ],
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content.strip()

    try:
        qa = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model output was not valid JSON:\n{raw_text}"
        ) from e

    if not isinstance(qa, dict):
        raise ValueError("Parsed output is not a JSON object.")
    if "question" not in qa or "answer" not in qa:
        raise ValueError(f"Missing keys in model output: {qa}")

    return qa


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, entry in enumerate(data):
        if "rule" not in entry:
            continue

        qa = generate_qa(entry["rule"])
        entry["question"] = qa["question"]
        entry["answer"] = qa["answer"]
        print(f"Processed {i + 1}/{len(data)}: {entry['rule']}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
