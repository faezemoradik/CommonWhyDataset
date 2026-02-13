import json
import time
import os
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv
import google.generativeai as genai

import torch
from bert_score import score as bertscore
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import sacrebleu

from transformers import pipeline
from openai import OpenAI


# -------------------------
# Configuration
# -------------------------

DATASET_PATH = "HeadRandomSamples.json"
head_or_tail = "head" if "head" in DATASET_PATH.lower() else "tail"

# GEN_MODEL = "gpt-5.1"
# GEN_MODEL = "gemini-2.5-flash"
# GEN_MODEL = "groq-qwen3-32b" 
# GEN_MODEL = "deepseek-reasoner"
GEN_MODEL = "o3"

OUTPUT_PATH = f"{GEN_MODEL}_evaluation_results_{head_or_tail}.json"

JUDGE_MODEL = "gpt-4o"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Load models
# -------------------------

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

judge_client = OpenAI(api_key=openai_api_key)

if "gpt" in GEN_MODEL or "o3" in GEN_MODEL:
    client = OpenAI(api_key=openai_api_key)
elif "deepseek" in GEN_MODEL:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
else:
    client = Groq(api_key=GROQ_API_KEY)
    
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)






from transformers import pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    framework="pt"
)

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# -------------------------
# Helper functions
# -------------------------

def generate_answer(question, client, genai):
    messages=[
                {"role": "system", "content": "Answer the question concisely and factually."},
                {"role": "user", "content": question}
            ]
    if "gpt" in GEN_MODEL or "o3" in GEN_MODEL:
        response = client.chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            # temperature=0.0
        )
        return response.choices[0].message.content.strip()

    elif "gemini" in GEN_MODEL:
            model = genai.GenerativeModel(
                model_name=GEN_MODEL,
                system_instruction="Answer the question concisely and factually."
            )
            response = model.generate_content(question)
            return response.text
    elif "groq" in GEN_MODEL.lower():
            completion = client.chat.completions.create(
                model= "qwen/qwen3-32b", messages=messages)
            return completion.choices[0].message.content
    
    elif "deepseek" in GEN_MODEL.lower():
            response = client.chat.completions.create(
                model=GEN_MODEL,
                messages=messages,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()

def bert_scores(pred, ref):
    P, R, F1 = bertscore([pred], [ref], lang="en", rescale_with_baseline=True)
    return P.item(), R.item(), F1.item()


def rouge_l(pred, ref):
    return rouge.score(ref, pred)["rougeL"].fmeasure


def bleu(pred, ref):
    return sacrebleu.sentence_bleu(pred, [ref]).score


def meteor(pred, ref):
    return meteor_score([ref.split()], pred.split())


def nli_entailment(pred, ref, classifier):
    result = classifier(pred, ref)['scores'][0]

    verdict = 1 if result > 0.5 else 0
    return verdict


def gpt_judge(question, pred, ref):
    prompt = f"""
    Question:
    {question}

    Gold answer:
    {ref}

    Model answer:
    {pred}

    You are a knowledgeable and precise judge. Evaluate the model answer against the gold answer for the given question. If the model answer is logically the same as the gold answer, even if phrased differently, return 1. Otherwise, return 0.
    Respond with ONLY 1 (correct) or 0 (incorrect).
    """

    response = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    verdict = response.choices[0].message.content.strip()
    return 1 if verdict == "1" else 0



    # -------------------------
    # Main evaluation loop
    # -------------------------

with open(DATASET_PATH) as f:
    data = json.load(f)


if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r") as f:
        results = json.load(f)
    print(f"Resuming from {len(results)} saved results")
    bert_scores_list_precision = [res["bertscore_precision"] for res in results]; bert_scores_list_recall = [res["bertscore_recall"] for res in results]; bert_scores_list_f1 = [res["bertscore_f1"] for res in results]
    rouge_l_list = [res["rougeL"] for res in results]; bleu_list = [res["bleu"] for res in results]; meteor_list = [res["meteor"] for res in results]; gpt_judge_list = [res["gpt_4o_judge"] for res in results]
else:
    results = []
    bert_scores_list_precision = []; bert_scores_list_recall = []; bert_scores_list_f1 = []
    rouge_l_list = []; bleu_list = []; meteor_list = []; nli_list = []; gpt_judge_list = []

for item in tqdm(data[len(results):1000]):
    question = item["question"]
    gold = item["answer"]
    pred = generate_answer(question, client, genai)
    bp, br, bf1 = bert_scores(pred, gold)
    rouge_l_f = rouge_l(pred, gold)
    bleu_s = bleu(pred, gold)
    meteor_s = meteor(pred, gold)
    # nli_s = nli_entailment(pred, gold, classifier)
    judge = gpt_judge(question, pred, gold)

    bert_scores_list_precision.append(bp); bert_scores_list_recall.append(br); bert_scores_list_f1.append(bf1)
    rouge_l_list.append(rouge_l_f); bleu_list.append(bleu_s); meteor_list.append(meteor_s); gpt_judge_list.append(judge)
    # nli_list.append(nli_s); 
    
    
    results.append({
        "question": question,
        "gold_answer": gold,
        "model_answer": pred,
        "bertscore_precision": bp,
        "bertscore_recall": br,
        "bertscore_f1": bf1,
        "rougeL": rouge_l_f,
        "bleu": bleu_s,
        "meteor": meteor_s,
        # "nli_entailment": nli_s,
        "gpt_4o_judge": judge,
        "average_bertscore_precision": sum(bert_scores_list_precision)/len(bert_scores_list_precision) if bert_scores_list_precision else 0,
        "average_bertscore_recall": sum(bert_scores_list_recall)/len(bert_scores_list_recall) if bert_scores_list_recall else 0,
        "average_bertscore_f1": sum(bert_scores_list_f1)/len(bert_scores_list_f1) if bert_scores_list_f1 else 0,
        "average_rougeL": sum(rouge_l_list)/len(rouge_l_list) if rouge_l_list else 0,
        "average_bleu": sum(bleu_list)/len(bleu_list) if bleu_list else 0,
        "average_meteor": sum(meteor_list)/len(meteor_list) if meteor_list else 0,
        # "average_nli_entailment": sum(nli_list)/len(nli_list) if nli_list else 0,
        "average_gpt_4o_judge": sum(gpt_judge_list)/len(gpt_judge_list) if gpt_judge_list else 0
    })

    print(sum(gpt_judge_list)/len(gpt_judge_list))
    # time.sleep(10)  # To avoid hitting rate limits

    # -------------------------
    # Save results
    # -------------------------
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

# with open(OUTPUT_PATH, "w") as f:
#     json.dump(results, f, indent=2)
print(f"Saved results to {OUTPUT_PATH}")


