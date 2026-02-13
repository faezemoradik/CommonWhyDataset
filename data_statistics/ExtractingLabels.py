import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np




def extract_labels(dict_path):
    """
    Load a dictionary from dict_path and return a dict with the same keys but list of labels as values.
    """
    with open(dict_path, "r") as f:
        data = json.load(f)

    newdata = {}
    counter = 0

    for rule, value in data.items():
        assigned_labels = gpt_labeler(rule)
        newdata[rule] = assigned_labels
        counter += 1
        print('-------- inference rule '+ str(counter)+' --------')



    return newdata


def gpt_labeler(rule):
    """
    Given a rule (string), return a list of labels assigned to it by GPT-4o.
    """

    prompt = f"""You are an expert in categorizing inference rules based on their content. 
    Based on the given statement, select all labels below that correspond to the knowledge or skills required to verify it. A statement may have multiple labels, so choose all that apply:

    Labels:
    - TemporalReasoning
    - Biological
    - SetInclusion
    - Physical
    - Technological
    - Foods
    - EntityComparison
    - Geographical
    - Political
    - Education
    - Entertainment
    - Cultural
    - LocationComparison
    - Sports
    - Definition
    - Religous
    - Music
    - Literature
    - Historical
    - Medical
    - Professions
    - Economic
    - Language
    - NumberComparison

    Statement:
    {rule}

    Please provide your answer as a list of labels. For example: ["Label1", "Label2"]
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    JUDGE_MODEL = "gpt-4o"

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    labels_str = response.choices[0].message.content.strip()
    ## extract a list of labels from the response string
    try:
        labels = json.loads(labels_str)
        if isinstance(labels, list):
            return labels
        else:
            print(f"Unexpected format for labels: {labels_str}")
            return []
    except json.JSONDecodeError:
        print(f"Failed to parse labels: {labels_str}")
        return []


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def Bar_Plot_for_Labels_percentage(LabelDict):

    sns.set_theme(style="darkgrid", font_scale=1.35)

    List_of_Labels = [
        "TemporalReasoning", "Biological", "SetInclusion", "Physical", "Technological", 
        "Foods", "EntityComparison", "Geographical", "Political", "Education", "Entertainment",
        "Cultural", "LocationComparison", "Sports", "Definition", "Religous", "Music", "Literature",
        "Historical", "Medical", "Professions", "Economic", "Language", "NumberComparison"
    ]

    # =========================
    # 1) Prepare data
    # =========================
    labels = [l for l in List_of_Labels if l in LabelDict]
    percents = np.array([LabelDict[l] for l in labels])

    sorted_idx = np.argsort(-percents)
    labels = np.array(labels)[sorted_idx]
    percents = percents[sorted_idx]

    mid = len(labels) // 2
    labels_top, percents_top = labels[:mid], percents[:mid]
    labels_bottom, percents_bottom = labels[mid:], percents[mid:]

    # =========================
    # 2) Color mapping
    # =========================
    reasoning_labels = {
        "TemporalReasoning", "SetInclusion", "EntityComparison",
        "LocationComparison", "NumberComparison", "Definition"
    }

    colors_top = ["hotpink" if l in reasoning_labels else "dodgerblue"
                  for l in labels_top]
    colors_bottom = ["hotpink" if l in reasoning_labels else "dodgerblue"
                     for l in labels_bottom]

    label_map = {
        "TemporalReasoning": "Temporal\nReasoning",
        "SetInclusion": "Set\nInclusion",
        "EntityComparison": "Entity\nComparison",
        "LocationComparison": "Location\nComparison",
        "NumberComparison": "Number\nComparison",
        "Religous": "Religion",
        "Language": " Language",
        "Geographical": "Geographical",
    }

    labels_top_disp = [label_map.get(l, l) for l in labels_top]
    labels_bottom_disp = [label_map.get(l, l) for l in labels_bottom]

    # =========================
    # 3) Plot
    # =========================
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    # ---- Top subplot ----
    sns.barplot(
        x=labels_top_disp,
        y=percents_top,
        palette=colors_top,
        ax=axes[0]
    )

    axes[0].set_ylabel("Percentage (%)", fontsize=20)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=0, labelsize=19)

    for i, v in enumerate(percents_top):
        axes[0].text(
            i, v + 0.5,
            f"{v:.1f}%",
            ha='center',
            fontsize=18,
            fontweight='bold'
        )

    # ---- Bottom subplot ----
    sns.barplot(
        x=labels_bottom_disp,
        y=percents_bottom,
        palette=colors_bottom,
        ax=axes[1]
    )

    axes[1].set_ylabel("Percentage (%)", fontsize=20)
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=0, labelsize=19)

    for i, v in enumerate(percents_bottom):
        axes[1].text(
            i, v + 0.1,
            f"{v:.1f}%",
            ha='center',
            fontsize=18,
            fontweight='bold'
        )

    # =========================
    # 4) Legend INSIDE top subplot
    # =========================
    pink_patch = mpatches.Patch(
        color='hotpink',
        label='Domain-independent skills'
    )
    blue_patch = mpatches.Patch(
        color='dodgerblue',
        label='Domain-dependent skills'
    )

    axes[0].legend(
        handles=[pink_patch, blue_patch],
        loc='upper right',
        frameon=True,
        fontsize=20
    )

    plt.tight_layout()
    plt.savefig("label_distribution_bar_plot.pdf", dpi=300)
    plt.show()




if __name__ == "__main__":

    dict_path = "inference_rule_entities_with_grounded_qa_and_popularity.json"
    old_data = json.load(open(dict_path, "r"))
    # labled_data = extract_labels(dict_path)
    # with open("LabeledRules.json", "w") as f:
    #     json.dump(labled_data, f, indent=4)   

    with open("LabeledRules.json", "r") as f:
        labled_data = json.load(f)

    LabelDict = {}    
    total_qas = 0   

    for key, value in labled_data.items():
        number_of_qas = len(old_data[key]["grounded_qa"])
        total_qas += number_of_qas
        for elem in value:
            if elem in LabelDict.keys():
                LabelDict[elem] += number_of_qas
            else:
                LabelDict[elem] = number_of_qas


    for key in LabelDict.keys():
        LabelDict[key] = (LabelDict[key] / total_qas) * 100


    Bar_Plot_for_Labels_percentage(LabelDict)


        
            
