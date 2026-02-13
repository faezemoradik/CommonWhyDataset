import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math


def extract_popularity_values(dict_path):
    """
    Load a dictionary from dict_path and return a list of all
    popularity values found in value["grounded_qa"].
    """
    with open(dict_path, "r") as f:
        data = json.load(f)

    data_list = []
    global_popularity_list = []
    first_split = []
    second_split = []
    first_popularity_list = []
    second_popularity_list = []

    for rule, value in data.items():
        grounded_qa = value["grounded_qa"]
        popularity_list = []
        curr_data = []
        index = 0
        for qa_item in grounded_qa:
            curr_data.append({"question": qa_item["grounded_question"], 
                              "answer": qa_item["grounded_answer"],
                              "rule": rule,
                              "popularity": qa_item["popularity"] })
            
            popularity_list.append((qa_item["popularity"], index))
            index += 1

        popularity_list = sorted(popularity_list)
        length = len(popularity_list)
        split_point = length // 2
        for item in popularity_list[:split_point]:
            first_split.append(curr_data[item[1]])
            first_popularity_list.append(item)

        for item in popularity_list[split_point:2*split_point]:
            second_split.append(curr_data[item[1]])
            second_popularity_list.append(item)
           
        global_popularity_list += popularity_list
        data_list += curr_data

    

    return data_list, first_split, second_split, global_popularity_list, first_popularity_list, second_popularity_list


def plot_popularity_histogram(popularity_list, bins=30):
    popularity_values = [x for x, _ in popularity_list]
    max_pop = max(popularity_values)
    sns.set_theme(style="darkgrid", font_scale=1.4)  

    plt.figure()
    sns.histplot(
        popularity_values,
        bins=bins,
        color="tab:orange",
        edgecolor="white"
    )

    max_pop = 800

    plt.xlim((0, max_pop + 10) ) 
    plt.xlabel("Popularity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    # plt.yscale('log')
    plt.savefig("popularity_histogram.pdf", dpi=300)
    plt.show()


def plot_two_split_histogram(smaller_half, larger_half):

    sns.set_theme(style="darkgrid", font_scale=1.4)  

    print(f"Smaller half size: {len(smaller_half)}")
    print(f"Larger half size: {len(larger_half)}")
    print('total size:', len(smaller_half) + len(larger_half))

    smaller_half_pop = [x for x, _ in smaller_half]
    larger_half_pop = [x for x, _ in larger_half]

    max_pop = max( max(smaller_half_pop), max(larger_half_pop) )

    plt.figure(figsize=(10, 5))
    sns.histplot(
        larger_half_pop,
        bins=125,
        color="tab:blue",
        edgecolor="white",  label="Head", alpha=0.7
    )

    
    sns.histplot(
        smaller_half_pop,
        bins=30,
        color="tab:orange",
        edgecolor="white", label="Long-tail", alpha=0.7
    )

    max_pop = 800

    plt.xlim((0, max_pop + 10)) 
    plt.xlabel("Popularity")
    plt.ylabel("Frequency")
    # plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig("popularity_split_histogram.pdf", dpi=300)
    plt.show()

def plot_rule_distribution(first_split, second_split, title):
    ### counting the number of samples in HeadRandomSamples.json has belongs to each specific rule
    rule_counts_head = {}
    for item in second_split:
        rule = item["rule"]
        if rule not in rule_counts_head:
            rule_counts_head[rule] = 0
        rule_counts_head[rule] += 1

    rule_counts_tail = {}
    for item in first_split:
        rule = item["rule"]
        if rule not in rule_counts_tail:
            rule_counts_tail[rule] = 0
        rule_counts_tail[rule] += 1


    ### plot the distribution of rules in head and long-tail random samples as two bar charts side by side
    rules = sorted(set(rule_counts_head.keys()) | set(rule_counts_tail.keys()))
    head_counts = [rule_counts_head.get(rule, 0) for rule in rules]
    tail_counts = [rule_counts_tail.get(rule, 0) for rule in rules] 
    x = range(len(rules))       
    width = 0.35        
    plt.figure(figsize=(60, 10))
    plt.bar([i - width/2 for i in x], head_counts, width=width, label='Head', color='tab:blue')
    plt.bar([i + width/2 for i in x], tail_counts, width=width, label='Long-tail', color='tab:orange')
    plt.xlabel('Rules')
    plt.ylabel('Counts')
    # plt.title('Distribution of Rules in Head and Long-tail Random Samples')
    plt.xticks(ticks=x, labels=x, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(title + ".pdf", dpi=300)
    plt.show()





if __name__ == "__main__":
    random.seed(0)
    dict_path = "inference_rule_entities_with_grounded_qa_and_popularity.json"
    _, first_split, second_split, global_popularity_list, first_popularity_list, second_popularity_list = extract_popularity_values(dict_path)


    plot_popularity_histogram(global_popularity_list, bins=70)
    plot_two_split_histogram(first_popularity_list, second_popularity_list)


    with open("Longtail.json", "w") as f:
        json.dump(first_split, f, indent=4)            
    with open("Head.json", "w") as f:
        json.dump(second_split, f, indent=4)


    ### get 1000 random samples from the long_tail_data   
    lt_random_samples = random.sample(first_split, 1000)
    with open("LongTailRandomSamples.json", "w") as f:
        json.dump(lt_random_samples, f, indent=4)

    ### get 1000 random samples from the head_data   
    head_random_samples = random.sample(second_split, 1000)
    with open("HeadRandomSamples.json", "w") as f:
        json.dump(head_random_samples, f, indent=4)


    # plot_rule_distribution(first_split, second_split, "rule_distribution_full")
    # plot_rule_distribution(lt_random_samples, head_random_samples, "rule_distribution_random_samples")

