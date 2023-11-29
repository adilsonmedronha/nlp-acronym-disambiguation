import matplotlib.pyplot as plt
import seaborn as sns
import random
from tabulate import tabulate


def plot_cfm(confusion, results_df):
    plt.figure(figsize=(20, 14))
    sns.heatmap(confusion, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                xticklabels=results_df["Label"].unique(), 
                yticklabels=results_df["Label"].unique())

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_bar(precision_df):
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Class', y='Precision', data=precision_df, palette='Blues')
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.title('Precision per Class')
    plt.xticks(rotation=90)
    plt.show()


def pprint(df_not_ambiguous, num_pairs):
    print(tabulate([['Sentence', 'Expansion']], headers='firstrow'))
    for _ in range(num_pairs):
        random_index = random.randint(0, len(df_not_ambiguous) - 1)
        random_sentence = df_not_ambiguous['sentence'].iloc[random_index]
        random_expansion = df_not_ambiguous['expansions'].iloc[random_index]
        print(tabulate([[random_sentence, random_expansion]], tablefmt='plain'))