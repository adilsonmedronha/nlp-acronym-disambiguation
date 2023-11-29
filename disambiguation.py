from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from create_embeddings import get_sentence_embedding
import re
import pandas as pd



def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) / len(set1.union(set2))

def jaccard_based(df_sintatic, results):
    for acronym in df_sintatic["Acronym"].unique():
        df_filtered = df_sintatic[df_sintatic["Acronym"] == acronym]
        expansion_unique = df_filtered["Expansion"].unique()

        for _, row in df_filtered.iterrows():
            sentence = row["Sentence"]
            label = row["Expansion"]
            predictions = {
                resolution: jaccard_similarity(sentence.split(), resolution.split())
                for resolution in expansion_unique
            }
            argmax = max(predictions, key=predictions.get)
            
            results["Label"].append(label)
            results["Prediction"].append(argmax)
            results["True"].append(label == argmax)

    return results

def get_acronym(sentence: str) -> str:
    pattern = r'\b[A-Z]+\d?[&\/]?[A-Z]+\b|\b[A-Z]{2,}(?:s\b)?'
    return re.findall(pattern, sentence)[0]

@lru_cache()
def get_acronyms_dataframe():
    return pd.read_pickle("data/not_ambiguous.csv")

def is_ambiguous(acronym: str) -> bool:
    df_acronyms = get_acronyms_dataframe()
    return len(df_acronyms[df_acronyms['Acronym'] == acronym]) > 1

def expanded_acronym(acronym):
    df_acronyms = get_acronyms_dataframe()

    df_acronyms[df_acronyms['Acronym'] == acronym]['Expansion']
    return df_acronyms[df_acronyms['Acronym'] == acronym]['Expansion']

def expand_sentence(sentence, acronym):
    df_acronyms = get_acronyms_dataframe()
    expansions_for_acronym = df_acronyms[df_acronyms['Acronym'] == acronym]['Expansion'].unique()
    return re.sub(re.escape(acronym), expansions_for_acronym[0], sentence)

def single_inference(sentence):
    acr_target = get_acronym(sentence)
    if not is_ambiguous(acr_target):
        return expand_sentence(sentence, acr_target) 
    # else:
    #     sentence_emb = get_sentence_embedding(sentence)
    #     acr_expanded_pred = classifier(sentence_emb, our_train)
    #     return expand_sentence(sentence, acr_expanded_pred) 
        
def embedding_based_only(df_sintatic, results):
    for acronym in df_sintatic["Acronym"].unique():
        if not is_ambiguous(acronym):
            expanded = expanded_acronym(acronym)
            print(expanded)
            results["Label"].append(expanded)
            results["Prediction"].append(expanded)
            results["True"].append(True)
        else:
            df_filtered = df_sintatic[df_sintatic["Acronym"] == acronym]
            expansion_unique = df_filtered["Expansion"].unique()
            # set of expanded embeddings of an specific acronym 
            unique_embeddings = {
                expansion: df_filtered[df_filtered["Expansion"] == expansion].iloc[0]["Expansion_Emb"]
                for expansion in expansion_unique
            }
            print(acronym)

            for _, row in df_filtered.iterrows():
                sentence_emb = row["Sentence_Emb"]
                label = row["Expansion"]
                predictions = {
                    resolution: cosine_similarity(sentence_emb, emb)[0][0]
                    for resolution, emb in unique_embeddings.items()
                }
                argmax = max(predictions, key=predictions.get)
                results["Label"].append(label)
                results["Prediction"].append(argmax)
                results["True"].append(label == argmax)

    return results