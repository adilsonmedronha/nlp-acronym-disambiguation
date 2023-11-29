from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from create_embeddings import get_sentence_embedding
from transformers import BertForSequenceClassification
import re
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import torch


CLASS_MAPPING = {
    0: 'Accounts Receivable',
    1: 'Annual Review',
    2: 'Artificial Intelligence',
    3: 'Chartered Accountant',
    4: 'Cost Per Mille',
    5: 'Cost Per Million',
    6: 'Public Relations',
    7: 'Small and Medium-sized Enterprises',
    8: 'Direct Deposit'
}
TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased')
LOADED_MODEL = BertForSequenceClassification.from_pretrained('model_weight')

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
    return pd.read_csv("data/acronyms_only.csv")

@lru_cache()
def get_dataset_dataframe():
    return pd.read_pickle("data/synthetic_wemb_MiniLM_L6_v2.pkl")

def is_ambiguous(acronym: str) -> bool:
    df_acronyms = get_acronyms_dataframe()

    return len(df_acronyms[df_acronyms['Acronym'] == acronym]['Expansion'].unique()) > 1

def expanded_acronym(acronym):
    df_acronyms = get_acronyms_dataframe()
    return df_acronyms[df_acronyms['Acronym'] == acronym]['Expansion'].values[0]

def expand_sentence(sentence, acronym):
    df_acronyms = get_acronyms_dataframe()
    expansions_for_acronym = df_acronyms[df_acronyms['Acronym'] == acronym]['Expansion'].unique()
    expanded_acr = expansions_for_acronym[0] if len(expansions_for_acronym) > 0 else expansions_for_acronym
    return re.sub(re.escape(acronym), f" >>[{expanded_acr}]<< ", sentence)


def single_inference(sentence):
    acr_target = get_acronym(sentence)
    if not is_ambiguous(acr_target):
        return expand_sentence(sentence, acr_target)
    elif acr_target in {'AI', 'AR', 'CA', 'CPM', 'DD', 'PR', 'SME'}:
        tokenized_input = TOKENIZER(sentence, padding=True, truncation=True, return_tensors='pt')
        tokenized_input = {key: value for key, value in tokenized_input.items()}
        with torch.no_grad():
            outputs = LOADED_MODEL(**tokenized_input)
        predicted_labels = torch.argmax(outputs.logits, dim=1)
        predicted_expansion = CLASS_MAPPING[predicted_labels.item()]
        return expand_sentence(sentence, predicted_expansion)
    else:    
        sentence_emb = np.array(get_sentence_embedding(sentence))
        our_dataset = get_dataset_dataframe()
        unique_embeddings = {}
        df_filtered = our_dataset[our_dataset["Acronym"] == acr_target]
        for expansion in df_filtered["Expansion"].unique():
            expansion_emb = np.array(df_filtered[df_filtered["Expansion"] == expansion].iloc[0]["Expansion_Emb"])
            unique_embeddings[expansion] = expansion_emb

        predictions = {
            expansion: cosine_similarity(sentence_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
            for expansion, emb in unique_embeddings.items()
        }
        acr_expanded_pred = max(predictions, key=predictions.get)
        print(acr_expanded_pred)
        return expand_sentence(sentence, acr_expanded_pred)
        

def embedding_based_only(df_sintatic, results):
    for acronym in df_sintatic["Acronym"].unique():
        df_filtered = df_sintatic[df_sintatic["Acronym"] == acronym]
        expansion_unique = df_filtered["Expansion"].unique()
        # set of expanded embeddings of an specific acronym 
        unique_embeddings = {
            expansion: df_filtered[df_filtered["Expansion"] == expansion].iloc[0]["Expansion_Emb"]
            for expansion in expansion_unique
        }
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