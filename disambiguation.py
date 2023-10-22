from sklearn.metrics.pairwise import cosine_similarity


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

def embedding_based_only(df_sintatic, results):
    for acronym in df_sintatic["Acronym"].unique():
        df_filtered = df_sintatic[df_sintatic["Acronym"] == acronym]
        expansion_unique = df_filtered["Expansion"].unique()

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