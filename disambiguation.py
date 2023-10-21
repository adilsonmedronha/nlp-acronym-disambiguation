from sklearn.metrics.pairwise import cosine_similarity



def embedding_based_only(df_sintatic, results):
    for acronym in df_sintatic["Acronym"].unique():
        df_filtered = df_sintatic[df_sintatic["Acronym"] == acronym]
        resolution_unique = df_filtered["Resolution"].unique()

        unique_embeddings = {
            resolution: df_filtered[df_filtered["Resolution"] == resolution].iloc[0]["Resolution_Emb"]
            for resolution in resolution_unique
        }

        for _, row in df_filtered.iterrows():
            sentence_emb = row["Sentence_Emb"]
            label = row["Resolution"]
            predictions = {
                resolution: cosine_similarity(sentence_emb, emb)[0][0]
                for resolution, emb in unique_embeddings.items()
            }
            argmax = max(predictions, key=predictions.get)
            
            results["Label"].append(label)
            results["Prediction"].append(argmax)
            results["True"].append(label == argmax)

    return results