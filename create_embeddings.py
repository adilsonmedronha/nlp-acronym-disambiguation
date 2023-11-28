import pandas as pd
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser


def get_sentence_embedding(sentence):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentence)
    return embeddings

def return_ambiguous(df):
    unique_acronyms = df.groupby('Acronym')['Resolution'].nunique()
    ambiguous_acronyms = unique_acronyms[unique_acronyms > 1].index
    ambiguous_df = df[df['Acronym'].isin(ambiguous_acronyms)]
    ambiguous_df = ambiguous_df.sort_values(by='Resolution')
    return ambiguous_df

def main():
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='data/sintatic.csv', help='Path to the data')
    parser.add_argument('--n_workers', type=int, default=4, help="Number of workers to use in the parallelization")
    parser.add_argument('--output', type=str, default='data/sintatic_wemb.pkl', help='Path to the output')
    args = parser.parse_args()

    df_sintatic = pd.read_csv(args.data, sep='\t')
    ambiguous_df = return_ambiguous(df_sintatic)

    sentence_embeddings = []
    resolution_embeddings = []

    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        for sentence, resolution in zip(ambiguous_df["Sentence"], ambiguous_df["Resolution"]):
            sentence_embeddings.append([get_sentence_embedding(sentence)])
            resolution_embeddings.append([get_sentence_embedding(resolution)])

    ambiguous_df["Sentence_Emb"] = sentence_embeddings
    ambiguous_df["Resolution_Emb"] = resolution_embeddings

    ambiguous_df = ambiguous_df.sort_values(by='Resolution')
    ambiguous_df.to_pickle(args.output)

if __name__ == '__main__':
    main()
