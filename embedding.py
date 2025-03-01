import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def generate_embeddings(csv_files, model_name='all-MiniLM-L6-v2', output_dir=None, use_cleaned=True):
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    embeddings_dict = {}
    
    for csv_file in tqdm(csv_files, desc="generate embedding"):
        chapter_name = os.path.basename(csv_file).replace("processed_chapter_", "").replace(".csv", "")
        
        df = pd.read_csv(csv_file)
        
        text_column = 'cleaned_sentence' if use_cleaned and 'cleaned_sentence' in df.columns else 'sentence'
        
        if text_column not in df.columns:
            print(f"Error: {csv_file} doesn't have {text_column} column, skipped!")
            continue

        sentences = []
        for sent in df[text_column].tolist():
            if isinstance(sent, str):
                sentences.append(sent)
            else:
                sentences.append(str(sent) if sent is not None else "")

        if not sentences:
            print(f"Error: {csv_file} doesn't have sentences, skipped!")
            continue
        
        batch_size = 32 
        embeddings = model.encode(sentences, batch_size=batch_size, show_progress_bar=True)

        embeddings_dict[chapter_name] = embeddings
        
        if output_dir:
            np.save(os.path.join(output_dir, f"embeddings_{chapter_name}.npy"), embeddings)
            df['embedding'] = [json.dumps(emb.tolist()) for emb in embeddings]
            df.to_csv(os.path.join(output_dir, f"embedded_{chapter_name}.csv"), index=False)
    
    if output_dir:
        with open(os.path.join(output_dir, 'all_embeddings.pkl'), 'wb') as f:
            pickle.dump(embeddings_dict, f)
    
    print(f"Embedding generated. Total {len(embeddings_dict)} chapters")
    return embeddings_dict


def analyze_sentence_similarities(embeddings_dict, output_dir=None):
    similarity_results = {}
    
    for chapter, embeddings in embeddings_dict.items():
        sim_matrix = cosine_similarity(embeddings)
        
        avg_similarities = np.mean(sim_matrix, axis=1)
        
        most_similar_idx = np.argmax(avg_similarities)
        least_similar_idx = np.argmin(avg_similarities)

        similarity_results[chapter] = {
            'similarity_matrix': sim_matrix,
            'avg_similarities': avg_similarities,
            'most_similar_idx': int(most_similar_idx),
            'least_similar_idx': int(least_similar_idx),
            'most_similar_score': float(avg_similarities[most_similar_idx]),
            'least_similar_score': float(avg_similarities[least_similar_idx])
        }
        
        if output_dir:
            np.save(os.path.join(output_dir, f"similarity_matrix_{chapter}.npy"), sim_matrix)
            
            result_json = {
                'avg_similarities': avg_similarities.tolist(),
                'most_similar_idx': int(most_similar_idx),
                'least_similar_idx': int(least_similar_idx),
                'most_similar_score': float(avg_similarities[most_similar_idx]),
                'least_similar_score': float(avg_similarities[least_similar_idx])
            }
            
            with open(os.path.join(output_dir, f"similarity_analysis_{chapter}.json"), 'w') as f:
                json.dump(result_json, f, indent=2)
    
    print("Cosine similarity finished!")
    return similarity_results

def main_embedding(processed_dir, output_dir, model_name='all-MiniLM-L6-v2'):
    embedding_dir = os.path.join(output_dir, 'embeddings')
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
    
    csv_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.startswith('processed_') and f.endswith('.csv')]
    
    print("Generating embeddings...")
    embeddings_dict = generate_embeddings(csv_files, model_name, embedding_dir)

    print("Start analyzing similarity...")
    similarity_results = analyze_sentence_similarities(embeddings_dict, embedding_dir)

    print("\nSentence similarity samples:")
    for chapter, results in list(similarity_results.items())[:2]:  # only show two chapters
        print(f"Chapter {chapter}:")
        print(f"  Most similar sentences index: {results['most_similar_idx']}, score: {results['most_similar_score']:.4f}")
        print(f"  Least similar sentences index: {results['least_similar_idx']}, score: {results['least_similar_score']:.4f}")
    
    print(f"\nAll results saved at: {embedding_dir}")

if __name__ == "__main__":
    processed_dir = "data/processed_csvs"  
    output_dir = "embeddings" 
    model_name = 'all-MiniLM-L6-v2'  # https://www.sbert.net/docs/pretrained_models.html
    
    main_embedding(processed_dir, output_dir, model_name)