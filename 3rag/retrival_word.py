import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import ast



class PrioritySentenceRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.df = None
        self.embeddings = None
        self.embeddings_computed = False
        

    def load_csv(self, csv_path, embedding_col=None, sentence_col='sentence'):
        self.df = pd.read_csv(csv_path)
        print(f"Loaded CSV file with {len(self.df)} records")
        
        self.df[sentence_col] = self.df[sentence_col].astype(str)
        
        if embedding_col and embedding_col in self.df.columns:
            try:
                sample = self.df[embedding_col].iloc[0]
                if isinstance(sample, str):
                    print("Parsing embeddings from CSV...")
                    self.embeddings = []
                    for emb_str in self.df[embedding_col]:
                        try:
                            emb = ast.literal_eval(emb_str)
                            self.embeddings.append(emb)
                        except:
                            try:
                                emb = np.fromstring(emb_str.strip('[]'), sep=' ')
                                self.embeddings.append(emb)
                            except:
                                print(f"Could not parse embeddings, will recompute if needed")
                                self.embeddings = None
                                break
                    
                    if self.embeddings:
                        self.embeddings = torch.tensor(self.embeddings)
                        self.embeddings_computed = True
                        print(f"Parsed embeddings, shape: {self.embeddings.shape}")
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                self.embeddings = None
    

    def compute_embeddings(self, sentence_col='sentence'):
        if not self.embeddings_computed:
            print("Computing sentence embeddings...")
            sentences = self.df[sentence_col].tolist()
            with torch.no_grad():
                self.embeddings = self.model.encode(sentences, convert_to_tensor=True)
                # Ensure the embeddings are on CPU
                if self.embeddings.device.type != 'cpu':
                    self.embeddings = self.embeddings.cpu()
            self.embeddings_computed = True
            print(f"Computed embeddings for {len(sentences)} sentences")
    

    def accurate_retrieve(self, keywords, sentence_col='sentence', case_sensitive=False):
        if self.df is None:
            return "Please load a CSV file first"
        
        filtered_df = self.df.copy()
        
        for keyword in keywords:
            if case_sensitive:
                filtered_df = filtered_df[filtered_df[sentence_col].str.contains(keyword, regex=False)]
            else:
                keyword_lower = keyword.lower()
                filtered_df = filtered_df[filtered_df[sentence_col].str.lower().str.contains(keyword_lower, regex=False)]

        results = filtered_df.to_dict('records')
        
        for result in results:
            sentence = result[sentence_col]
            matches = {}
            
            for keyword in keywords:
                if case_sensitive:
                    start_pos = sentence.find(keyword)
                else:
                    start_pos = sentence.lower().find(keyword.lower())
                
                if start_pos >= 0:
                    matches[keyword] = start_pos
            
            result['keyword_positions'] = matches
            result['match_type'] = 'accurate'  
        
        return results
    

    def semantic_retrieve(self, keywords, sentence_col='sentence', threshold=0.5, top_k=5):
        if self.df is None:
            return "Please load a CSV file first"
        
        self.compute_embeddings(sentence_col)

        query = " ".join(keywords)
        with torch.no_grad():
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            if query_embedding.device.type != 'cpu':
                query_embedding = query_embedding.cpu()

        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        actual_top_k = min(top_k, len(cos_scores))
        try:
            if not isinstance(cos_scores, torch.Tensor):
                cos_scores = torch.tensor(cos_scores)
            
            # Debug info
            print(f"Cos scores shape: {cos_scores.shape}, attempting to get top {actual_top_k} scores")

            top_scores, top_indices = torch.topk(cos_scores, k=actual_top_k)
            top_scores = top_scores.cpu().numpy().tolist()
            top_indices = top_indices.cpu().numpy().tolist()
            
            results = []
            for score, idx in zip(top_scores, top_indices):
                idx = int(idx)
                
                if score >= threshold and 0 <= idx < len(self.df):
                    row_dict = self.df.iloc[idx].to_dict()
                    row_dict['similarity'] = float(score)
                    
                    sentence = str(row_dict[sentence_col]).lower()
                    contained_keywords = [kw for kw in keywords if kw.lower() in sentence]
                    
                    row_dict['keywords_contained'] = contained_keywords
                    row_dict['match_type'] = 'semantic'  # Mark as semantic match
                    
                    results.append(row_dict)
            
            return results
        
        except Exception as e:
            print(f"Error during topk operation: {e}")
            print("Falling back to manual sorting for top-k selection...")

            scores_np = cos_scores.cpu().numpy()
            sorted_indices = np.argsort(-scores_np)
            
            results = []
            for i in range(min(actual_top_k, len(sorted_indices))):
                idx = int(sorted_indices[i])
                score = float(scores_np[idx])
                
                if score >= threshold and 0 <= idx < len(self.df):
                    row_dict = self.df.iloc[idx].to_dict()
                    row_dict['similarity'] = score
                    sentence = str(row_dict[sentence_col]).lower()
                    contained_keywords = [kw for kw in keywords if kw.lower() in sentence]
                    
                    row_dict['keywords_contained'] = contained_keywords
                    row_dict['match_type'] = 'semantic'  # Mark as semantic match
                    
                    results.append(row_dict)
            
            return results
    
    def retrieve(self, keywords, sentence_col='sentence', case_sensitive=False, 
                 fallback_to_semantic=True, semantic_threshold=0.5, top_k=5):
        if self.df is None:
            return "Please load a CSV file first"

        accurate_results = self.accurate_retrieve(keywords, sentence_col, case_sensitive)
        
        if accurate_results and len(accurate_results) > 0:
            print(f"Found {len(accurate_results)} results with accurate retrieval")
            return accurate_results

        if fallback_to_semantic:
            print("No exact matches found, trying semantic search...")
            try:
                semantic_results = self.semantic_retrieve(
                    keywords, sentence_col, semantic_threshold, top_k)
                
                if semantic_results and len(semantic_results) > 0:
                    print(f"Found {len(semantic_results)} results with semantic search")
                    return semantic_results
            except Exception as e:
                print(f"Error during semantic search: {e}")
                print(f"Error details: {str(e)}")
                # Print the current device information for debugging
                if hasattr(self, 'embeddings') and self.embeddings is not None:
                    print(f"Embeddings device: {self.embeddings.device}")
        
        
        print("No results found")
        return []


if __name__ == "__main__":
    retriever = PrioritySentenceRetriever()

    retriever.load_csv("embeddings/embedded_all_chapters.csv", embedding_col="embedding", sentence_col="sentence")
    
    keywords = ["Moria", "Gandalf", "the Ring", "Sam"]
    results = retriever.retrieve(keywords, sentence_col="sentence")
    
    if results and len(results) > 0:
        print("\nRetrieval Results:")
        
        for i, result in enumerate(results, 1):
            match_type = result.get('match_type', 'unknown')
            
            if match_type == 'accurate':
                print(f"{i}. [ACCURATE MATCH] ID: {result.get('sentence_id', 'N/A')}")
                print(f"   Sentence: {result.get('sentence', '')}")
                print(f"   Keyword positions: {result.get('keyword_positions', {})}")
            elif match_type == 'semantic':
                print(f"{i}. [SEMANTIC MATCH] ID: {result.get('sentence_id', 'N/A')}")
                print(f"   Similarity: {result.get('similarity', 0):.4f}")
                print(f"   Keywords contained: {result.get('keywords_contained', [])}")
                print(f"   Sentence: {result.get('sentence', '')}")
            
            print()
    else:
        print("No results found")