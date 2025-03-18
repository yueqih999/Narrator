# StoryLens
## Project Structure
```
StoryLens/
├── data/                       # Raw text data
├── embeddings/                 # Stored embeddings
│   ├── all_embeddings.pkl      # Main embedding file
│   ├── embedded_chapter.csv    # Processed text for each chapter
│   ├── similarity_chapter.json # Cosine similarity for each chapter
│   └── similarity_matrix.npy   # Similarity matrix for each chapter
├── output/                     # Output visualizations and analysis
│   ├── umap_results/           # UMAP visualizations
│   ├── hierarchy/              # Hierarchical clustering results
│   └── graph/                  # Graph analysis results
├── 0model/                     # Model training, text preprocessing and embedding scripts, finished
├── 1visualization/             # Visualization scripts, finished
├── 2hierarchy/                 # Hierarchical analysis, in progress
└── 3rag/                       # Retrieval based accurate matching and semantic search, finished
```

## Current Progress
- **Text Preprocessing & Embedding**: Completed
- **Visualization & Clustering**: Completed
- **Hierarchical Analysis**: In Progress
- **Retrieval-Augmented Generation**: Completed

- **Model Deployment**: Planned

## Guide
```bash
pip install -r requirements.txt
``` 

## 0 model
The training code is under a CPU environment and we use booking corpus dataset to train the pretrain-model 'all-MiniLM-L6-v2': https://huggingface.co/datasets/bookcorpus/bookcorpus
```bash
python 0model/model.py
``` 
The trained model will be saved at model/.
Then upload your txt file in data/, run:
```bash
python 0model/text_process.py
python 0model/embedding.py
``` 

## 1 visualization
Based on calculated sentence embeddings, we use UMAP to do clustering and visualization: https://umap-learn.readthedocs.io/en/latest/basic_usage.html 
```bash
python 1cluster/cluster_umap_viz.py
``` 

## 3 rag
The codes firstly do an accurate retrival, if there is no matching results, it will return top-5 results based on semantic research.
```bash
python 3rag/retrival_word.py
``` 