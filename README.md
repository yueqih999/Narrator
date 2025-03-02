# StoryLens
## Project Structure

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
├── 0preprocessing/             # Text preprocessing scripts, finished
├── 1visualization/             # Visualization scripts, finished?
├── 2hierarchy/                 # Hierarchical analysis, unfinished
└── 3rag/                       # Retrieval, unfinished


## Current Progress
- **Text Preprocessing & Embedding**: Completed
- **Visualization & Clustering**: Completed
- **Hierarchical Analysis**: In Progress
- **Retrieval-Augmented Generation**: Planned
- **Model Deployment**: Planned

## Guide
```bash
pip install -r requirements.txt
``` 

## Step 1
upload your txt file in data/
```bash
python 0preprocess/text_process.py
python 0preprocess/embedding.py
``` 

## Step 2
```bash
python 1cluster/cluster_umap_viz.py
``` 