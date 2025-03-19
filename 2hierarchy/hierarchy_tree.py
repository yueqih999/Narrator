import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors
import os



def load_word_pairs(file_path):
    df = pd.read_csv(file_path, dtype={'MI_score': float})
    print(f"Successfully loading {len(df)} word pairs")
    return df


def create_word_similarity_matrix(df):
    all_words = pd.concat([df['word1'], df['word2']]).unique()
    n_words = len(all_words)
    print(f"Totally {n_words} unique words")

    word_to_idx = {word: i for i, word in enumerate(all_words)}

    similarity_matrix = np.zeros((n_words, n_words))

    for _, row in df.iterrows():
        word1, word2 = row['word1'], row['word2']
        mi_score = float(row['MI_score']) 
        i, j = word_to_idx[word1], word_to_idx[word2]

        similarity_matrix[i, j] = mi_score
        similarity_matrix[j, i] = mi_score  
    
    return similarity_matrix, all_words, word_to_idx



def simplify_large_word_network(df, max_words=100):
    all_words = pd.concat([df['word1'], df['word2']]).unique()
    if len(all_words) <= max_words:
        return df  
    
    print(f"Simplify network from {len(all_words)} to {max_words} words")

    word_scores = {}
    for _, row in df.iterrows():
        word1, word2 = row['word1'], row['word2']
        score = float(row['MI_score']) 
        word_scores[word1] = word_scores.get(word1, 0.0) + score
        word_scores[word2] = word_scores.get(word2, 0.0) + score
    
    top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:max_words]
    top_word_set = set(word for word, _ in top_words)
    
    filtered_pairs = df[(df['word1'].isin(top_word_set)) & (df['word2'].isin(top_word_set))]
    
    print(f"Filtered {len(filtered_pairs)} word pairs")
    return filtered_pairs


def perform_hierarchical_clustering(similarity_matrix, method='average'):
    scaler = MinMaxScaler()
    if np.max(similarity_matrix) > 0:  
        normalized_sim = scaler.fit_transform(similarity_matrix.reshape(-1, 1)).reshape(similarity_matrix.shape)
    else:
        normalized_sim = similarity_matrix

    distance_matrix = 1 - normalized_sim

    np.fill_diagonal(distance_matrix, 0)

    n = similarity_matrix.shape[0]
    condensed_distance = []
    for i in range(n):
        for j in range(i+1, n):
            condensed_distance.append(distance_matrix[i, j])

    Z = linkage(condensed_distance, method=method)
    
    return Z

def generate_hierarchy_dendrogram(Z, words, max_display=50, figsize=(16, 10), 
                                 n_clusters=None, color_threshold=0.7, 
                                 orientation='top', title="Word Hierarchy Analysis",
                                 output_path='word_hierarchy_tree.png'):

    plt.figure(figsize=figsize)
    
    if n_clusters is not None:
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        color_threshold = Z[-(n_clusters-1), 2]
        print(f"Setting color threshold to {color_threshold:.4f} for {n_clusters} clusters")

    if len(words) > max_display:

        R = dendrogram(
            Z,
            orientation=orientation,
            truncate_mode='lastp', 
            p=max_display,
            leaf_font_size=9,
            color_threshold=color_threshold,
            above_threshold_color='gray',
            no_labels=False,
            show_leaf_counts=True
        )
    else:
        R = dendrogram(
            Z,
            orientation=orientation,
            labels=words,
            leaf_font_size=9,
            color_threshold=color_threshold,
            above_threshold_color='gray'
        )

    clusters = fcluster(Z, color_threshold, criterion='distance')
    unique_clusters = np.unique(clusters)
    print(f"Generated {len(unique_clusters)} clusters with threshold {color_threshold:.4f}")

    cluster_sizes = {}
    for cluster_id in unique_clusters:
        cluster_sizes[cluster_id] = np.sum(clusters == cluster_id)

    for cluster_id, size in sorted(cluster_sizes.items()):
        print(f"Cluster {cluster_id}: {size} words")
    
    plt.title(title, fontsize=16)
    if orientation in ['top', 'bottom']:
        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
    else:
        plt.xlabel('Distance', fontsize=12)
        plt.ylabel('Words', fontsize=12)

    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Dendrogram saved to {output_path}")
    
    return plt.gcf()

def main(file_path, max_words=100, max_display=50, n_clusters=7, output_path='output/cluster/word_hierarchy_tree.png'):

    df = load_word_pairs(file_path)
    if df is None:
        return

    df = simplify_large_word_network(df, max_words)

    similarity_matrix, words, word_to_idx = create_word_similarity_matrix(df)
    Z = perform_hierarchical_clustering(similarity_matrix, method='average')
    
    fig = generate_hierarchy_dendrogram(
        Z,
        words,
        max_display=max_display,
        n_clusters=n_clusters,
        orientation='top',  # if words more than 50 better to change to "right"
        title=f"Word Hierarchy Analysis ({len(words)} words)",
        output_path=output_path
    )
    
    return {
        'words': words,
        'linkage': Z,
        'similarity_matrix': similarity_matrix,
        'figure': fig
    }

if __name__ == "__main__":
    file_path = "data/analysis/mutual_information.csv"
    max_words = 100  
    max_display = 45
    n_clusters = 7  
    output_path = "output/hierarchy/word_hierarchy_tree.png" 
    
    main(file_path, max_words, max_display, n_clusters, output_path)
