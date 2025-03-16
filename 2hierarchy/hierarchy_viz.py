import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors


def load_word_pairs(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loading {len(df)} word pairs")
        return df
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def create_word_similarity_matrix(df):
    all_words = pd.concat([df['word1'], df['word2']]).unique()
    n_words = len(all_words)
    print(f"All {n_words} unique words")

    word_to_idx = {word: i for i, word in enumerate(all_words)}
    similarity_matrix = np.zeros((n_words, n_words))

    for _, row in df.iterrows():
        word1, word2, mi_score = row['word1'], row['word2'], row['MI_score']
        i, j = word_to_idx[word1], word_to_idx[word2]

        similarity_matrix[i, j] = mi_score
        similarity_matrix[j, i] = mi_score  
    
    return similarity_matrix, all_words, word_to_idx


def generate_hierarchy_tree(similarity_matrix, words, method='average', 
                           max_display=50, color_threshold=None,
                           figsize=(16, 10)):
    scaler = MinMaxScaler()
    if np.max(similarity_matrix) > 0:
        normalized_sim = scaler.fit_transform(similarity_matrix.reshape(-1, 1)).reshape(similarity_matrix.shape)
    else:
        normalized_sim = similarity_matrix

    distance_matrix = 1 - normalized_sim

    np.fill_diagonal(distance_matrix, 0)
    n = len(words)
    condensed_distance = []
    for i in range(n):
        for j in range(i+1, n):
            condensed_distance.append(distance_matrix[i, j])

    Z = linkage(condensed_distance, method=method)

    plt.figure(figsize=figsize)

    display_labels = words
    if len(words) > max_display:
        word_scores = np.sum(similarity_matrix, axis=1)
        top_indices = np.argsort(word_scores)[-max_display:]

        display_labels = np.array(['' for _ in range(len(words))])
        display_labels[top_indices] = np.array(words)[top_indices]
        
        print(f"Only display {max_display} most important labels")

    colors = list(mcolors.TABLEAU_COLORS.values())
    R = dendrogram(
        Z,
        labels=display_labels,
        orientation='top',
        leaf_rotation=90,
        leaf_font_size=9,
        color_threshold=color_threshold,
        above_threshold_color='gray',
        link_color_func=lambda k: colors[k % len(colors)]
    )
    plt.title('Word hierarchy structure', fontsize=16)
    plt.xlabel('word', fontsize=12)
    plt.ylabel('distance', fontsize=12)
    plt.tight_layout()
    
    plt.savefig('word_hierarchy_tree.png', dpi=300, bbox_inches='tight')
    
    return Z, plt.gcf()


def simplify_large_word_network(df, max_words=100):
    all_words = pd.concat([df['word1'], df['word2']]).unique()
    if len(all_words) <= max_words:
        return df  
    
    print(f"Simplify from {len(all_words)} words to {max_words} words")

    word_scores = {}
    for _, row in df.iterrows():
        word1, word2 = row['word1'], row['word2']
        score = float(row['MI_score'])  
        word_scores[word1] = word_scores.get(word1, 0.0) + score
        word_scores[word2] = word_scores.get(word2, 0.0) + score

    top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:max_words]
    top_word_set = set(word for word, _ in top_words)

    filtered_pairs = df[(df['word1'].isin(top_word_set)) & (df['word2'].isin(top_word_set))]
    
    print(f"All {len(filtered_pairs)} word pairs")
    return filtered_pairs


def main(file_path, max_words=100, max_display=50):
    df = load_word_pairs(file_path)
    if df is None:
        return

    df = simplify_large_word_network(df, max_words)

    similarity_matrix, words, word_to_idx = create_word_similarity_matrix(df)
    Z, fig = generate_hierarchy_tree(
        similarity_matrix, 
        words,
        method='average',
        max_display=max_display,
        color_threshold=0.7  
    )
    print(f"All saved at 'word_hierarchy_tree.png'")
    return {
        'words': words,
        'linkage': Z,
        'similarity_matrix': similarity_matrix
    }

if __name__ == "__main__":
    file_path = "data/analysis/mutual_information.csv" 
    max_words = 100  
    max_display = 50 

    main(file_path)