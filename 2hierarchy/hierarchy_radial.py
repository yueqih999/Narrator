import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from sklearn.preprocessing import MinMaxScaler
import math


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
        score = row['MI_score']
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


def generate_radial_tree(Z, words, max_display=100, figsize=(15, 15), 
                        n_clusters=None, color_threshold=0.7, 
                        fontsize=9, title="Radial Hierarchy Analysis"):
    
    class Node:
        def __init__(self, id, label=None, x=0, y=0, size=1, cluster=None):
            self.id = id
            self.label = label
            self.x = x
            self.y = y
            self.size = size
            self.cluster = cluster
            self.children = []
        
        def add_child(self, child):
            self.children.append(child)


    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    n_words = len(words)

    if n_clusters is None and color_threshold is not None:
        clusters = fcluster(Z, color_threshold, criterion='distance')
        n_clusters = len(np.unique(clusters))
    elif n_clusters is not None:
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
    else:
        n_clusters = 7
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
    
    print(f"have {n_clusters} clusters")

    cm = get_cmap('tab20')
    colors = [cm(i/n_clusters) for i in range(n_clusters)]

    if n_words > max_display:
        cluster_counts = {}
        for c in clusters:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1

        max_per_cluster = {}
        for c, count in cluster_counts.items():
            max_per_cluster[c] = max(1, round(max_display * count / n_words))

        word_importance = {}
        for i, word in enumerate(words):
            word_importance[i] = i

        display_mask = np.zeros(n_words, dtype=bool)
        
        for c in range(1, n_clusters + 1):
            indices = np.where(clusters == c)[0]
            if len(indices) == 0:
                continue

            sorted_indices = sorted(indices, key=lambda i: word_importance[i])

            selected = sorted_indices[:max_per_cluster.get(c, 1)]
            display_mask[selected] = True
    else:
        display_mask = np.ones(n_words, dtype=bool)
    
    nodes = {}
    for i in range(n_words):
        nodes[i] = Node(i, words[i] if display_mask[i] else None, cluster=clusters[i]-1)

    for i, merge in enumerate(Z):
        a, b = int(merge[0]), int(merge[1])
        node_id = n_words + i

        nodes[node_id] = Node(node_id, cluster=None)  
        nodes[node_id].add_child(nodes[a])
        nodes[node_id].add_child(nodes[b])
    
    root = nodes[n_words + len(Z) - 1]
    

    def layout(node, angle, angle_range, radius, level=0):
        node.level = level
        
        if not node.children:
            node.angle = angle
            node.radius = radius
            node.x = radius * math.cos(angle)
            node.y = radius * math.sin(angle)
            return 1 

        count = 0
        start_angle = angle - angle_range / 2

        child_counts = []
        for child in node.children:
            if hasattr(child, 'count'):
                child_counts.append(child.count)
            else:
                child_count = count_nodes(child)  
                child.count = child_count
                child_counts.append(child_count)
        
        total_children = sum(child_counts)

        current_angle = start_angle
        for i, child in enumerate(node.children):
            child_ratio = child_counts[i] / max(total_children, 1)  
            child_range = angle_range * child_ratio

            layout(child, current_angle + child_range/2, child_range, radius, level+1)
            current_angle += child_range
            count += child_counts[i]
        
        node.angle = angle
        node.radius = radius * 0.5  
        node.x = (radius * 0.5) * math.cos(angle)
        node.y = (radius * 0.5) * math.sin(angle)
        
        return count

    def count_nodes(node):
        if not node.children:
            return 1
        return 1 + sum(count_nodes(child) for child in node.children)
    
    # total_nodes = layout(root, 0, 0, 0, 0)

    layout(root, math.pi/2, 2*math.pi, 10, 0)

    def draw_node(node, parent=None):

        if not node.children:
            marker_color = colors[node.cluster] if node.cluster is not None else 'gray'
            ax.plot(node.x, node.y, 'o', color=marker_color, markersize=5, alpha=0.8)

            if node.label:
                text_angle = node.angle * 180 / math.pi - 90
                if 90 < text_angle < 270:
                    text_angle -= 180
                
                ax.text(node.x * 1.1, node.y * 1.1, node.label, 
                       ha='center', va='center', rotation=text_angle,
                       fontsize=fontsize, color=marker_color,
                       rotation_mode='anchor')
                
        else:
            ax.plot(node.x, node.y, 'o', color='gray', markersize=3, alpha=0.5)
            # pass

        if parent:
            line_color = colors[node.cluster] if node.cluster is not None else 'gray'
            ax.plot([node.x, parent.x], [node.y, parent.y], '-', color=line_color, alpha=0.6, linewidth=1)

        for child in node.children:
            draw_node(child, node)
    
    draw_node(root)

    legend_elements = []
    for i in range(n_clusters):
        cluster_size = np.sum(clusters == i+1)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        label=f'Cluster {i+1} ({cluster_size} words)',
                                        markerfacecolor=colors[i], markersize=8))
    
    ax.legend(handles=legend_elements, loc='upper right', title='Cluster')
    
    for r in np.linspace(2, 10, 5):
        circle = Circle((0, 0), r, fill=False, linestyle='--', alpha=0.3, color='gray')
        ax.add_patch(circle)

    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_axis_off()
    plt.title(title, fontsize=16, pad=20)

    background = Circle((0, 0), 10.5, fill=True, alpha=0.1, color='lightgray')
    ax.add_patch(background)

    plt.tight_layout()
    plt.savefig('output/hierarchy/word_radial_radial.png', dpi=300, bbox_inches='tight')
    
    print("Fig saved at 'output/hierarchy/word_hierarchy_radial.png'")
    
    return fig


def main(file_path, max_words=100, max_display=50, n_clusters=7):
    df = load_word_pairs(file_path)
    if df is None:
        return

    df = simplify_large_word_network(df, max_words)

    similarity_matrix, words, word_to_idx = create_word_similarity_matrix(df)

    Z = perform_hierarchical_clustering(similarity_matrix, method='average')

    fig = generate_radial_tree(
        Z,
        words,
        max_display=max_display,
        n_clusters=n_clusters,
        title=f"Mutual Information radial hierarchy analysis: ({len(words)} words)"
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
    max_display = 100  
    n_clusters = 20
    
    main(file_path, max_words, max_display, n_clusters)