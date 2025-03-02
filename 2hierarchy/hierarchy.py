import os
import gc
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import hdbscan
import time
import logging
import psutil
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)


def process_single_chapter(chapter, embeddings, output_dir, n_clusters=None):
    chapter_output_dir = os.path.join(output_dir, f"chapter_{chapter}")
    os.makedirs(chapter_output_dir, exist_ok=True)
    
    logger.info(f"Processing chapter {chapter}, embeddings shape: {embeddings.shape}")
    logger.info(f"Memory usage before processing: {get_memory_usage():.2f} MB")
    
    if embeddings.shape[1] > 50:
        logger.info(f"Reducing dimensions with PCA from {embeddings.shape[1]} to 50")
        try:
            pca = PCA(n_components=min(50, embeddings.shape[0] - 1))
            embeddings_reduced = pca.fit_transform(embeddings)
            del pca
            gc.collect()
            logger.info(f"PCA completed. Memory usage: {get_memory_usage():.2f} MB")
        except Exception as e:
            logger.error(f"PCA failed: {e}")
            embeddings_reduced = embeddings
    else:
        embeddings_reduced = embeddings
    
    logger.info("Computing distance matrix...")
    try:
        distance_matrix = pdist(embeddings_reduced, metric='cosine')
        logger.info(f"Distance matrix computed. Memory usage: {get_memory_usage():.2f} MB")
    except Exception as e:
        logger.error(f"Failed to compute distance matrix: {e}")
        try:
            distance_matrix = pdist(embeddings_reduced, metric='euclidean')
        except:
            logger.error("Failed with euclidean distance too. Using dummy distance.")
            distance_matrix = np.ones((len(embeddings_reduced) * (len(embeddings_reduced) - 1)) // 2)
    
    logger.info("Performing hierarchical clustering...")
    try:
        Z = linkage(distance_matrix, method='ward')
        logger.info(f"Linkage computed. Memory usage: {get_memory_usage():.2f} MB")
    except Exception as e:
        logger.error(f"Linkage failed: {e}")
        Z = np.zeros((len(embeddings_reduced) - 1, 4))
        for i in range(len(Z)):
            Z[i] = [i, i+1, 1.0, 2]
    
    if n_clusters is None:
        n_clusters = min(max(2, len(embeddings_reduced) // 20), 10)
        logger.info(f"Using {n_clusters} clusters based on data size")
    
    logger.info(f"Performing AgglomerativeClustering with {n_clusters} clusters...")
    try:
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(embeddings_reduced)
        logger.info(f"Clustering completed. Memory usage: {get_memory_usage():.2f} MB")
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        labels = np.zeros(len(embeddings_reduced), dtype=int)
    
    logger.info("Performing t-SNE...")
    try:
        perplexity = min(30, max(5, len(embeddings_reduced) // 5))
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity,
            init='pca',  
            method='barnes_hut' 
        )
        embeddings_2d = tsne.fit_transform(embeddings_reduced)
        logger.info(f"t-SNE completed. Memory usage: {get_memory_usage():.2f} MB")
    except Exception as e:
        logger.error(f"t-SNE failed: {e}")
        embeddings_2d = np.zeros((len(embeddings_reduced), 2))
    
    logger.info("Performing UMAP...")
    try:
        n_neighbors = min(15, max(2, len(embeddings_reduced) // 10))
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=42,
            n_components=2,
            metric='cosine',
            low_memory=True,
            verbose=True
        )
        embeddings_umap = reducer.fit_transform(embeddings_reduced)
        logger.info(f"UMAP completed. Memory usage: {get_memory_usage():.2f} MB")
    except Exception as e:
        logger.error(f"UMAP failed: {e}")
        embeddings_umap = np.zeros((len(embeddings_reduced), 2))
    
    result = {
        'linkage': Z,
        'labels': labels.tolist(),
        'n_clusters': n_clusters,
        'embeddings_2d': embeddings_2d.tolist(),
        'embeddings_umap': embeddings_umap.tolist()
    }
    
    np.save(os.path.join(chapter_output_dir, f"hierarchy_labels_{chapter}.npy"), labels)
    np.save(os.path.join(chapter_output_dir, f"tsne_embeddings_{chapter}.npy"), embeddings_2d)
    np.save(os.path.join(chapter_output_dir, f"umap_embeddings_{chapter}.npy"), embeddings_umap)
    
    try:
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
        plt.title(f'Chapter {chapter} - t-SNE Visualization')
        plt.tight_layout()
        plt.savefig(os.path.join(chapter_output_dir, "tsne_viz.png"), dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
        plt.title(f'Chapter {chapter} - UMAP Visualization')
        plt.tight_layout()
        plt.savefig(os.path.join(chapter_output_dir, "umap_viz.png"), dpi=300)
        plt.close()
        
        plt.figure(figsize=(15, 8))
        dendrogram(Z, leaf_rotation=90, leaf_font_size=8, truncate_mode='lastp', p=30)  
        plt.title(f'Chapter {chapter} - Hierarchical Clustering Dendrogram')
        plt.xlabel('Sentence Index')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(os.path.join(chapter_output_dir, "dendrogram.png"), dpi=300)
        plt.close()
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
    
    gc.collect()
    logger.info(f"Processing of chapter {chapter} completed. Memory usage: {get_memory_usage():.2f} MB")
    
    return result


def hierarchy_analysis(embeddings_dict, output_dir=None, n_clusters=None):
    """修改后的层次分析函数，处理每个章节然后立即释放内存"""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    hierarchy_results = {}
    
    for chapter, embeddings in tqdm(embeddings_dict.items(), desc="Hierarchy cluster"):
        logger.info(f"Starting hierarchy analysis for chapter {chapter}")
        
        try:
            # 处理单个章节
            start_time = time.time()
            result = process_single_chapter(chapter, embeddings, output_dir, n_clusters)
            
            # 保存单个章节的结果，并立即从内存中清除
            with open(os.path.join(output_dir, f'hierarchy_results_{chapter}.pkl'), 'wb') as f:
                pickle.dump(result, f)
            
            hierarchy_results[chapter] = result
            logger.info(f"Chapter {chapter} completed in {time.time() - start_time:.2f} seconds")
            
            # 删除中间变量并强制垃圾回收
            del result
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing chapter {chapter}: {e}")
            # 继续处理下一个章节
    
    # 如果需要，保存所有结果
    if output_dir:
        try:
            with open(os.path.join(output_dir, 'hierarchy_results.pkl'), 'wb') as f:
                pickle.dump(hierarchy_results, f)
        except Exception as e:
            logger.error(f"Error saving combined results: {e}")
            # 尝试分块保存
            for chapter, result in hierarchy_results.items():
                try:
                    with open(os.path.join(output_dir, f'hierarchy_results_{chapter}.pkl'), 'wb') as f:
                        pickle.dump({chapter: result}, f)
                except:
                    pass
    
    logger.info("Hierarchy analysis finished")
    return hierarchy_results


def build_sentence_graph(embeddings_dict, similarity_threshold=0.7, output_dir=None):
    from sklearn.metrics.pairwise import cosine_similarity
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    graph_results = {}
    
    for chapter, embeddings in tqdm(embeddings_dict.items(), desc="Building sentence-level graphs"):
        sim_matrix = cosine_similarity(embeddings)

        G = nx.Graph()
        
        for i in range(len(embeddings)):
            G.add_node(i)
        
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                if sim_matrix[i, j] > similarity_threshold:
                    G.add_edge(i, j, weight=sim_matrix[i, j])

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        avg_degree = sum(dict(G.degree()).values()) / n_nodes if n_nodes > 0 else 0

        try:
            betweenness = nx.betweenness_centrality(G)
            closeness = nx.closeness_centrality(G)
            degree = dict(G.degree())

            important_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        except:
            betweenness = {}
            closeness = {}
            degree = {}
            important_nodes = []

        graph_results[chapter] = {
            'graph': G,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'avg_degree': avg_degree,
            'betweenness': betweenness,
            'closeness': closeness,
            'degree': degree,
            'important_nodes': important_nodes
        }
        
        if output_dir:
            nx.write_gexf(G, os.path.join(output_dir, f"sentence_graph_{chapter}.gexf"))
            
            stats = {
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'avg_degree': avg_degree,
                'important_nodes': [[int(node), float(score)] for node, score in important_nodes]
            }
            
            with open(os.path.join(output_dir, f"graph_stats_{chapter}.json"), 'w') as f:
                json.dump(stats, f, indent=2)
            
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=False, node_size=50, node_color='blue', alpha=0.7)
            nx.draw_networkx_nodes(G, pos, nodelist=[n for n, _ in important_nodes], 
                                  node_color='red', node_size=100, alpha=1.0)
            plt.title(f'Chapter {chapter} - Sentence Similarity Graph')
            plt.savefig(os.path.join(output_dir, f"sentence_graph_{chapter}.png"), dpi=300)
            plt.close()
    
    print("Sentence-level graphs finished!")
    return graph_results


def build_sentence_graph(embeddings_dict, similarity_threshold=0.7, output_dir=None):
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    graph_results = {}
    
    for chapter, embeddings in tqdm(embeddings_dict.items(), desc="Building sentence-level graphs"):
        logger.info(f"Building graph for chapter {chapter}, embeddings shape: {embeddings.shape}")
        logger.info(f"Memory usage before processing: {get_memory_usage():.2f} MB")
        
        try:
            if embeddings.shape[1] > 50:
                logger.info("Reducing dimensions for similarity calculation")
                pca = PCA(n_components=min(50, embeddings.shape[0] - 1))
                embeddings_reduced = pca.fit_transform(embeddings)
                del pca
                gc.collect()
            else:
                embeddings_reduced = embeddings
        except Exception as e:
            logger.error(f"Dimension reduction for graph failed: {e}")
            embeddings_reduced = embeddings
        
        logger.info("Computing similarity matrix...")
        try:
            if len(embeddings_reduced) > 1000:
                batch_size = 500
                sim_matrix = np.zeros((len(embeddings_reduced), len(embeddings_reduced)))
                
                for i in range(0, len(embeddings_reduced), batch_size):
                    end_i = min(i + batch_size, len(embeddings_reduced))
                    batch_i = embeddings_reduced[i:end_i]
                    
                    for j in range(0, len(embeddings_reduced), batch_size):
                        end_j = min(j + batch_size, len(embeddings_reduced))
                        batch_j = embeddings_reduced[j:end_j]
                        
                        batch_sim = cosine_similarity(batch_i, batch_j)
                        sim_matrix[i:end_i, j:end_j] = batch_sim
                        
                    gc.collect()
            else:
                sim_matrix = cosine_similarity(embeddings_reduced)
                
            logger.info(f"Similarity matrix computed. Memory usage: {get_memory_usage():.2f} MB")
        except Exception as e:
            logger.error(f"Similarity matrix computation failed: {e}")
            sim_matrix = np.eye(len(embeddings_reduced))
        
        G = nx.Graph()
        
        for i in range(len(embeddings_reduced)):
            G.add_node(i)
        
        edge_count = 0
        for i in range(len(embeddings_reduced)):
            for j in range(i+1, len(embeddings_reduced)):
                if sim_matrix[i, j] > similarity_threshold:
                    G.add_edge(i, j, weight=sim_matrix[i, j])
                    edge_count += 1
                    
                    if edge_count > 10000 and similarity_threshold < 0.9:
                        similarity_threshold += 0.05
                        logger.info(f"Too many edges, increasing threshold to {similarity_threshold}")
        
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        avg_degree = sum(dict(G.degree()).values()) / n_nodes if n_nodes > 0 else 0
        
        try:
            logger.info("Computing graph centrality metrics...")
            if n_edges > 0:
                if n_nodes > 1000:
                    betweenness = nx.betweenness_centrality(G, k=min(100, n_nodes))
                else:
                    betweenness = nx.betweenness_centrality(G)
                    
                closeness = nx.closeness_centrality(G)
                degree = dict(G.degree())
                
                important_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
            else:
                betweenness = {i: 0 for i in range(n_nodes)}
                closeness = {i: 0 for i in range(n_nodes)}
                degree = {i: 0 for i in range(n_nodes)}
                important_nodes = []
        except Exception as e:
            logger.error(f"Centrality computation failed: {e}")
            betweenness = {}
            closeness = {}
            degree = {}
            important_nodes = []
        
        graph_results[chapter] = {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'avg_degree': avg_degree,
            'betweenness': betweenness,
            'closeness': closeness,
            'degree': degree,
            'important_nodes': important_nodes
        }
        
        if output_dir:
            try:
                nx.write_gexf(G, os.path.join(output_dir, f"sentence_graph_{chapter}.gexf"))
                
                stats = {
                    'n_nodes': n_nodes,
                    'n_edges': n_edges,
                    'avg_degree': avg_degree,
                    'important_nodes': [[int(node), float(score)] for node, score in important_nodes]
                }
                
                with open(os.path.join(output_dir, f"graph_stats_{chapter}.json"), 'w') as f:
                    json.dump(stats, f, indent=2)
                
                if n_nodes <= 500:
                    plt.figure(figsize=(12, 10))
                    pos = nx.spring_layout(G, seed=42)
                    nx.draw(G, pos, with_labels=False, node_size=50, node_color='blue', alpha=0.7)
                    
                    if important_nodes:
                        nx.draw_networkx_nodes(G, pos, nodelist=[n for n, _ in important_nodes], 
                                            node_color='red', node_size=100, alpha=1.0)
                    
                    plt.title(f'Chapter {chapter} - Sentence Similarity Graph')
                    plt.savefig(os.path.join(output_dir, f"sentence_graph_{chapter}.png"), dpi=300)
                    plt.close()
            except Exception as e:
                logger.error(f"Error saving graph results: {e}")
        
        del sim_matrix, G
        gc.collect()
        logger.info(f"Graph building for chapter {chapter} completed. Memory usage: {get_memory_usage():.2f} MB")
    
    logger.info("Sentence-level graphs finished!")
    return graph_results

def main_hierarchy(embeddings_path, output_dir):

    hierarchy_dir = os.path.join(output_dir, 'hierarchy')
    graph_dir = os.path.join(output_dir, 'graph')
    log_dir = os.path.join(output_dir, 'logs')
    
    for dir_path in [hierarchy_dir, graph_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    file_handler = logging.FileHandler(os.path.join(log_dir, f"hierarchy_analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting analysis. Current memory usage: {get_memory_usage():.2f} MB")
    logger.info(f"Loading embedding files: {embeddings_path}")
    
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        
        logger.info(f"Embeddings loaded. Memory usage: {get_memory_usage():.2f} MB")
        
        for chapter, embeddings in embeddings_dict.items():
            logger.info(f"Chapter {chapter}: Embeddings shape {embeddings.shape}")
        
        logger.info("Hierarchy Analysis Start...")
        hierarchy_results = hierarchy_analysis(embeddings_dict, hierarchy_dir)

        del hierarchy_results
        gc.collect()
        
        logger.info("Sentence-level graph start...")
        graph_results = build_sentence_graph(embeddings_dict, similarity_threshold=0.7, output_dir=graph_dir)
        
        logger.info(f"\nAll hierarchy analysis results saved at: {hierarchy_dir}")
        logger.info(f"All hierarchy graphs saved at: {graph_dir}")

    except Exception as e:
        logger.error(f"Fatal error in main processing: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    embeddings_path = "embeddings/all_embeddings.pkl" 
    output_dir = "output"  
    
    main_hierarchy(embeddings_path, output_dir)