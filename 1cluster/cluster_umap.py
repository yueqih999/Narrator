import os
import gc
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import umap
import umap.plot
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_umap_visualization(embeddings_path, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info(f"Loading embeddings from {embeddings_path}")
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        logger.info(f"Successfully loaded embeddings for {len(embeddings_dict)} chapters")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return
    
    all_embeddings = []
    chapter_labels = []
    
    for chapter, embeddings in tqdm(embeddings_dict.items(), desc="Preparing data"):
        chapter_size = embeddings.shape[0]
        all_embeddings.append(embeddings)
        chapter_labels.extend([chapter] * chapter_size)
    
    combined_embeddings = np.vstack(all_embeddings)
    logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")
    
    df = pd.DataFrame({'chapter': chapter_labels})
    
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams["font.size"] = 1
    
    logger.info("Fitting UMAP with n_neighbors=50, min_dist=0.3")
    try:
        mapper = umap.UMAP(
            n_neighbors=50,
            min_dist=0.3,
            random_state=42,
            n_jobs=-1,  # 使用所有可用CPU
            verbose=True
        ).fit(combined_embeddings)
        
        logger.info("UMAP fitting completed")
        
        # 保存UMAP结果
        with open(os.path.join(output_dir, 'umap_mapper.pkl'), 'wb') as f:
            pickle.dump(mapper, f)
        logger.info(f"UMAP mapper saved to {os.path.join(output_dir, 'umap_mapper.pkl')}")
        
        # 绘制点图
        logger.info("Generating points visualization")
        fig = plt.figure(figsize=(30, 30))
        umap.plot.points(
            mapper, 
            labels=df['chapter'],
            width=3000, 
            height=3000, 
            color_key_cmap='Spectral', 
            background='white',
            show_legend=True
        )
        plt.savefig(os.path.join(output_dir, 'umap_points.png'))
        plt.close(fig)
        logger.info(f"Points visualization saved to {os.path.join(output_dir, 'umap_points.png')}")
        
        # 绘制连接图
        logger.info("Generating connectivity visualization")
        fig = plt.figure(figsize=(40, 40))
        umap.plot.connectivity(
            mapper, 
            width=4000, 
            height=4000, 
            show_points=False, 
            edge_cmap="GnBu", 
            background="black"
        )
        plt.savefig(os.path.join(output_dir, 'umap_connectivity.png'))
        plt.close(fig)
        logger.info(f"Connectivity visualization saved to {os.path.join(output_dir, 'umap_connectivity.png')}")
        
    except Exception as e:
        logger.error(f"UMAP visualization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    gc.collect()
    logger.info("UMAP visualization completed")

if __name__ == "__main__":
    embeddings_path = "embeddings/all_embeddings.pkl"
    output_dir = "output/umap_results"
    
    create_umap_visualization(embeddings_path, output_dir)