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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.spatial import cKDTree
import networkx as nx


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_umap_per_chapter(embeddings_path, output_dir):
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
    
    for chapter, embeddings in tqdm(embeddings_dict.items(), desc="Processing chapters"):
        logger.info(f"Processing chapter {chapter}, embeddings shape: {embeddings.shape}")

        chapter_dir = os.path.join(output_dir, f"chapter_{chapter}")
        os.makedirs(chapter_dir, exist_ok=True)
        
        plt.rcParams['figure.dpi'] = 600
        plt.rcParams["font.size"] = 1
        
        try:
            logger.info(f"Fitting UMAP for chapter {chapter}")
            mapper = umap.UMAP(
                n_neighbors=50,
                min_dist=0.3,
                random_state=42,
                verbose=True
            ).fit(embeddings)
            
            with open(os.path.join(chapter_dir, f'umap_mapper_chapter_{chapter}.pkl'), 'wb') as f:
                pickle.dump(mapper, f)
            
            embedding = mapper.embedding_
            
            labels = np.arange(len(embedding))
            
            fig = plt.figure(figsize=(30, 30))
            umap.plot.points(
                mapper,
                width=3000, 
                height=3000, 
                color_key_cmap='Spectral', 
                background='white'
            )
            plt.savefig(os.path.join(chapter_dir, f'umap_points_chapter_{chapter}.png'))
            plt.close(fig)
            
            fig = plt.figure(figsize=(40, 40))
            umap.plot.connectivity(
                mapper, 
                width=4000, 
                height=4000, 
                show_points=False, 
                edge_cmap="GnBu", 
                background="black"
            )
            plt.savefig(os.path.join(chapter_dir, f'umap_connectivity_chapter_{chapter}.png'))
            plt.close(fig)
            
            fig = plt.figure(figsize=(30, 30))
            umap.plot.connectivity(
                mapper, 
                edge_bundling='hammer',
                width=3000, 
                height=3000, 
                show_points=True, 
                edge_cmap="GnBu_r", 
                background="black"
            )
            plt.savefig(os.path.join(chapter_dir, f'umap_edge_bundling_chapter_{chapter}.png'))
            plt.close(fig)
            
            df = pd.DataFrame({
                'UMAP1': embedding[:, 0],
                'UMAP2': embedding[:, 1],
                'index': labels
            })
            
            fig = px.scatter(
                df, 
                x='UMAP1', 
                y='UMAP2',
                color=df.index, 
                color_continuous_scale='Spectral',
                hover_data=['index'],
                title=f'UMAP Visualization - Chapter {chapter}',
                width=1200,
                height=800
            )
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            pio.write_html(fig, os.path.join(chapter_dir, f'interactive_umap_points_chapter_{chapter}.html'))
            
            graph = mapper.graph_
            edges_x = []
            edges_y = []
            
            for i, neighbors in enumerate(graph.tocsr().indices.reshape(graph.shape[0], -1)):
                neighbors = neighbors[neighbors >= 0]  
                for j in neighbors:
                    edges_x.extend([embedding[i, 0], embedding[j, 0], None])
                    edges_y.extend([embedding[i, 1], embedding[j, 1], None])
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=edges_x, 
                    y=edges_y,
                    mode='lines',
                    line=dict(color='rgba(0, 150, 255, 0.2)', width=1),
                    hoverinfo='none'
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=embedding[:, 0], 
                    y=embedding[:, 1],
                    mode='markers',
                    marker=dict(
                        color=labels,
                        colorscale='Spectral',
                        size=5
                    ),
                    text=labels,
                    hoverinfo='text'
                )
            )
            
            fig.update_layout(
                title=f'UMAP Graph Connectivity - Chapter {chapter}',
                width=1200,
                height=800,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white')
            )
            
            pio.write_html(fig, os.path.join(chapter_dir, f'interactive_umap_connectivity_chapter_{chapter}.html'))
            
            G = nx.Graph()
            
            for i in range(len(embedding)):
                G.add_node(i, pos=(embedding[i, 0], embedding[i, 1]))
            
            for i, neighbors in enumerate(graph.tocsr().indices.reshape(graph.shape[0], -1)):
                neighbors = neighbors[neighbors >= 0]
                for j in neighbors:
                    G.add_edge(i, j)
            
            pos = nx.spring_layout(G, pos={i: (embedding[i, 0], embedding[i, 1]) for i in range(len(embedding))}, fixed=list(range(len(embedding))), iterations=50)
            
            fig = go.Figure()
            
            for edge in G.edges():
                i, j = edge
                x0, y0 = embedding[i]
                x1, y1 = embedding[j]
                
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                
                t = np.linspace(0, 1, 100)
                x = (1-t)**2 * x0 + 2*(1-t)*t * mid_x + t**2 * x1
                y = (1-t)**2 * y0 + 2*(1-t)*t * mid_y + t**2 * y1
                
                fig.add_trace(
                    go.Scatter(
                        x=x, 
                        y=y,
                        mode='lines',
                        line=dict(
                            color='rgba(0, 200, 255, 0.1)', 
                            width=1
                        ),
                        hoverinfo='none'
                    )
                )
            
            fig.add_trace(
                go.Scatter(
                    x=embedding[:, 0], 
                    y=embedding[:, 1],
                    mode='markers',
                    marker=dict(
                        color=labels,
                        colorscale='Spectral',
                        size=5,
                        line=dict(color='white', width=0.5)
                    ),
                    text=labels,
                    hoverinfo='text'
                )
            )
            
            fig.update_layout(
                title=f'UMAP Edge Bundling Visualization - Chapter {chapter}',
                width=1200,
                height=800,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white')
            )
            
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
            
            pio.write_html(fig, os.path.join(chapter_dir, f'interactive_umap_edge_bundling_chapter_{chapter}.html'))
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=(
                    f'Points', 
                    f'Connectivity',
                    f'Edge Bundling'
                ),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=embedding[:, 0], 
                    y=embedding[:, 1],
                    mode='markers',
                    marker=dict(
                        color=labels,
                        colorscale='Spectral',
                        size=5
                    ),
                    text=labels,
                    hoverinfo='text'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=edges_x, 
                    y=edges_y,
                    mode='lines',
                    line=dict(color='rgba(0, 150, 255, 0.2)', width=1),
                    hoverinfo='none'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=embedding[:, 0], 
                    y=embedding[:, 1],
                    mode='markers',
                    marker=dict(
                        color=labels,
                        colorscale='Spectral',
                        size=3
                    ),
                    text=labels,
                    hoverinfo='text'
                ),
                row=1, col=2
            )
            
            for edge in G.edges():
                i, j = edge
                x0, y0 = embedding[i]
                x1, y1 = embedding[j]
                
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                
                t = np.linspace(0, 1, 100)
                x = (1-t)**2 * x0 + 2*(1-t)*t * mid_x + t**2 * x1
                y = (1-t)**2 * y0 + 2*(1-t)*t * mid_y + t**2 * y1
                
                fig.add_trace(
                    go.Scatter(
                        x=x, 
                        y=y,
                        mode='lines',
                        line=dict(
                            color='rgba(0, 200, 255, 0.1)', 
                            width=1
                        ),
                        hoverinfo='none'
                    ),
                    row=1, col=3
                )
            
            fig.add_trace(
                go.Scatter(
                    x=embedding[:, 0], 
                    y=embedding[:, 1],
                    mode='markers',
                    marker=dict(
                        color=labels,
                        colorscale='Spectral',
                        size=3,
                        line=dict(color='white', width=0.5)
                    ),
                    text=labels,
                    hoverinfo='text'
                ),
                row=1, col=3
            )
            
            fig.update_layout(
                height=600,
                width=1800,
                title_text=f"UMAP Visualizations for Chapter {chapter}",
                showlegend=False
            )
            
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            for col in [2, 3]:
                fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=col)
                fig.update_yaxes(showgrid=False, zeroline=False, row=1, col=col)
                fig.update_xaxes(backgroundcolor="black", row=1, col=col)
                fig.update_yaxes(backgroundcolor="black", row=1, col=col)
            
            pio.write_html(
                fig, 
                os.path.join(chapter_dir, f'interactive_umap_all_visualizations_chapter_{chapter}.html')
            )
            
            logger.info(f"Successfully created UMAP visualizations for chapter {chapter}")
            
        except Exception as e:
            logger.error(f"Failed to create UMAP for chapter {chapter}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        gc.collect()
    
    logger.info("All chapter UMAP visualizations completed")
    
    create_index_page(embeddings_dict.keys(), output_dir)

def create_index_page(chapters, output_dir):
    """创建索引页面，链接到所有章节的可视化"""
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>UMAP Visualizations Index</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .chapter-list { margin-top: 20px; }
            .chapter-item { margin-bottom: 15px; }
            .chapter-title { font-size: 18px; font-weight: bold; margin-bottom: 5px; }
            .viz-links a { margin-right: 15px; color: #0066cc; text-decoration: none; }
            .viz-links a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>UMAP Visualizations for All Chapters</h1>
        <div class="chapter-list">
    """
    
    for chapter in sorted(chapters):
        chapter_dir = f"chapter_{chapter}"
        
        index_html += f"""
        <div class="chapter-item">
            <div class="chapter-title">Chapter {chapter}</div>
            <div class="viz-links">
                <a href="{chapter_dir}/interactive_umap_points_chapter_{chapter}.html" target="_blank">Points Visualization</a>
                <a href="{chapter_dir}/interactive_umap_connectivity_chapter_{chapter}.html" target="_blank">Connectivity Visualization</a>
                <a href="{chapter_dir}/interactive_umap_edge_bundling_chapter_{chapter}.html" target="_blank">Edge Bundling Visualization</a>
                <a href="{chapter_dir}/interactive_umap_all_visualizations_chapter_{chapter}.html" target="_blank">All Visualizations Combined</a>
            </div>
        </div>
        """
    
    index_html += """
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'index.html'), 'w') as f:
        f.write(index_html)
    
    logger.info(f"Created index page at {os.path.join(output_dir, 'index.html')}")

if __name__ == "__main__":
    embeddings_path = "embeddings/all_embeddings.pkl"
    output_dir = "output/umap_results"
    
    create_umap_per_chapter(embeddings_path, output_dir)