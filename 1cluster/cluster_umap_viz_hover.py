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
import networkx as nx
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_original_sentences():
    """
    Load original sentences from embedded_chapter.csv files
    Returns a dictionary with chapter numbers as keys and lists of sentences as values
    """
    sentences_dict = {}
    try:
        # Try to load from all chapters
        chapters = os.listdir('embeddings')
        for filename in chapters:
            if filename.startswith('embedded_chapter') and filename.endswith('.csv'):
                chapter_num = filename.split('_')[-1].split('.')[0]
                
                # Load the CSV file
                df = pd.read_csv(os.path.join('embeddings', filename))
                
                # Extract sentences
                if 'text' in df.columns:
                    sentences_dict[chapter_num] = df['text'].tolist()
                elif 'sentence' in df.columns:
                    sentences_dict[chapter_num] = df['sentence'].tolist()
    except Exception as e:
        logger.error(f"Failed to load original sentences: {e}")
        
        # Try alternative approach - check if there's a JSON file with sentences
        try:
            with open('embeddings/sentences.json', 'r') as f:
                sentences_dict = json.load(f)
        except:
            logger.error("Could not find sentences.json either")
    
    return sentences_dict

def create_umap_per_chapter(embeddings_path, output_dir):
    """
    Generate UMAP visualizations for each chapter with labeled edge bundling
    and original sentences displayed on hover
    
    Parameters:
    embeddings_path: Path to the pickle file with embeddings
    output_dir: Directory for output
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_path}")
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        logger.info(f"Successfully loaded embeddings for {len(embeddings_dict)} chapters")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return
    
    # Load original sentences
    logger.info("Loading original sentences")
    sentences_dict = load_original_sentences()
    
    # Create UMAP for each chapter
    for chapter, embeddings in tqdm(embeddings_dict.items(), desc="Processing chapters"):
        logger.info(f"Processing chapter {chapter}, embeddings shape: {embeddings.shape}")
        
        # Create output directory for this chapter
        chapter_dir = os.path.join(output_dir, f"chapter_{chapter}")
        os.makedirs(chapter_dir, exist_ok=True)
        
        # Set matplotlib params
        plt.rcParams['figure.dpi'] = 600
        plt.rcParams["font.size"] = 1
        
        try:
            # Get sentences for this chapter if available
            chapter_sentences = sentences_dict.get(str(chapter), [])
            if len(chapter_sentences) == 0:
                # If no sentences found, create dummy sentences
                chapter_sentences = [f"Sentence {i} from Chapter {chapter}" for i in range(len(embeddings))]
            
            # If length doesn't match, adjust
            if len(chapter_sentences) != len(embeddings):
                logger.warning(f"Number of sentences ({len(chapter_sentences)}) doesn't match number of embeddings ({len(embeddings)}) for chapter {chapter}")
                if len(chapter_sentences) > len(embeddings):
                    chapter_sentences = chapter_sentences[:len(embeddings)]
                else:
                    chapter_sentences.extend([f"Sentence {i} from Chapter {chapter}" for i in range(len(embeddings) - len(chapter_sentences))])
            
            # Fit UMAP
            logger.info(f"Fitting UMAP for chapter {chapter}")
            mapper = umap.UMAP(
                n_neighbors=50,
                min_dist=0.3,
                random_state=42,
                verbose=True
            ).fit(embeddings)
            
            # Save UMAP mapper
            with open(os.path.join(chapter_dir, f'umap_mapper_chapter_{chapter}.pkl'), 'wb') as f:
                pickle.dump(mapper, f)
            
            # Get the embedding coordinates
            embedding = mapper.embedding_
            
            # Create dataframe with chapter labels and original sentences
            df = pd.DataFrame({
                'UMAP1': embedding[:, 0],
                'UMAP2': embedding[:, 1],
                'index': np.arange(len(embedding)),
                'chapter': [chapter] * len(embedding),
                'sentence': chapter_sentences  # Add original sentences
            })
            
            # Truncate long sentences for better display
            df['hover_text'] = df['sentence'].apply(lambda x: (x[:100] + '...') if len(x) > 100 else x)
            
            # 1. Basic point visualization
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
            
            # 2. Basic connectivity visualization
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
            
            # 3. Edge bundling visualization with chapter labels
            fig = plt.figure(figsize=(30, 30))
            labels_array = df['chapter'].values  # Convert to numpy array
            umap.plot.connectivity(
                mapper,
                labels=labels_array,
                edge_bundling='hammer',
                width=3000, 
                height=3000, 
                show_points=True, 
                edge_cmap="GnBu_r", 
                background="black"
            )
            plt.savefig(os.path.join(chapter_dir, f'umap_edge_bundling_chapter_{chapter}.png'))
            plt.close(fig)
            
            # Create interactive visualizations with Plotly
            # 1. Points visualization with sentence hover
            fig = px.scatter(
                df, 
                x='UMAP1', 
                y='UMAP2',
                color='chapter',
                hover_data={
                    'index': True,
                    'chapter': True,
                    'UMAP1': False,
                    'UMAP2': False,
                    'hover_text': True
                },
                custom_data=['hover_text'],  # Add sentences as custom data
                title=f'UMAP Visualization - Chapter {chapter}',
                width=1200,
                height=800
            )
            fig.update_traces(
                marker=dict(size=5),
                hovertemplate="<b>Chapter:</b> %{customdata[0]}<br><b>Index:</b> %{index}<br><b>Text:</b> %{customdata[0]}"
            )
            fig.update_layout(
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            pio.write_html(fig, os.path.join(chapter_dir, f'interactive_umap_points_chapter_{chapter}.html'))
            
            # 2. Connectivity visualization with sentence hover
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
                        color=[chapter] * len(embedding),
                        colorscale='Spectral',
                        size=5
                    ),
                    text=df['hover_text'],  # Use hover_text for display
                    hovertemplate="<b>Chapter:</b> %{text}<br><b>Index:</b> %{text}<br><b>Text:</b> %{text}",
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
                font=dict(color='white'),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial",
                    font_color="black"
                )
            )
            
            pio.write_html(fig, os.path.join(chapter_dir, f'interactive_umap_connectivity_chapter_{chapter}.html'))
            
            # 3. Edge bundling interactive visualization with sentence hover
            G = nx.Graph()
            
            for i in range(len(embedding)):
                G.add_node(i, pos=(embedding[i, 0], embedding[i, 1]))
            
            for i, neighbors in enumerate(graph.tocsr().indices.reshape(graph.shape[0], -1)):
                neighbors = neighbors[neighbors >= 0]
                for j in neighbors:
                    G.add_edge(i, j)
            
            # Simplified edge bundling for interactive viz
            pos = nx.spring_layout(G, pos={i: (embedding[i, 0], embedding[i, 1]) for i in range(len(embedding))}, fixed=list(range(len(embedding))), iterations=50)
            
            fig = go.Figure()
            
            # Add bundled edges
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
            
            # Add points with sentence hover
            fig.add_trace(
                go.Scatter(
                    x=embedding[:, 0], 
                    y=embedding[:, 1],
                    mode='markers',
                    marker=dict(
                        color=[chapter] * len(embedding),
                        colorscale='Spectral',
                        size=5,
                        line=dict(color='white', width=0.5)
                    ),
                    text=df['hover_text'],  # Display original sentences on hover
                    hovertemplate="<b>Chapter:</b> %{text}<br><b>Index:</b> %{text}<br><b>Text:</b> %{text}",
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
                font=dict(color='white'),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial",
                    font_color="black"
                )
            )
            
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
            
            pio.write_html(fig, os.path.join(chapter_dir, f'interactive_umap_edge_bundling_chapter_{chapter}.html'))
            
            logger.info(f"Successfully created UMAP visualizations for chapter {chapter}")
            
        except Exception as e:
            logger.error(f"Failed to create UMAP for chapter {chapter}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Force garbage collection
        gc.collect()
    
    # Create the combined visualization with all chapters
    try:
        create_combined_visualization(embeddings_dict, output_dir, sentences_dict)
    except Exception as e:
        logger.error(f"Failed to create combined visualization: {e}")
    
    logger.info("All chapter UMAP visualizations completed")
    
    # Create index page
    create_index_page(embeddings_dict.keys(), output_dir)

def create_combined_visualization(embeddings_dict, output_dir, sentences_dict):
    """
    Create a combined visualization with all chapters and sentence hover
    """
    logger.info("Creating combined visualization of all chapters")
    
    # Combine all embeddings and sentences
    all_embeddings = []
    chapter_labels = []
    all_sentences = []
    
    for chapter, embeddings in embeddings_dict.items():
        # Use PCA to reduce dimension for very high-dimensional embeddings
        if embeddings.shape[1] > 100:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=100)
            embeddings = pca.fit_transform(embeddings)
        
        all_embeddings.append(embeddings)
        chapter_labels.extend([chapter] * len(embeddings))
        
        # Get sentences for this chapter if available
        chapter_sentences = sentences_dict.get(str(chapter), [])
        if len(chapter_sentences) == 0:
            # If no sentences found, create dummy sentences
            chapter_sentences = [f"Sentence {i} from Chapter {chapter}" for i in range(len(embeddings))]
        
        # If length doesn't match, adjust
        if len(chapter_sentences) != len(embeddings):
            if len(chapter_sentences) > len(embeddings):
                chapter_sentences = chapter_sentences[:len(embeddings)]
            else:
                chapter_sentences.extend([f"Sentence {i} from Chapter {chapter}" for i in range(len(embeddings) - len(chapter_sentences))])
        
        all_sentences.extend(chapter_sentences)
    
    combined_embeddings = np.vstack(all_embeddings)
    
    # Fit UMAP on combined data
    logger.info(f"Fitting UMAP on combined data with shape {combined_embeddings.shape}")
    try:
        mapper = umap.UMAP(
            n_neighbors=50,
            min_dist=0.3,
            random_state=42,
            verbose=True
        ).fit(combined_embeddings)
        
        # Get embeddings
        embedding = mapper.embedding_
        
        # Create dataframe with sentences
        df = pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'chapter': chapter_labels,
            'sentence': all_sentences
        })
        
        # Truncate long sentences for better display
        df['hover_text'] = df['sentence'].apply(lambda x: (x[:100] + '...') if len(x) > 100 else x)
        
        # Save combined UMAP mapper
        with open(os.path.join(output_dir, 'combined_umap_mapper.pkl'), 'wb') as f:
            pickle.dump(mapper, f)
        
        # Matplotlib visualizations
        # 1. Points
        plt.figure(figsize=(30, 30))
        umap.plot.points(
            mapper,
            labels=np.array(chapter_labels),
            width=3000, 
            height=3000, 
            color_key_cmap='Spectral', 
            background='white'
        )
        plt.savefig(os.path.join(output_dir, 'combined_umap_points.png'))
        plt.close()
        
        # 2. Edge bundling with chapter labels
        plt.figure(figsize=(30, 30))
        labels_array = np.array(chapter_labels)
        umap.plot.connectivity(
            mapper,
            labels=labels_array,
            edge_bundling='hammer',
            width=3000, 
            height=3000, 
            show_points=True, 
            edge_cmap="GnBu_r", 
            background="black"
        )
        plt.savefig(os.path.join(output_dir, 'combined_umap_edge_bundling.png'))
        plt.close()
        
        # Create interactive visualization with sentence hover
        fig = px.scatter(
            df, 
            x='UMAP1', 
            y='UMAP2',
            color='chapter',
            hover_data={
                'chapter': True,
                'UMAP1': False,
                'UMAP2': False,
                'hover_text': True
            },
            title='Combined UMAP Visualization - All Chapters',
            width=1200,
            height=800
        )
        fig.update_traces(
            marker=dict(size=5),
            hovertemplate="<b>Chapter:</b> %{customdata[0]}<br><b>Text:</b> %{customdata[1]}"
        )
        fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        pio.write_html(fig, os.path.join(output_dir, 'interactive_combined_umap.html'))
        
        logger.info("Successfully created combined visualization")
        
    except Exception as e:
        logger.error(f"Failed to create combined visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())

def create_index_page(chapters, output_dir):
    """Create an index page with links to all visualizations"""
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
        <h1>UMAP Visualizations</h1>
        
        <div class="chapter-item">
            <div class="chapter-title">Combined Visualization (All Chapters)</div>
            <div class="viz-links">
                <a href="interactive_combined_umap.html" target="_blank">Interactive Combined View</a>
                <a href="combined_umap_points.png" target="_blank">Points Visualization</a>
                <a href="combined_umap_edge_bundling.png" target="_blank">Edge Bundling Visualization</a>
            </div>
        </div>
        
        <h2>Individual Chapter Visualizations</h2>
        <div class="chapter-list">
    """
    
    for chapter in sorted(chapters):
        chapter_dir = f"chapter_{chapter}"
        
        index_html += f"""
        <div class="chapter-item">
            <div class="chapter-title">Chapter {chapter}</div>
            <div class="viz-links">
                <a href="{chapter_dir}/interactive_umap_points_chapter_{chapter}.html" target="_blank">Points</a>
                <a href="{chapter_dir}/interactive_umap_connectivity_chapter_{chapter}.html" target="_blank">Connectivity</a>
                <a href="{chapter_dir}/interactive_umap_edge_bundling_chapter_{chapter}.html" target="_blank">Edge Bundling</a>
                <a href="{chapter_dir}/umap_edge_bundling_chapter_{chapter}.png" target="_blank">Static Edge Bundling</a>
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
    # Set input and output paths
    embeddings_path = "embeddings/all_embeddings.pkl"
    output_dir = "output/umap_results"
    
    # Run UMAP visualization
    create_umap_per_chapter(embeddings_path, output_dir)