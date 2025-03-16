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
from sklearn.decomposition import PCA

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_original_sentences():
    """
    Load original sentences from embedded_chapter.csv files
    Returns a dictionary with chapter numbers as keys and lists of sentences as values
    """
    sentences_dict = {}
    chapters = os.listdir('embeddings')

    for filename in chapters:
        if filename.startswith('embedded_chapter') and filename.endswith('.csv'):
            chapter_num = filename.split('_')[-1].split('.')[0]
            df = pd.read_csv(os.path.join('embeddings', filename))
            
            sentences_dict[chapter_num] = df['sentence'].tolist()

    return sentences_dict


def load_character_and_place_data():
    """
    Load character and place information from embedded_chapter.csv files
    """
    character_dict = {}
    place_dict = {}

    chapters = os.listdir('embeddings')
    for filename in chapters:
        if filename.startswith('embedded_chapter') and filename.endswith('.csv'):
            chapter_num = filename.split('_')[-1].split('.')[0]
            df = pd.read_csv(os.path.join('embeddings', filename))

            character_dict[chapter_num] = df['character'].tolist()
            place_dict[chapter_num] = df['place'].tolist()

    return character_dict, place_dict


def get_first_value_or_default(value, default="unknown"):
    """
    Helper function to extract first value from a comma-separated string
    or return a default value if empty
    """
    if isinstance(value, str) and value.strip():
        return value.split(',')[0].strip()
    return default


def create_umap_per_chapter(embeddings_path, output_dir):
    """
    Generate UMAP visualizations for each chapter with labeled edge bundling
    and original sentences displayed on hover
    
    Parameters:
    embeddings_path: Path to the pickle file with embeddings
    output_dir: Directory for output
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info(f"Loading embeddings from {embeddings_path}")
    
    with open(embeddings_path, 'rb') as f:
        embeddings_dict = pickle.load(f)
    logger.info(f"Successfully loaded embeddings for {len(embeddings_dict)} chapters")
    
    logger.info("Loading original sentences")
    sentences_dict = load_original_sentences()
    
    logger.info("Loading character and place data")
    character_dict, place_dict = load_character_and_place_data()

    for chapter, embeddings in tqdm(embeddings_dict.items(), desc="Processing chapters"):
        logger.info(f"Processing chapter {chapter}, embeddings shape: {embeddings.shape}")
        
        chapter_dir = os.path.join(output_dir, f"chapter_{chapter}")
        os.makedirs(chapter_dir, exist_ok=True)
        
        plt.rcParams['figure.dpi'] = 600
        plt.rcParams["font.size"] = 1
        
        try:
            chapter_sentences = sentences_dict.get(str(chapter), [])
            chapter_characters = character_dict.get(str(chapter), [])
            chapter_places = place_dict.get(str(chapter), [])

            if len(chapter_sentences) != len(embeddings):
                logger.warning(f"Number of sentences ({len(chapter_sentences)}) doesn't match number of embeddings ({len(embeddings)}) for chapter {chapter}")
                if len(chapter_sentences) > len(embeddings):
                    chapter_sentences = chapter_sentences[:len(embeddings)]
                else:
                    chapter_sentences.extend([f"Sentence {i} from Chapter {chapter}" for i in range(len(embeddings) - len(chapter_sentences))])

            if len(chapter_characters) != len(embeddings):
                logger.warning(f"Number of character entries ({len(chapter_characters)}) doesn't match number of embeddings ({len(embeddings)}) for chapter {chapter}")
                if len(chapter_characters) > len(embeddings):
                    chapter_characters = chapter_characters[:len(embeddings)]
                else:
                    chapter_characters.extend(["unknown"] * (len(embeddings) - len(chapter_characters)))
            
            if len(chapter_places) != len(embeddings):
                logger.warning(f"Number of place entries ({len(chapter_places)}) doesn't match number of embeddings ({len(embeddings)}) for chapter {chapter}")
                if len(chapter_places) > len(embeddings):
                    chapter_places = chapter_places[:len(embeddings)]
                else:
                    chapter_places.extend(["unknown"] * (len(embeddings) - len(chapter_places)))
            
            logger.info(f"Fitting UMAP for chapter {chapter}")
            mapper = umap.UMAP(n_neighbors=20, min_dist=0.3, random_state=42, verbose=True).fit(embeddings)
            
            with open(os.path.join(chapter_dir, f'umap_mapper_chapter_{chapter}.pkl'), 'wb') as f:
                pickle.dump(mapper, f)
            
            embedding = mapper.embedding_
            processed_characters = [get_first_value_or_default(char) for char in chapter_characters]
            processed_places = [get_first_value_or_default(place) for place in chapter_places]
            
            df = pd.DataFrame({
                'UMAP1': embedding[:, 0],
                'UMAP2': embedding[:, 1],
                'index': np.arange(len(embedding)),
                'chapter': [chapter] * len(embedding),
                'character': chapter_characters,  # Keep original for hover info
                'place': chapter_places,  # Keep original for hover info
                'character_label': processed_characters,  # Processed for visualization labels
                'place_label': processed_places,  # Processed for visualization labels
                'sentence': chapter_sentences  # Original sentences
            })
            
            df['hover_text'] = df['sentence'].apply(lambda x: (x[:100] + '...') if len(x) > 100 else x)
            
            fig = plt.figure(figsize=(100, 100))
            umap.plot.points(
                mapper,
                width=4000, 
                height=4000, 
                color_key_cmap='Spectral', 
                background='black'
            )
            plt.savefig(os.path.join(chapter_dir, f'umap_points_chapter_{chapter}.png'))
            plt.close(fig)

            try:
                fig = plt.figure(figsize=(100, 100))
                character_labels_array = np.array(df['character_label'])
                umap.plot.connectivity(
                    mapper,
                    labels=character_labels_array,
                    edge_bundling='hammer',
                    width=4000, 
                    height=4000, 
                    show_points=True, 
                    edge_cmap="GnBu_r", 
                    background="black"
                )
                plt.savefig(os.path.join(chapter_dir, f'umap_edge_bundling_character_{chapter}.png'))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to create character edge bundling visualization for chapter {chapter}: {e}")

            try:
                fig = plt.figure(figsize=(100, 100))
                place_labels_array = np.array(df['place_label'])
                umap.plot.connectivity(
                    mapper,
                    labels=place_labels_array,
                    edge_bundling='hammer',
                    width=4000, 
                    height=4000, 
                    show_points=True, 
                    edge_cmap="viridis", 
                    background="black"
                )
                plt.savefig(os.path.join(chapter_dir, f'umap_edge_bundling_place_{chapter}.png'))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to create place edge bundling visualization for chapter {chapter}: {e}")
            
            fig = px.scatter(
                df, 
                x='UMAP1', 
                y='UMAP2',
                color='chapter',
                hover_data={
                    'index': False,
                    'chapter': True,
                    'character': True,
                    'place': True,
                    'UMAP1': False,
                    'UMAP2': False,
                    'hover_text': True
                },
                custom_data=["chapter", "character", "place", "sentence"],
                title=f'UMAP Visualization - Chapter {chapter}',
                width=1200,
                height=800
            )
            fig.update_traces(
                marker=dict(size=5),
                hovertemplate="<b>Chapter:</b> %{customdata[0]}<br><b>Character:</b> %{customdata[1]}<br><b>Place:</b> %{customdata[2]}<br><b>Text:</b> %{customdata[3]}"
            )
            fig.update_layout(
                plot_bgcolor='black',
                margin=dict(l=0, r=0, t=0, b=0),
                hoverlabel=dict(
                    bgcolor="black",
                    font_size=12,
                    font_family="Arial"
                )
            )
            pio.write_html(fig, os.path.join(chapter_dir, f'interactive_umap_points_chapter_{chapter}.html'))

            fig = px.scatter(
                df, 
                x='UMAP1', 
                y='UMAP2',
                color='character_label',
                hover_data={
                    'index': False,
                    'chapter': True,
                    'character': True,
                    'place': True,
                    'UMAP1': False,
                    'UMAP2': False,
                    'hover_text': True
                },
                custom_data=["chapter", "character", "place", "sentence"],
                title=f'UMAP Visualization by Character - Chapter {chapter}',
                width=1200,
                height=800
            )

            fig.update_traces(
                marker=dict(size=5),
                hovertemplate="<b>Chapter:</b> %{customdata[0]}<br><b>Character:</b> %{customdata[1]}<br><b>Place:</b> %{customdata[2]}<br><b>Text:</b> %{customdata[3]}"
            )
            fig.update_layout(
                plot_bgcolor='black',
                margin=dict(l=0, r=0, t=0, b=0),
                hoverlabel=dict(
                    bgcolor="black",
                    font_size=12,
                    font_family="Arial"
                )
            )
            pio.write_html(fig, os.path.join(chapter_dir, f'interactive_umap_points_character_{chapter}.html'))

            fig = px.scatter(
                df, 
                x='UMAP1', 
                y='UMAP2',
                color='place_label',
                hover_data={
                    'index': True,
                    'chapter': True,
                    'character': True,
                    'place': True,
                    'UMAP1': False,
                    'UMAP2': False,
                    'hover_text': True
                },
                custom_data=["chapter", "character", "place", "sentence"],
                title=f'UMAP Visualization by Place - Chapter {chapter}',
                width=1200,
                height=800
            )

            fig.update_traces(
                marker=dict(size=5),
                hovertemplate="<b>Chapter:</b> %{customdata[0]}<br><b>Character:</b> %{customdata[1]}<br><b>Place:</b> %{customdata[2]}<br><b>Text:</b> %{customdata[3]}"
            )
            fig.update_layout(
                plot_bgcolor='black',
                margin=dict(l=0, r=0, t=0, b=0),
                hoverlabel=dict(
                    bgcolor="black",
                    font_size=12,
                    font_family="Arial"
                )
            )
            pio.write_html(fig, os.path.join(chapter_dir, f'interactive_umap_points_place_{chapter}.html'))
            
            graph = mapper.graph_
            edges_x = []
            edges_y = []

            graph_csr = graph.tocsr()
            row_indices = graph_csr.indptr
            indices = graph_csr.indices

            for i in range(graph.shape[0]):
                row_start = row_indices[i]
                row_end = row_indices[i+1]
                neighbors = indices[row_start:row_end]
                
                for j in neighbors:
                    edges_x.extend([embedding[i, 0], embedding[j, 0], None])
                    edges_y.extend([embedding[i, 1], embedding[j, 1], None])
            """
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
                        color=df['character_label'],
                        colorscale='Spectral',
                        size=5
                    ),
                    text=df['hover_text'],  # Use hover_text for display
                    hovertemplate="<b>Character:</b> %{text}<br><b>Place:</b> %{text}<br><b>Text:</b> %{text}",
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
                    bgcolor="black",
                    font_size=12,
                    font_family="Arial",
                    font_color="white"
                )
            )
            
            pio.write_html(fig, os.path.join(chapter_dir, f'interactive_umap_connectivity_character_{chapter}.html'))

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
                        color=df['place_label'],
                        colorscale='viridis',
                        size=5
                    ),
                    text=df['hover_text'],
                    hovertemplate="<b>Character:</b> %{text}<br><b>Place:</b> %{text}<br><b>Text:</b> %{text}",
                    hoverinfo='text'
                )
            )
            
            fig.update_layout(
                title=f'UMAP Graph Connectivity (Place) - Chapter {chapter}',
                width=1200,
                height=800,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                hoverlabel=dict(
                    bgcolor="black",
                    font_size=12,
                    font_family="Arial",
                    font_color="white"
                )
            )
            
            pio.write_html(fig, os.path.join(chapter_dir, f'interactive_umap_connectivity_place_{chapter}.html'))
                
            G = nx.Graph()
            
            for i in range(len(embedding)):
                G.add_node(i, pos=(embedding[i, 0], embedding[i, 1]))
            
            for i in range(graph.shape[0]):
                row_start = row_indices[i]
                row_end = row_indices[i+1]
                neighbors = indices[row_start:row_end]
                
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
                        color=[chapter] * len(embedding),
                        colorscale='Spectral',
                        size=5,
                        line=dict(color='white', width=0.5)
                    ),
                    text=df['hover_text'], 
                    hovertemplate="<b>Character:</b> %{text}<br><b>Place:</b> %{text}<br><b>Text:</b> %{text}",
                    hoverinfo='text'
                )
            )
            
            fig.update_layout(
                title=f'UMAP Edge Bundling (Character) - Chapter {chapter}',
                width=1200,
                height=800,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                hoverlabel=dict(
                    bgcolor="black",
                    font_size=12,
                    font_family="Arial",
                    font_color="white"
                )
            )
            
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
            
            pio.write_html(fig, os.path.join(chapter_dir, f'interactive_umap_edge_bundling_character_{chapter}.html'))

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
                        color=df['place_label'],
                        colorscale='viridis',
                        size=5,
                        line=dict(color='white', width=0.5)
                    ),
                    text=df['hover_text'],
                    hovertemplate="<b>Character:</b> %{text}<br><b>Place:</b> %{text}<br><b>Text:</b> %{text}",
                    hoverinfo='text'
                )
            )
            
            fig.update_layout(
                title=f'UMAP Edge Bundling (Place) - Chapter {chapter}',
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
            
            pio.write_html(fig, os.path.join(chapter_dir, f'interactive_umap_edge_bundling_place_{chapter}.html'))"""
            
            logger.info(f"Successfully created UMAP visualizations for chapter {chapter}")
            
        except Exception as e:
            logger.error(f"Failed to create UMAP for chapter {chapter}: {e}")
            import traceback
            logger.error(traceback.format_exc())

        gc.collect()
    
    try:
        create_combined_visualization(embeddings_dict, output_dir, sentences_dict, character_dict, place_dict)
    except Exception as e:
        logger.error(f"Failed to create combined visualization: {e}")
    
    logger.info("All chapter UMAP visualizations completed")

    create_index_page(embeddings_dict.keys(), output_dir)

def create_combined_visualization(embeddings_dict, output_dir, sentences_dict, character_dict, place_dict):
    """
    Create a combined visualization with all chapters and sentence hover
    """
    logger.info("Creating combined visualization of all chapters")
    
    all_embeddings = []
    chapter_labels = []
    all_sentences = []
    all_characters = []
    all_places = []
    processed_characters = []
    processed_places = []
    
    for chapter, embeddings in embeddings_dict.items():
        if embeddings.shape[1] > 100:
            pca = PCA(n_components=100)
            embeddings = pca.fit_transform(embeddings)
        
        all_embeddings.append(embeddings)
        chapter_labels.extend([chapter] * len(embeddings))
        
        chapter_sentences = sentences_dict.get(str(chapter), [])
        if len(chapter_sentences) == 0:
            chapter_sentences = [f"Sentence {i} from Chapter {chapter}" for i in range(len(embeddings))]
        
        if len(chapter_sentences) != len(embeddings):
            if len(chapter_sentences) > len(embeddings):
                chapter_sentences = chapter_sentences[:len(embeddings)]
            else:
                chapter_sentences.extend([f"Sentence {i} from Chapter {chapter}" for i in range(len(embeddings) - len(chapter_sentences))])
        
        all_sentences.extend(chapter_sentences)
    
        chapter_characters = character_dict.get(str(chapter), []) if character_dict else []
        if len(chapter_characters) == 0:
            chapter_characters = ["unknown"] * len(embeddings)

        if len(chapter_characters) != len(embeddings):
            if len(chapter_characters) > len(embeddings):
                chapter_characters = chapter_characters[:len(embeddings)]
            else:
                chapter_characters.extend(["unknown"] * (len(embeddings) - len(chapter_characters)))
        
        all_characters.extend(chapter_characters)

        p_characters = [get_first_value_or_default(char) for char in chapter_characters]
        processed_characters.extend(p_characters)
        
        chapter_places = place_dict.get(str(chapter), []) if place_dict else []
        if len(chapter_places) == 0:
            chapter_places = ["unknown"] * len(embeddings)

        if len(chapter_places) != len(embeddings):
            if len(chapter_places) > len(embeddings):
                chapter_places = chapter_places[:len(embeddings)]
            else:
                chapter_places.extend(["unknown"] * (len(embeddings) - len(chapter_places)))
        
        all_places.extend(chapter_places)
    
        p_places = [get_first_value_or_default(place) for place in chapter_places]
        processed_places.extend(p_places)

    combined_embeddings = np.vstack(all_embeddings)
    
    logger.info(f"Fitting UMAP on combined data with shape {combined_embeddings.shape}")
    try:
        mapper = umap.UMAP(
            n_neighbors=20,
            min_dist=0.3,
            random_state=42,
            verbose=True
        ).fit(combined_embeddings)
        
        embedding = mapper.embedding_
        
        df = pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'chapter': chapter_labels,
            'sentence': all_sentences,
            'character': all_characters,
            'place': all_places
        })
        
        df['hover_text'] = df['sentence'].apply(lambda x: (x[:100] + '...') if len(x) > 100 else x)
        
        with open(os.path.join(output_dir, 'combined_umap_mapper.pkl'), 'wb') as f:
            pickle.dump(mapper, f)

        plt.figure(figsize=(30, 30))
        umap.plot.points(
            mapper,
            labels=np.array(chapter_labels),
            width=3000, 
            height=3000, 
            color_key_cmap='Spectral', 
            background='black'
        )
        plt.savefig(os.path.join(output_dir, 'combined_umap_points.png'))
        plt.close()
        
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
        
        fig = px.scatter(
            df, 
            x='UMAP1', 
            y='UMAP2',
            color='chapter',
            hover_data={
                'chapter': True,
                'UMAP1': False,
                'UMAP2': False,
                'character': True,
                'place': True,
                'hover_text': True
            },
            custom_data=["chapter", "character", "place", "sentence"],
            title='Combined UMAP Visualization - All Chapters',
            width=1200,
            height=800
        )
        fig.update_traces(
            marker=dict(size=5),
            hovertemplate="<b>Chapter:</b> %{customdata[0]}<br><b>Character:</b> %{customdata[1]}<br><b>Place:</b> %{customdata[2]}<br><b>Text:</b> %{customdata[3]}"
        )
        fig.update_layout(
            hoverlabel=dict(
                bgcolor="black",
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
                <a href="interactive_combined_umap_by_chapter.html" target="_blank">Interactive View by Chapter</a>
                <a href="combined_umap_points.png" target="_blank">Points Visualization</a>
                <a href="combined_umap_edge_bundling_character.png" target="_blank">Edge Bundling by Character</a>
                <a href="combined_umap_edge_bundling_place.png" target="_blank">Edge Bundling by Place</a>
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
                <a href="{chapter_dir}/interactive_umap_points_chapter_{chapter}.html" target="_blank">Points by Chapter</a>
                <a href="{chapter_dir}/interactive_umap_points_character_{chapter}.html" target="_blank">Points by Character</a>
                <a href="{chapter_dir}/interactive_umap_points_place_{chapter}.html" target="_blank">Points by Place</a>
                <a href="{chapter_dir}/umap_edge_bundling_character_{chapter}.png" target="_blank">Static Edge Bundling (Character)</a>
                <a href="{chapter_dir}/umap_edge_bundling_place_{chapter}.png" target="_blank">Static Edge Bundling (Place)</a>
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
    output_dir = "output/new_umap_results"
    
    create_umap_per_chapter(embeddings_path, output_dir)