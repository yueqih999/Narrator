import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import umap
import umap.plot
import plotly.express as px
from sentence_transformers.util import cos_sim


def preprocess_data(file_path):
    print("Processing data...")
    df = pd.read_csv(file_path)
    df['character'] = df['character'].fillna('').astype(str)
    df['character'] = df['character'].apply(lambda x: x.split(',')[0] if x else '')
    df = df[df['character'] != '']
    df["character_label"] = pd.factorize(df["character"])[0]
    return df


def create_embeddings(sentences, model_name='model/novel-search-model/final'):
    print(f"Using {model_name} to create embedding...")
    print(f"Loading model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        print(f"Successfully loaded your fine-tuned model from: {model_name}")
    except Exception as e:
        print(f"Error loading custom model: {e}")
        print("Falling back to default model: all-MiniLM-L6-v2")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings


# 2D UMAP
def create_2d_umap(embeddings, n_neighbors=50, min_dist=0.1, spread=6, random_state=42):
    print("Create 2d embedding...")
    mapper = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_epochs=2,
        metric='cosine',
        random_state=random_state
    ).fit(embeddings)
    return mapper


# 3D UMAP
def create_3d_umap(embeddings, n_neighbors=50, min_dist=0.1, spread=9, random_state=42):
    print("Crete 3d embedding...")
    mapper_3d = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=3,
        min_dist=min_dist,
        spread=spread,
        n_epochs=2,
        metric='cosine',
        random_state=random_state
    )
    embedding_3d = mapper_3d.fit_transform(embeddings)
    return mapper_3d, embedding_3d


def setup_matplotlib(dpi=600, font_size=5, font_style="oblique"):
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams["font.size"] = font_size
    plt.rcParams["font.style"] = font_style


def save_points_plot(mapper, labels, output_path, figsize=(10, 10)):
    print(f"Generating scatter graph to {output_path}...")
    setup_matplotlib(dpi=600, font_size=5)
    fig, ax = plt.subplots(figsize=figsize)
    umap.plot.points(mapper, labels=labels, color_key_cmap='BuPu_r', background='black', ax=ax, alpha=0.7)

    for path in ax.collections:
        path.set_sizes([1])
    
    plt.savefig(output_path)
    plt.close()


def save_connectivity_plot(mapper, labels, output_path, width=4000, height=4000, edge_bundling=None):
    print(f"Generating connectivity {'(bundling)' if edge_bundling else ''} to {output_path}...")
    setup_matplotlib(dpi=600, font_size=1)
    
    kwargs = {
        'labels': labels,
        'width': width,
        'height': height,
        'show_points': True,
        'edge_cmap': "BuPu_r",
        'background': "black"
    }
    
    if edge_bundling:
        kwargs['edge_bundling'] = edge_bundling
    
    umap.plot.connectivity(mapper, **kwargs)
    plt.savefig(output_path)
    plt.close()


def create_interactive_2d_plot(embedding, df, output_path):
    print(f"Creating interactive 2d to {output_path}...")
    umap_df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'character': df['character'],
        'sentence': df['sentence']
    })
    
    fig = px.scatter(
        umap_df, 
        x='UMAP1', 
        y='UMAP2',
        color='character',
        hover_name='sentence',
        color_discrete_sequence=px.colors.sequential.BuPu_r,
        template='plotly_dark',
        opacity=0.9,
        title='Sentence Map'
    )
    
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(
            size=10,
            style='italic'
        ),
        autosize=True,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        )
    )
    
    fig.write_html(output_path)
    return fig


def create_interactive_3d_plot(embedding_3d, df, output_path):
    print(f"Creating interactive 3d to {output_path}...")
    umap_df = pd.DataFrame({
        'x': embedding_3d[:, 0],
        'y': embedding_3d[:, 1],
        'z': embedding_3d[:, 2],
        'character': df['character'],
        'sentence': df['sentence']
    })
    
    fig = px.scatter_3d(
        umap_df,
        x='x',
        y='y',
        z='z',
        color='character',
        hover_name='sentence',
        color_discrete_sequence=px.colors.sequential.BuPu_r,
        template='plotly_dark',
        opacity=0.9,
        title='Sentence Map'
    )
    
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(
            size=10,
            style='italic'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            aspectmode='cube'
        )
    )
    
    fig.write_html(
        output_path,
        full_html=True,
        include_plotlyjs=True,
        config={
            'responsive': True,
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
        }
    )
    return fig


def main():
    df = preprocess_data('new_embeddings/embeddings/embedded_all_chapters.csv')
    embeddings = create_embeddings(df['embedding'].tolist())

    mapper = create_2d_umap(embeddings)

    save_points_plot(mapper, df['character'], 'output/cluster/points.png')
    save_connectivity_plot(mapper, df['character'], 'output/cluster/connectivity.png')
    save_connectivity_plot(mapper, df['character'], 'output/cluster/bundling.png', edge_bundling='hammer')

    embedding = mapper.embedding_ if hasattr(mapper, 'embedding_') else mapper

    create_interactive_2d_plot(embedding, df, "output/cluster/umap_interactive_plot.html")

    mapper_3d, embedding_3d = create_3d_umap(embeddings)

    create_interactive_3d_plot(embedding_3d, df, "output/cluster/umap_3d_embedding.html")
    
    print("All finished!")

if __name__ == "__main__":
    main()