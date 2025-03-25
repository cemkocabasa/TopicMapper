import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Any
import pandas as pd
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import io

def create_topic_visualization(topics: List[Dict[str, Any]]):
    """
    Create a visualization for the topic analysis.
    
    Args:
        topics: List of topic dictionaries containing name and weight
    
    Returns:
        Plotly figure object
    """
    # Extract topic names and weights
    topic_names = [topic['topic'] for topic in topics]
    weights = [topic['weight'] for topic in topics]
    
    # Create a pie chart
    fig = go.Figure(data=[go.Pie(
        labels=topic_names,
        values=weights,
        hole=0.3,
        marker_colors=px.colors.qualitative.Set3
    )])
    
    # Update layout
    fig.update_layout(
        title="Topic Distribution",
        height=500
    )
    
    # Add text annotation in the center
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        insidetextorientation='radial'
    )
    
    return fig

def create_cluster_visualization(embeddings, clusters, responses, n_clusters):
    """
    Create a visualization for the clustering results.
    
    Args:
        embeddings: Numpy array of embeddings
        clusters: List of cluster assignments
        responses: List of text responses
        n_clusters: Number of clusters
    
    Returns:
        Plotly figure object
    """
    # Use PCA to reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'cluster': [f"Cluster {c+1}" for c in clusters],
        'response': responses
    })
    
    # Generate a color map
    colors = px.colors.qualitative.G10[:n_clusters]
    
    # Create a scatter plot
    fig = px.scatter(
        df, 
        x='x', 
        y='y', 
        color='cluster',
        hover_data=['response'],
        color_discrete_sequence=colors,
        title="Response Clusters",
        labels={'x': 'Component 1', 'y': 'Component 2'},
        height=600
    )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        legend_title_text='Cluster'
    )
    
    # Add cluster centers
    cluster_centers = []
    for i in range(n_clusters):
        cluster_points = reduced_embeddings[np.array(clusters) == i]
        if len(cluster_points) > 0:
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append((center[0], center[1], i))
    
    # Add annotations for cluster centers
    for x, y, i in cluster_centers:
        fig.add_annotation(
            x=x,
            y=y,
            text=f"Cluster {i+1}",
            showarrow=True,
            arrowhead=1,
            font=dict(size=12, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
    
    return fig

def create_word_cloud(responses, language):
    """
    Create a word cloud visualization from the responses.
    
    Args:
        responses: List of text responses
        language: Language code ('en' for English, 'tr' for Turkish)
    
    Returns:
        Matplotlib figure
    """
    # Join all responses
    text = ' '.join(responses)
    
    # Define stopwords based on language
    stopwords = []
    if language == 'tr':
        # Common Turkish stopwords
        stopwords = [
            've', 'ile', 'bu', 'için', 'bir', 'da', 'de', 'ama', 'fakat', 'ki',
            'çünkü', 'sonra', 'önce', 'kadar', 'gibi', 'daha', 'kendi', 'her',
            'bazı', 'bütün', 'çok', 'defa', 'kez', 'ben', 'sen', 'o', 'biz',
            'siz', 'onlar', 'bana', 'sana', 'ona', 'bize', 'size', 'onlara'
        ]
    else:  # English
        # Common English stopwords
        stopwords = [
            'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 
            'by', 'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are',
            'from', 'at', 'as', 'your', 'have', 'more', 'has', 'if', 'my',
            'do', 'will', 'can', 'about', 'which', 'their', 'when', 'what'
        ]
    
    # Create and configure the word cloud
    wc = WordCloud(
        background_color='white',
        max_words=100,
        width=800,
        height=400,
        stopwords=set(stopwords),
        contour_width=1,
        contour_color='steelblue'
    )
    
    # Generate the word cloud
    wc.generate(text)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    
    return fig
