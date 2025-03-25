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

def create_likert_scale_visualization(responses, language):
    """
    Create a Likert scale visualization of response sentiment.
    
    Args:
        responses: List of text responses
        language: Language code ('en' for English, 'tr' for Turkish)
    
    Returns:
        Plotly figure
    """
    # Calculate a simple sentiment score for each response
    sentiment_scores = []
    
    # Simple keyword-based sentiment analysis
    positive_words = {
        'en': ['good', 'great', 'excellent', 'positive', 'happy', 'like', 'love', 'best', 'better', 'improved'],
        'tr': ['iyi', 'güzel', 'harika', 'olumlu', 'mutlu', 'sevmek', 'aşk', 'en iyi', 'daha iyi', 'gelişmiş']
    }
    
    negative_words = {
        'en': ['bad', 'poor', 'terrible', 'negative', 'unhappy', 'dislike', 'hate', 'worst', 'worse', 'problem'],
        'tr': ['kötü', 'zayıf', 'korkunç', 'olumsuz', 'mutsuz', 'sevmemek', 'nefret', 'en kötü', 'daha kötü', 'sorun']
    }
    
    lang_key = 'tr' if language == 'tr' else 'en'
    
    for response in responses:
        response_lower = response.lower()
        pos_count = sum(1 for word in positive_words[lang_key] if word in response_lower)
        neg_count = sum(1 for word in negative_words[lang_key] if word in response_lower)
        
        # Calculate sentiment score between -2 (very negative) and 2 (very positive)
        if pos_count == 0 and neg_count == 0:
            score = 0  # neutral
        else:
            score = (pos_count - neg_count) / max(1, pos_count + neg_count) * 2
        
        sentiment_scores.append(score)
    
    # Create bins for Likert scale
    bins = [-2, -1, 0, 1, 2]
    bin_labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'] if lang_key == 'en' else ['Çok Olumsuz', 'Olumsuz', 'Nötr', 'Olumlu', 'Çok Olumlu']
    counts = [0, 0, 0, 0, 0]  # Initialize counts for each bin
    
    # Count responses in each bin
    for score in sentiment_scores:
        # Convert score to bin index (0-4)
        bin_idx = min(4, max(0, int((score + 2) / 4 * 5)))
        counts[bin_idx] += 1
    
    # Create a bar chart for Likert scale
    fig = go.Figure(data=[
        go.Bar(
            x=bin_labels,
            y=counts,
            marker_color=['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'],
            text=counts,
            textposition='auto'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title="Response Sentiment Distribution" if lang_key == 'en' else "Cevap Duygu Dağılımı",
        xaxis_title="Sentiment" if lang_key == 'en' else "Duygu",
        yaxis_title="Number of Responses" if lang_key == 'en' else "Cevap Sayısı",
        height=500
    )
    
    return fig
