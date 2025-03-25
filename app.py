import streamlit as st
import pandas as pd
import os
from langdetect import detect, LangDetectException
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from utils.gemini_utils import (
    initialize_gemini_model,
    extract_topics,
    generate_embeddings,
    generate_cluster_summaries
)
from utils.visualization import (
    create_topic_visualization,
    create_cluster_visualization,
    create_likert_scale_visualization
)
from utils.clustering import perform_clustering
from utils.data_storage import save_analysis, load_previous_analyses

# Set page config
st.set_page_config(
    page_title="Multilingual Q&A Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'analyses' not in st.session_state:
    st.session_state.analyses = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

def detect_language(text):
    """Detect language of input text."""
    try:
        return detect(text)
    except LangDetectException:
        return "en"  # Default to English if detection fails

def main():
    # Initialize Gemini model
    gemini_model = initialize_gemini_model()
    
    # Page title and description
    st.title("Multilingual Q&A Analysis")
    st.markdown("Analyze open-ended responses in Turkish and English using Gemini 2.0 Flash")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["New Analysis", "Previous Analyses"])
    
    if page == "New Analysis":
        display_analysis_form(gemini_model)
    else:
        display_previous_analyses()

def display_analysis_form(gemini_model):
    """Display the form for new analysis input."""
    st.header("New Analysis")
    
    with st.form("analysis_form"):
        question = st.text_input("Enter the question:", placeholder="e.g., What challenges do you face in your work?")
        responses = st.text_area(
            "Enter responses (one per line):",
            height=200,
            placeholder="Response 1\nResponse 2\nResponse 3"
        )
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
        
        submitted = st.form_submit_button("Analyze Responses")
        
        if submitted and question and responses:
            with st.spinner("Analyzing responses..."):
                # Process the responses
                response_list = [r.strip() for r in responses.split('\n') if r.strip()]
                
                if not response_list:
                    st.error("No valid responses found. Please enter at least one response.")
                    return
                
                # Detect language (using the first non-empty response)
                language = detect_language(response_list[0])
                st.info(f"Detected language: {'Turkish' if language == 'tr' else 'English'}")
                
                # Extract topics using Gemini
                topics = extract_topics(gemini_model, question, response_list, language)
                
                # Generate embeddings for clustering
                embeddings = generate_embeddings(gemini_model, response_list)
                
                # Perform clustering
                clusters = perform_clustering(embeddings, n_clusters)
                
                # Create timestamp for this analysis
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Store the analysis results
                analysis_result = {
                    'timestamp': timestamp,
                    'question': question,
                    'responses': response_list,
                    'language': language,
                    'topics': topics,
                    'clusters': clusters,
                    'embeddings': embeddings.tolist()
                }
                
                # Save the analysis to session state and persistent storage
                st.session_state.analyses.append(analysis_result)
                st.session_state.current_analysis = analysis_result
                save_analysis(analysis_result)
                
                # Display results
                display_analysis_results(analysis_result, n_clusters)

def display_analysis_results(analysis, n_clusters):
    """Display the analysis results with visualizations."""
    st.header("Analysis Results")
    
    st.subheader("Question")
    st.write(analysis['question'])
    
    st.subheader("Language")
    st.write("Turkish" if analysis['language'] == 'tr' else "English")
    
    st.subheader("Responses")
    with st.expander("View all responses"):
        for i, response in enumerate(analysis['responses']):
            st.write(f"{i+1}. {response}")
    
    # Generate cluster summaries if they don't exist
    if 'cluster_summaries' not in analysis:
        with st.spinner("Generating cluster summaries..."):
            # Initialize Gemini model
            gemini_model = initialize_gemini_model()
            
            # Generate cluster summaries
            analysis['cluster_summaries'] = generate_cluster_summaries(
                gemini_model,
                analysis['question'],
                analysis['responses'],
                analysis['clusters'],
                n_clusters,
                analysis['language']
            )
            
            # Save the updated analysis
            save_analysis(analysis)
    
    # Display visualizations in tabs
    tab1, tab2, tab3 = st.tabs(["Topics", "Clusters", "Sentiment"])
    
    with tab1:
        st.subheader("Topic Analysis")
        topic_chart = create_topic_visualization(analysis['topics'])
        st.plotly_chart(topic_chart, use_container_width=True)
    
    with tab2:
        st.subheader("Response Clustering")
        
        # Display cluster summaries
        for i, summary in enumerate(analysis.get('cluster_summaries', [])):
            with st.expander(f"Cluster {i+1} Summary"):
                st.write(summary)
                
                # Get responses in this cluster
                cluster_responses = [analysis['responses'][j] for j in range(len(analysis['responses'])) 
                                    if analysis['clusters'][j] == i]
                
                # Show responses in this cluster
                if cluster_responses:
                    st.write("**Responses in this cluster:**")
                    for j, resp in enumerate(cluster_responses):
                        st.write(f"{j+1}. {resp}")
        
        # Display cluster visualization
        cluster_chart = create_cluster_visualization(
            analysis['embeddings'], 
            analysis['clusters'], 
            analysis['responses'],
            n_clusters
        )
        st.plotly_chart(cluster_chart, use_container_width=True)
    
    with tab3:
        st.subheader("Sentiment Analysis")
        sentiment_chart = create_likert_scale_visualization(analysis['responses'], analysis['language'])
        st.plotly_chart(sentiment_chart, use_container_width=True)

def display_previous_analyses():
    """Display the list of previous analyses."""
    st.header("Previous Analyses")
    
    # Load previous analyses from storage
    previous_analyses = load_previous_analyses()
    
    if not previous_analyses:
        st.info("No previous analyses found.")
        return
    
    # Display selection for previous analyses
    timestamps = [analysis['timestamp'] for analysis in previous_analyses]
    questions = [analysis['question'] for analysis in previous_analyses]
    
    selected_index = st.selectbox(
        "Select a previous analysis:",
        range(len(previous_analyses)),
        format_func=lambda i: f"{timestamps[i]} - {questions[i][:50]}{'...' if len(questions[i]) > 50 else ''}"
    )
    
    selected_analysis = previous_analyses[selected_index]
    
    # Display the selected analysis
    if st.button("Show Analysis"):
        st.session_state.current_analysis = selected_analysis
        n_clusters = len(set(selected_analysis['clusters']))
        display_analysis_results(selected_analysis, n_clusters)

if __name__ == "__main__":
    main()
