import sys
import openai
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json
import requests
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Add custom style
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap'); 

* {
    font-family: 'Inter', sans-serif; 
}
h1 {
    font-family: "Inter", sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Dictionary with main category names as keys
categories_data = {
    'sports_stats': 'Statistics and performance metrics of professional athletes and sports personalities',
    'entertainment_stats': 'Statistical insights into the achievements and activities of personalities in the entertainment industry',
    'current_events': 'Up-to-the-minute coverage of breaking news on global current events',
    'entertainment_updates': 'Latest developments, news, and highlights about your favorite movies, TV shows, and celebrities',
    'research_papers': 'Exploration of research papers and scholarly articles on various topics',
}



# Set up ANYSCALE API key
ANYSCALE_ENDPOINT_TOKEN = "YOUR_ANYSCALE_API_KEY"

oai_client = openai.OpenAI(
    base_url="https://api.endpoints.anyscale.com/v1",
    api_key=ANYSCALE_ENDPOINT_TOKEN,
)


# Set up Streamlit session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "mistralai/Mistral-7B-Instruct-v0.1"

if "messages" not in st.session_state:
    st.session_state.messages = []


# Set up SentenceTransformer model
model_HF_name = "Sakil/sentence_similarity_semantic_search"
model_HF = SentenceTransformer(model_HF_name)

# Set up Google Generative AI API key
genai.configure(api_key="GEMINI_API_KEY")

# Set up Bing News API subscription key
BING_NEWS_API_KEY = "YOUR_BING_NEWS_API_KEY"  # Replace with your Bing News API key

# Define a function to calculate the embeddings of the input text
def model_MM(input_text):
    categories_data_embeddings_MM_ = genai.embed_content(
        model="models/embedding-001",
        content=input_text,
        task_type="retrieval_document",
        title="Embedding of inputs")
    return categories_data_embeddings_MM_['embedding']

# Define a function to find the best result based on both MM and HF models
def find_best_result_MM(user_prompt, embedding_model_MM, categories_data_embeddings_MM):
    # Convert user prompt to embedding vector
    user_prompt_embedding_MM = embedding_model_MM([user_prompt])

    # Calculate cosine similarity with categories data embeddings
    similarity_scores_MM = cosine_similarity(user_prompt_embedding_MM, categories_data_embeddings_MM)

    # Find the index of the best result with the highest score for both models
    best_result_index_MM = np.argmax(similarity_scores_MM)

    # Get the best first result with the highest score for both models
    best_result_MM = list(categories_data.keys())[best_result_index_MM]

    return best_result_MM

# Define a function to query ArXiv for information
def query_arxiv(search_query, start=0, max_results=2):
    base_url = "http://export.arxiv.org/api/query?"
    query_params = {
        'search_query': f'all:{search_query}',
        'start': start,
        'max_results': max_results
    }

    response = requests.get(base_url, params=query_params)
    summaries = re.findall('<summary>(.*?)</summary>', response.text, re.DOTALL)
    merged_summary = ' '.join(summary.replace('\n', ' ').replace('\t', ' ').strip() for summary in summaries)
    return merged_summary

# Define a function to extract keywords from text
def extract_keywords(text):
    merged_words = ' '.join([word for word in word_tokenize(text) if word.lower() not in set(stopwords.words('english'))])
    merged_words = re.sub('[\?\!\,\.\']', '', merged_words)
    merged_words = merged_words.strip()
    return merged_words

# Define a function to get current information from Bing News
def get_current_information(query):
    search_url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": BING_NEWS_API_KEY}
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()

    descriptions = [item.get('description', '') for item in json.loads(json.dumps(response.json())).get('value', [])]
    cleaned_description = re.sub('<.*?>', '', '\n'.join(descriptions))
    cleaned_description = re.sub('\.\.\.', '', cleaned_description)
    cleaned_description = re.sub('&#39;', "'", cleaned_description)

    return cleaned_description

# calculate embedding of sentence2 and categories_data
categories_data_embeddings_MM_ = model_MM(list(categories_data.values()))


# Display chat history
for message in st.session_state.messages:
    with st.empty():
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):

    if find_best_result_MM(prompt, model_MM, categories_data_embeddings_MM_) == 'research_papers':
        information = query_arxiv(prompt)
    elif find_best_result_MM(prompt, model_MM, categories_data_embeddings_MM_) == 'current_events':
        information = get_current_information(extract_keywords(prompt))

    # pass this information in the prompt and get the answer
    prompt_template = f'''
    {information}

    answer the following question in a detailed manner based on the above information
    {prompt}
    '''

    st.session_state.messages.append({"role": "user", "content": prompt_template})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in oai_client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ], stream=True
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})