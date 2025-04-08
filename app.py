import streamlit as st
import os
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uuid

# Ensure backend directories exist
OPINION_PDF_DIR = 'backend/opinion_pdfs'
DATA_PDF_DIR = 'backend/data_pdfs'

# Create directories if they don't exist
os.makedirs(OPINION_PDF_DIR, exist_ok=True)
os.makedirs(DATA_PDF_DIR, exist_ok=True)

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Function to extract text from PDF
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

# Function for opinion-based Q&A
def opinion_based_qa(query, opinion_text):
    # Prepare prompt for opinion-based retrieval
    prompt = f"""
    Context: {opinion_text}
    
    Question: {query}
    
    Please provide an answer based on the subjective opinions in the context. 
    If the context doesn't directly answer the question, 
    explain why and provide insights from the available subjective perspectives.
    """
    
    # Generate response using OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant interpreting subjective opinions about naturalness."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Function for data-driven Q&A
def data_driven_qa(query, data_text):
    # Prepare prompt for data-driven retrieval
    prompt = f"""
    Scientific Data Context: {data_text}
    
    Question: {query}
    
    Please provide a data-driven, objective answer based on the scientific context. 
    Use quantitative insights and empirical observations to support your response.
    """
    
    # Generate response using OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a scientific assistant providing objective, data-driven insights."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Function to save uploaded PDF
def save_uploaded_pdf(uploaded_file, directory):
    # Generate a unique filename
    unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
    file_path = os.path.join(directory, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

# Streamlit App
def main():
    st.title("Naturalness Exploration: Subjective vs Objective Approaches")
    
    # Sidebar for navigation
    app_mode = st.sidebar.selectbox(
        "Choose Exploration Path",
        ["Home", "Subjective Opinions", "Data-Driven Analysis"]
    )
    
    # Home page
    if app_mode == "Home":
        st.write("""
        # Naturalness Research Explorer
        
        This application offers two unique approaches to understanding naturalness:
        
        1. **Subjective Opinions Path**: 
           - Explore naturalness through human perceptions
           - Based on questionnaire and interview insights
        
        2. **Data-Driven Analysis Path**:
           - Quantitative approach to understanding naturalness
           - Leveraging empirical data and scientific observations
        
        Choose your path in the sidebar and start exploring!
        """)
    
    # Subjective Opinions Path
    elif app_mode == "Subjective Opinions":
        st.header("Naturalness through Subjective Opinions")
        
        # Upload opinion PDF
        opinion_pdf = st.file_uploader("Upload PDF with Subjective Opinions", type=['pdf'])
        
        if opinion_pdf is not None:
            # Save uploaded PDF to backend
            pdf_path = save_uploaded_pdf(opinion_pdf, OPINION_PDF_DIR)
            
            # Extract text
            opinion_text = extract_pdf_text(pdf_path)
            
            # Query input
            query = st.text_input("Ask a question about naturalness based on opinions")
            
            if query:
                with st.spinner('Analyzing subjective perspectives...'):
                    answer = opinion_based_qa(query, opinion_text)
                    st.write("### Insight from Subjective Perspectives")
                    st.write(answer)
    
    # Data-Driven Path
    elif app_mode == "Data-Driven Analysis":
        st.header("Naturalness through Quantitative Analysis")
        
        # Upload data PDF
        data_pdf = st.file_uploader("Upload PDF with Scientific Data", type=['pdf'])
        
        if data_pdf is not None:
            # Save uploaded PDF to backend
            pdf_path = save_uploaded_pdf(data_pdf, DATA_PDF_DIR)
            
            # Extract text
            data_text = extract_pdf_text(pdf_path)
            
            # Query input
            query = st.text_input("Ask a data-driven question about naturalness")
            
            if query:
                with st.spinner('Analyzing scientific data...'):
                    answer = data_driven_qa(query, data_text)
                    st.write("### Data-Driven Insight")
                    st.write(answer)

# Run the app
if __name__ == "__main__":
    main()
import streamlit as st
import os
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uuid

# Function to validate OpenAI API key
def validate_openai_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        # Attempt a simple API call to verify the key
        client.models.list()
        return True
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
        return False

# Function to extract text from PDF
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

# Function for opinion-based Q&A
def opinion_based_qa(query, opinion_text, api_key):
    # Initialize client with user's API key
    client = OpenAI(api_key=api_key)
    
    # Prepare prompt for opinion-based retrieval
    prompt = f"""
    Context: {opinion_text}
    
    Question: {query}
    
    Please provide an answer based on the subjective opinions in the context. 
    If the context doesn't directly answer the question, 
    explain why and provide insights from the available subjective perspectives.
    """
    
    # Generate response using OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant interpreting subjective opinions about naturalness."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Function for data-driven Q&A
def data_driven_qa(query, data_text, api_key):
    # Initialize client with user's API key
    client = OpenAI(api_key=api_key)
    
    # Prepare prompt for data-driven retrieval
    prompt = f"""
    Scientific Data Context: {data_text}
    
    Question: {query}
    
    Please provide a data-driven, objective answer based on the scientific context. 
    Use quantitative insights and empirical observations to support your response.
    """
    
    # Generate response using OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a scientific assistant providing objective, data-driven insights."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Function to save uploaded PDF
def save_uploaded_pdf(uploaded_file, directory):
    # Generate a unique filename
    unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
    file_path = os.path.join(directory, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

# Streamlit App
def main():
    # Set page title and favicon
    st.set_page_config(page_title="Naturalness Expert", page_icon="🔬")
    
    # Ensure backend directories exist
    OPINION_PDF_DIR = 'backend/opinion_pdfs'
    DATA_PDF_DIR = 'backend/data_pdfs'
    os.makedirs(OPINION_PDF_DIR, exist_ok=True)
    os.makedirs(DATA_PDF_DIR, exist_ok=True)
    
    # API Key Input
    st.sidebar.header("OpenAI API Configuration")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    
    # Validate API Key
    api_key_valid = False
    if api_key:
        api_key_valid = validate_openai_key(api_key)
        if api_key_valid:
            st.sidebar.success("API Key is valid!")
        else:
            st.sidebar.warning("Please enter a valid OpenAI API Key")
    
    # Main Title
    st.title("Naturalness Expert: Exploring Perceptions and Data")
    
    # Sidebar for navigation
    app_mode = st.sidebar.selectbox(
        "Choose Exploration Path",
        ["Home", "Subjective Opinions", "Data-Driven Analysis"]
    )
    
    # Home page
    if app_mode == "Home":
        st.write("""
        # Welcome to Naturalness Expert
        
        This application offers two unique approaches to understanding naturalness:
        
        1. **Subjective Opinions Path**: 
           - Explore naturalness through human perceptions
           - Based on questionnaire and interview insights
        
        2. **Data-Driven Analysis Path**:
           - Quantitative approach to understanding naturalness
           - Leveraging empirical data and scientific observations
        
        ## Getting Started
        - Enter your OpenAI API Key in the sidebar
        - Choose your exploration path
        - Upload a PDF and start asking questions!
        
        ### Note
        You'll need an active OpenAI API Key to use this application.
        """)
    
    # Check if API key is valid before proceeding
    if not api_key_valid:
        st.warning("Please enter a valid OpenAI API Key in the sidebar to continue.")
        return
    
    # Subjective Opinions Path
    elif app_mode == "Subjective Opinions":
        st.header("Naturalness through Subjective Opinions")
        
        # Upload opinion PDF
        opinion_pdf = st.file_uploader("Upload PDF with Subjective Opinions", type=['pdf'])
        
        if opinion_pdf is not None:
            # Save uploaded PDF to backend
            pdf_path = save_uploaded_pdf(opinion_pdf, OPINION_PDF_DIR)
            
            # Extract text
            opinion_text = extract_pdf_text(pdf_path)
            
            # Query input
            query = st.text_input("Ask a question about naturalness based on opinions")
            
            if query:
                with st.spinner('Analyzing subjective perspectives...'):
                    try:
                        answer = opinion_based_qa(query, opinion_text, api_key)
                        st.write("### Insight from Subjective Perspectives")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
    
    # Data-Driven Path
    elif app_mode == "Data-Driven Analysis":
        st.header("Naturalness through Quantitative Analysis")
        
        # Upload data PDF
        data_pdf = st.file_uploader("Upload PDF with Scientific Data", type=['pdf'])
        
        if data_pdf is not None:
            # Save uploaded PDF to backend
            pdf_path = save_uploaded_pdf(data_pdf, DATA_PDF_DIR)
            
            # Extract text
            data_text = extract_pdf_text(pdf_path)
            
            # Query input
            query = st.text_input("Ask a data-driven question about naturalness")
            
            if query:
                with st.spinner('Analyzing scientific data...'):
                    try:
                        answer = data_driven_qa(query, data_text, api_key)
                        st.write("### Data-Driven Insight")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

# Run the app
if __name__ == "__main__":
    main()
import streamlit as st
import os
from openai import OpenAI
from PyPDF2 import PdfReader
import uuid

# Function to validate OpenAI API key
def validate_openai_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        # Attempt a simple API call to verify the key
        client.models.list()
        return True
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
        return False

# Function to extract text from PDF
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

# Function for opinion-based Q&A
def opinion_based_qa(query, opinion_text, api_key):
    # Initialize client with user's API key
    client = OpenAI(api_key=api_key)
    
    # Prepare prompt for opinion-based retrieval
    prompt = f"""
    Context: {opinion_text}
    
    Question: {query}
    
    Please provide an answer based on the subjective opinions in the context. 
    If the context doesn't directly answer the question, 
    explain why and provide insights from the available subjective perspectives.
    """
    
    # Generate response using OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant interpreting subjective opinions about naturalness."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Function for data-driven Q&A
def data_driven_qa(query, data_text, api_key):
    # Initialize client with user's API key
    client = OpenAI(api_key=api_key)
    
    # Prepare prompt for data-driven retrieval
    prompt = f"""
    Scientific Data Context: {data_text}
    
    Question: {query}
    
    Please provide a data-driven, objective answer based on the scientific context. 
    Use quantitative insights and empirical observations to support your response.
    """
    
    # Generate response using OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a scientific assistant providing objective, data-driven insights."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Streamlit App
def main():
    # Set page title and favicon
    st.set_page_config(page_title="Naturalness Expert", page_icon="🔬")
    
    # Predefined PDFs in backend
    OPINION_PDF_PATH = 'backend/opinion_pdfs/naturalness_opinions.pdf'
    DATA_PDF_PATH = 'backend/data_pdfs/naturalness_data.pdf'
    
    # Ensure backend directories exist
    os.makedirs(os.path.dirname(OPINION_PDF_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(DATA_PDF_PATH), exist_ok=True)
    
    # API Key Input
    st.sidebar.header("OpenAI API Configuration")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    
    # Validate API Key
    api_key_valid = False
    if api_key:
        api_key_valid = validate_openai_key(api_key)
        if api_key_valid:
            st.sidebar.success("API Key is valid!")
        else:
            st.sidebar.warning("Please enter a valid OpenAI API Key")
    
    # Main Title
    st.title("Naturalness Expert: Exploring Perceptions and Data")
    
    # Sidebar for navigation
    app_mode = st.sidebar.selectbox(
        "Choose Exploration Path",
        ["Home", "Subjective Opinions", "Data-Driven Analysis"]
    )
    
    # Home page
    if app_mode == "Home":
        st.write("""
        # Welcome to Naturalness Expert
        
        This application offers two unique approaches to understanding naturalness:
        
        1. **Subjective Opinions Path**: 
           - Explore naturalness through human perceptions
           - Based on predefined questionnaire and interview insights
        
        2. **Data-Driven Analysis Path**:
           - Quantitative approach to understanding naturalness
           - Leveraging predefined empirical data
        
        ## Getting Started
        - Enter your OpenAI API Key in the sidebar
        - Choose your exploration path
        - Start asking questions!
        
        ### Note
        You'll need an active OpenAI API Key to use this application.
        """)
    
    # Check if API key is valid before proceeding
    if not api_key_valid:
        st.warning("Please enter a valid OpenAI API Key in the sidebar to continue.")
        return
    
    # Subjective Opinions Path
    elif app_mode == "Subjective Opinions":
        st.header("Naturalness through Subjective Opinions")
        
        # Check if PDF exists
        if not os.path.exists(OPINION_PDF_PATH):
            st.error("Opinion PDF not found in backend.")
            return
        
        # Extract text from predefined PDF
        opinion_text = extract_pdf_text(OPINION_PDF_PATH)
        
        # Query input
        query = st.text_input("Ask a question about naturalness based on opinions")
        
        if query:
            with st.spinner('Analyzing subjective perspectives...'):
                try:
                    answer = opinion_based_qa(query, opinion_text, api_key)
                    st.write("### Insight from Subjective Perspectives")
                    st.write(answer)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    
    # Data-Driven Path
    elif app_mode == "Data-Driven Analysis":
        st.header("Naturalness through Quantitative Analysis")
        
        # Check if PDF exists
        if not os.path.exists(DATA_PDF_PATH):
            st.error("Data PDF not found in backend.")
            return
        
        # Extract text from predefined PDF
        data_text = extract_pdf_text(DATA_PDF_PATH)
        
        # Query input
        query = st.text_input("Ask a data-driven question about naturalness")
        
        if query:
            with st.spinner('Analyzing scientific data...'):
                try:
                    answer = data_driven_qa(query, data_text, api_key)
                    st.write("### Data-Driven Insight")
                    st.write(answer)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Run the app
if __name__ == "__main__":
    main()
