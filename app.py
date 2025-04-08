import streamlit as st
import os
from openai import OpenAI
from PyPDF2 import PdfReader

# Function to validate OpenAI API key
def validate_openai_key(api_key):
    try:
        # Remove any whitespace from the API key
        api_key = api_key.strip()
        
        # Create OpenAI client with minimal arguments
        client = OpenAI(api_key=api_key)
        
        # Attempt to list models to verify the key
        client.models.list()
        return True
    except Exception as e:
        st.error(f"API Key Validation Error: {e}")
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
    # Create client with stripped API key
    client = OpenAI(api_key=api_key.strip())
    
    # Truncate text to avoid context length issues
    truncated_text = opinion_text[:10000] + "..." if len(opinion_text) > 10000 else opinion_text
    
    # Prepare prompt for opinion-based retrieval
    prompt = f"""
    Context: {truncated_text}
    
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
    # Create client with stripped API key
    client = OpenAI(api_key=api_key.strip())
    
    # Truncate text to avoid context length issues
    truncated_text = data_text[:10000] + "..." if len(data_text) > 10000 else data_text
    
    # Prepare prompt for data-driven retrieval
    prompt = f"""
    Scientific Data Context: {truncated_text}
    
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
    
    # Main Title
    st.title("Naturalness Expert: Exploring Perceptions and Data")
    
    # API Key Input
    st.header("OpenAI API Configuration")
    api_key = st.text_input("Enter your OpenAI API Key", type="password", 
                             help="Your API key is required to use the OpenAI language model.")
    
    # Validate API Key
    api_key_valid = False
    if api_key:
        api_key_valid = validate_openai_key(api_key)
    
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
        - Enter your OpenAI API Key above
        - Choose your exploration path
        - Start asking questions!
        
        ### Note
        You'll need an active OpenAI API Key to use this application.
        """)
    
    # Check if API key is valid before proceeding
    if not api_key_valid:
        st.warning("Please enter a valid OpenAI API Key to continue.")
        return
    
    # Subjective Opinions Path
    elif app_mode == "Subjective Opinions":
        st.header("Naturalness through Subjective Opinions")
        
        # Check if PDF exists
        if not os.path.exists(OPINION_PDF_PATH):
            st.warning("Default opinion PDF not found in backend.")
            
            # Allow user to upload their own PDF
            uploaded_file = st.file_uploader("Upload your own opinions PDF", type="pdf")
            
            if uploaded_file is not None:
                # Save the uploaded file temporarily
                with open("temp_opinion.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract text from uploaded PDF
                opinion_text = extract_pdf_text("temp_opinion.pdf")
                
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
            else:
                st.info("Please upload a PDF with subjective opinions about naturalness.")
        else:
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
            st.warning("Default data PDF not found in backend.")
            
            # Allow user to upload their own PDF
            uploaded_file = st.file_uploader("Upload your own data PDF", type="pdf")
            
            if uploaded_file is not None:
                # Save the uploaded file temporarily
                with open("temp_data.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract text from uploaded PDF
                data_text = extract_pdf_text("temp_data.pdf")
                
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
            else:
                st.info("Please upload a PDF with quantitative data about naturalness.")
        else:
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
