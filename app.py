import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit APP Configuration
st.set_page_config(page_title="Coding video summarizer", 
                   page_icon="⌨️", layout="wide")

# Title and Subtitle        
st.title("🦜 LangChain: Summarize Coding Videos From YouTube or Website")
st.subheader("Summarize any URL")

## Sidebar: Get the Groq API Key and URL (YouTube or website) to be summarized
with st.sidebar:
    st.header("🔐 Enter your Groq API Key")
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    st.header("🌐 Enter the URL (YouTube or Website)")
    generic_url = st.text_input("URL", placeholder="https://example.com")

## Define the Prompt Template for Summarization
prompt_template = """
You are a coding teacher helping students solve DSA questions. Your job is to summarize the youtube video 
which is actually explanation of solutions to some DSA question and explain it to weak students. Also provide code/codes in C++:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

## Summarize Button and Action
if st.button("✨ Summarize the Content from YT or Website"):
    # Validate the API Key and URL inputs
    if not groq_api_key.strip():
        st.error("🚨 Please enter your Groq API Key.")
    elif not generic_url.strip():
        st.error("🚨 Please provide a URL (YouTube or Website).")
    elif not validators.url(generic_url):
        st.error("🚨 Please enter a valid URL.")
    else:
        try:
            with st.spinner("⏳ Summarizing content..."):
                # Initialize the Gemma model using Groq API
                llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

                # Loading the website or YouTube video data
                if "youtube.com" in generic_url:
                    st.info("🎥 Detected a YouTube URL. Loading video content...")
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    st.info("🌐 Detected a website URL. Loading webpage content...")
                    loader = UnstructuredURLLoader(
                        urls=[generic_url], 
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                
                # Attempt to load the documents (video transcripts or webpage data)
                docs = loader.load()

                # Check if the documents are successfully loaded
                if not docs:
                    st.error("🚨 Could not retrieve content from the URL. Please check if the URL is correct.")
                else:
                    st.write("### Loaded Content:")
                    st.write(docs)  # Display the loaded documents for debugging

                    # Chain for Summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

                    # Use the chain to summarize the loaded documents
                    output_summary = chain.invoke({"input_documents": docs})
                    
                    st.success("✅ Summary Generated Successfully!")
                    st.markdown(f"### 📄 Summary:\n{output_summary['output_text']}")
                    
        except Exception as e:
            st.exception(f"⚠️ Exception occurred: {e}")
