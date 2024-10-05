import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

st.set_page_config(page_title="Coding video summarizer", 
                   page_icon="⌨️", layout="wide")

st.title("🦜 LangChain: Summarize Coding Videos From YouTube or Website")
st.subheader("Summarize any URL")

with st.sidebar:
    st.header("🔐 Enter your Groq API Key")
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    st.header("🌐 Enter the URL (YouTube or Website)")
    generic_url = st.text_input("URL", placeholder="https://example.com")

prompt_template = """
You are a coding teacher helping students solve DSA questions. Your job is to summarize the youtube video 
which is actually explanation of solutions to some DSA question and explain it to weak students. Also provide code/codes in C++:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("✨ Summarize the Content from YT or Website"):
    if not groq_api_key.strip():
        st.error("🚨 Please enter your Groq API Key.")
    elif not generic_url.strip():
        st.error("🚨 Please provide a URL (YouTube or Website).")
    elif not validators.url(generic_url):
        st.error("🚨 Please enter a valid URL.")
    else:
        try:
            with st.spinner("⏳ Summarizing content..."):
                llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

                if "youtube.com" in generic_url:
                    st.info("🎥 Detected a YouTube URL. Loading video content...")
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    st.info("🌐 Detected a website URL. Loading webpage content...")
                    loader = UnstructuredURLLoader(
                        urls=[generic_url], 
                        ssl_verify=True,  
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}  # Updated user-agent
                    )
                
                docs = loader.load()

                if not docs:
                    st.error("🚨 Could not retrieve content from the URL. Please check if the URL is correct.")
                else:
                    st.write("### Loaded Content:")
                    st.write(docs)  # Display the loaded documents for debugging

                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

                    output_summary = chain.invoke({"input_documents": docs})
                    
                    st.success("✅ Summary Generated Successfully!")
                    st.markdown(f"### 📄 Summary:\n{output_summary['output_text']}")
                    
        except Exception as e:
            st.exception(f"⚠️ Failed to load content: {str(e)}")
