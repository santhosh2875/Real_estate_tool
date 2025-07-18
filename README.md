🏡 RealEstate Research Tool

A powerful, AI-driven research tool to analyze and extract insights from real estate news articles using LLMs and LangChain. Built with the latest GenAI stack: **Groq (LLaMA3)**, **LangChain**, **ChromaDB**, and **HuggingFace embeddings**, this app helps users ask intelligent questions about real estate articles and get precise, source-backed answers.

> 🔄 While designed for the real estate domain, this tool can easily be extended to support any industry or topic.

![App Screenshot](resources/image.png)

---

## ✨ Features

- 🔗 Input article URLs through the sidebar in a simple Streamlit UI
- 📄 Load and process web content using `UnstructuredURLLoader`
- 🧠 Embed article text using HuggingFace’s `all-MiniLM-L6-v2` model
- 🗂️ Store and retrieve documents via **ChromaDB**
- 🤖 Query articles using **LLaMA3 (via Groq)** and receive answers with cited sources
- ✅ Supports multi-article processing and semantic understanding

---

## ⚙️ Setup Instructions

1. **Install all dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Groq credentials**:  
   Create a `.env` file in your project root with the following:
   ```env
   GROQ_MODEL=llama-3-70b-8192
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Start the Streamlit app**:
   ```bash
   streamlit run main.py
   ```

---

## 🧪 Example Use Case

Try loading the following real estate news articles:

- https://www.barrons.com/articles/zillow-real-estate-housing-market-ccca0d1c
- https://www.marketwatch.com/story/why-these-homeowners-say-the-15-year-mortgage-is-the-most-underrated-offering-in-real-estate-right-now-91deafd1
-https://www.marketwatch.com/story/why-these-homeowners-say-the-15-year-mortgage-is-the-most-underrated-offering-in-real-estate-right-now-91deafd1

Then ask questions like:

-“What’s driving Europe’s commercial real estate stagnation in mid‑2025?”
-“Why are some homeowners choosing 15‑year mortgages now?”



---

## 🛠️ Tech Stack

- [LangChain 0.3.26](https://docs.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Groq (LLaMA3)](https://groq.com/)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [ChromaDB](https://www.trychroma.com/)
- [Unstructured.io](https://www.unstructured.io/) for URL content extraction

---

## 🙋‍♂️ Built By

**Santhosh Kumar**  
Final Year B.Sc. Computer Science, Loyola College  
Aspiring GenAI / LLM Engineer

