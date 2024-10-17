import gradio as gr
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import fitz  # PyMuPDF
from openai import OpenAI  # Add this import
import os  # Add this import
import logging  # Add this import


def get_openai_models():
    return ["gpt-3.5-turbo", "gpt-4"]

def get_hf_models():
    return ["Qwen/Qwen2.5-3B-Instruct", "HuggingFaceH4/zephyr-7b-beta", "mistralai/Mistral-7B-Instruct-v0.1"] + get_openai_models()

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def manual_rag(query, context, client, model):
    if model in get_openai_models():
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        return response.choices[0].message.content
    else:
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = client.text_generation(prompt, max_new_tokens=512)
        return response

def classic_rag(query, text, client, embedder, model):
    """
    Performs classic RAG by searching for relevant chunks in the provided text.

    Args:
        query (str): The user query.
        text (str): The text to search within (pre-summarized).
        client: The language model client.
        embedder (str): The embedder model name.
        model (str): The language model to use for generating the answer.

    Returns:
        tuple: Response from the model and the context used.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=embedder)
    db = FAISS.from_texts(chunks, embeddings)
    docs = db.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in docs])
    response = manual_rag(query, context, client, model)
    return response, context

def no_rag(query, client, model):
    if model in get_openai_models():
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            max_tokens=512
        )
        return response.choices[0].message.content
    else:
        response = client.text_generation(query, max_new_tokens=512)
        return response

def process_query(query, pdf_path, llm_choice, embedder_choice, use_manual_rag):
    """
    Processes the user query by performing RAG with optional summarization.

    Args:
        query (str): The user's question.
        pdf_path (UploadedFile | None): The uploaded PDF file.
        llm_choice (str): The selected language model.
        embedder_choice (str): The selected embedder model.
        use_manual_rag (str): Flag to use manual RAG.

    Returns:
        tuple: Responses from no RAG, manual RAG, classic RAG, full text, and context.
    """
    if llm_choice in get_openai_models():
        # Use the API key directly in the code (not recommended for production)
        api_key = os.getenv("OPENAI_API_KEY", "")
        client = OpenAI(api_key=api_key)
    else:
        client = InferenceClient(llm_choice)
    
    no_rag_response = no_rag(query, client, llm_choice)
    
    if pdf_path is None:
        return (
            no_rag_response,
            "RAG non utilisé (pas de fichier PDF)",
            "RAG non utilisé (pas de fichier PDF)",
            "Pas de fichier PDF fourni",
            "Pas de contexte extrait"
        )
    
    full_text = extract_text_from_pdf(pdf_path)
    
    # Generate summary of the full text
    summary_text = summarize_text(full_text, client, llm_choice)
    
    # RAG manuel seulement si choisi
    if use_manual_rag == "Oui":
        manual_rag_response = manual_rag(query, summary_text, client, llm_choice)
    else:
        manual_rag_response = "RAG manuel non utilisé"
    
    classic_rag_response, classic_rag_context = classic_rag(
        query, summary_text, client, embedder_choice, llm_choice
    )
    
    return (
        no_rag_response,
        manual_rag_response,
        classic_rag_response,
        summary_text,
        classic_rag_context
    )


def summarize_text(text, client, model):
    """
    Generates a summary of the provided text using the specified language model.

    Args:
        text (str): The text to summarize.
        client: The language model client (OpenAI or HuggingFace).
        model (str): The model name to use for summarization.

    Returns:
        str: The summarized text.
    """
    from transformers import AutoTokenizer

    try:
        # Initialize tokenizer to get model's maximum context length
        if model in get_openai_models():
            # OpenAI models have different token limits; adjust accordingly
            max_tokens = 4096  # Example limit for GPT-4
        else:
            tokenizer = AutoTokenizer.from_pretrained(model)
            max_tokens = tokenizer.model_max_length

        # Define a function to split text into chunks
        def split_into_chunks(text, max_tokens):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_tokens - 500,  # Reserve tokens for the summary
                chunk_overlap=100
            )
            return text_splitter.split_text(text)

        # Split the text into manageable chunks
        chunks = split_into_chunks(text, max_tokens)

        summaries = []
        for chunk in chunks:
            if model in get_openai_models():
                prompt = f"Please provide a concise summary of the following text:\n\n{chunk}"
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512
                )
                summary = response.choices[0].message.content.strip()
            else:
                prompt = f"Please provide a concise summary of the following text:\n\n{chunk}"
                response = client.text_generation(prompt, max_new_tokens=150)
                summary = response[0]['generated_text'].strip()
            
            summaries.append(summary)

        # Combine all chunk summaries into a final summary
        if model in get_openai_models():
            final_prompt = "Please provide a concise summary of the following summaries:\n\n" + "\n".join(summaries)
            final_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=512
            )
            final_summary = final_response.choices[0].message.content.strip()
        else:
            final_prompt = "Please provide a concise summary of the following summaries:\n\n" + "\n".join(summaries)
            final_response = client.text_generation(final_prompt, max_new_tokens=150)
            final_summary = final_response[0]['generated_text'].strip()

        return final_summary

    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        return "Résumé non disponible en raison d'une erreur."


iface = gr.Interface(
    fn=process_query,
    inputs=[
        gr.Textbox(label="Votre question"),
        gr.File(label="Chargez un nouveau PDF"),
        gr.Dropdown(choices=get_hf_models(), label="Choisissez le LLM", value="HuggingFaceH4/zephyr-7b-beta"),
        gr.Dropdown(
            choices=["sentence-transformers/all-MiniLM-L6-v2", "nomic-ai/nomic-embed-text-v1.5"],
            label="Choisissez l'Embedder",
            value="sentence-transformers/all-MiniLM-L6-v2"
        ),
        gr.Dropdown(choices=["Oui", "Non"], label="Utiliser RAG manuel ?", value="Non")
    ],
    outputs=[
        gr.Textbox(label="Réponse sans RAG"),
        gr.Textbox(label="Réponse avec RAG manuel"),
        gr.Textbox(label="Réponse avec RAG classique"),
        gr.Textbox(label="Résumé du PDF (pour RAG manuel)", lines=10),
        gr.Textbox(label="Contexte extrait (pour RAG classique)", lines=10)
    ],
    title="Tutoriel RAG - Comparaison des méthodes",
    description="Posez une question sur le contenu d'un PDF et comparez les réponses obtenues avec différentes méthodes de RAG.",
    theme="default"
)

if __name__ == "__main__":
    iface.launch()
