import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from sentence_transformers import CrossEncoder

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX")

# 🔥 Cross-encoder reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query, docs, top_k=3):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in ranked[:top_k]]


def build_context_with_citations(docs):
    context = []
    citations = []

    for d in docs:
        file = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page", "N/A")

        citations.append({
            "file": file,
            "page": page
        })

        context.append(
            f"[Source: {file}, Page: {page}]\n{d.page_content}"
        )

    return "\n\n".join(context), citations


def get_rag_components():

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    # 🔍 retrieve more initially
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template="""
You are an expert assistant.

Use the chat history and document context to answer.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        streaming=True
    )

    return retriever, prompt, llm
