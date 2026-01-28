from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag_chain import (
    get_rag_components,
    build_context_with_citations,
    rerank_documents
)

app = FastAPI(title="Insurance RAG Assistant")

retriever, prompt, llm = get_rag_components()

chat_memory = {}


class Question(BaseModel):
    query: str
    session_id: str = "default"


@app.post("/ask")
def ask_question(data: Question):

    history = chat_memory.get(data.session_id, [])

    # 🔍 retrieve more docs
    docs = retriever.invoke(data.query)

    # 🎯 rerank
    top_docs = rerank_documents(data.query, docs)

    context, citations = build_context_with_citations(top_docs)

    chat_history = "\n".join(history)

    final_prompt = prompt.format(
        chat_history=chat_history,
        context=context,
        question=data.query
    )

    answer = llm.invoke(final_prompt).content

    history.append(f"User: {data.query}")
    history.append(f"AI: {answer}")
    chat_memory[data.session_id] = history

    return {
        "answer": answer,
        "citations": citations
    }


@app.post("/ask-stream")
def ask_stream(data: Question):

    history = chat_memory.get(data.session_id, [])

    docs = retriever.invoke(data.query)

    # 🔥 rerank here also
    top_docs = rerank_documents(data.query, docs)

    context, citations = build_context_with_citations(top_docs)

    chat_history = "\n".join(history)

    final_prompt = prompt.format(
        chat_history=chat_history,
        context=context,
        question=data.query
    )

    def token_generator():
        collected = ""
        for chunk in llm.stream(final_prompt):
            collected += chunk.content
            yield chunk.content

        history.append(f"User: {data.query}")
        history.append(f"AI: {collected}")
        chat_memory[data.session_id] = history

    return StreamingResponse(
        token_generator(),
        media_type="text/plain"
    )
