import os
import time

from google import genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please add it to your .env file."
        )

    return genai.Client(api_key=api_key)


def call_gemini(prompt: str) -> str:
    """
    Call Gemini with retry for temporary API errors.
    """

    client = get_llm()

    delays = [5, 10, 20]

    for attempt in range(4):

        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
                )

            time.sleep(2)

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except Exception as e:

            error_text = str(e).lower()

            is_retryable = (
                "429" in error_text
                or "rate limit" in error_text
                or "resource exhausted" in error_text
                or "500" in error_text
                or "503" in error_text
            )

            if not is_retryable:
                raise

            if attempt >= 3:
                raise RuntimeError(
                    "Gemini API request failed after multiple retries. "
                    "Please try again later."
                ) from e

            time.sleep(delays[attempt])


def format_docs(docs):
    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(
        vector_store,
        k=4
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting transcript
context provided below.

If the answer is not found in the context, say:

"I could not find this information in the meeting transcript."

Always be concise and precise.
If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""
            ),
            ("human", "{question}"),
        ]
    )

    def ask_with_context(inputs):

        context = format_docs(
            retriever.invoke(inputs["question"])
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": inputs["question"]
            }
        )

        return call_gemini(
            final_prompt.to_string()
        )

    rag_chain = (
        {
            "question": RunnablePassthrough()
        }
        | RunnableLambda(ask_with_context)
    )

    return rag_chain


def load_rag_chain():

    vector_store = load_vector_store()

    retriever = get_retriever(
        vector_store
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting transcript
context provided below.

If the answer is not found in the context, say:

"I could not find this information in the meeting transcript."

Always be concise and precise.
If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""
            ),
            ("human", "{question}"),
        ]
    )

    def ask_with_context(inputs):

        context = format_docs(
            retriever.invoke(inputs["question"])
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": inputs["question"]
            }
        )

        return call_gemini(
            final_prompt.to_string()
        )

    rag_chain = (
        {
            "question": RunnablePassthrough()
        }
        | RunnableLambda(ask_with_context)
    )

    return rag_chain


def ask_question(
    rag_chain,
    question: str
) -> str:

    print(f"Question: {question}")

    answer = rag_chain.invoke(
        question
    )

    print(f"Answer: {answer}")

    return answer