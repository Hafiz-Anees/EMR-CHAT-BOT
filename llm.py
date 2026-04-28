import os
from groq import Groq
from refine_query import refine_query
from emrchains_chat_bot import retriever

groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

def ask_rag(question):
    updated_question = refine_query(question)
    docs = retriever.invoke(updated_question)
    context = "\n\n".join([doc.page_content for doc in docs])

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """
                You are an AI assistant for EMR Chain.

                Instructions:
                - Answer the user's question directly and concisely.
                - Do NOT introduce yourself unless explicitly asked (e.g., "Who are you?").
                - if the query is hi please response like this (e.g , "Hello. How can I assist you today?")
                - Do NOT include unnecessary phrases like "According to the context".
                - Use the provided context to answer.

                Strict Instructions:
                - DO NOT say phrases like:
                "Based on the provided information"
                "According to the context"
                "From the given data"
                - DO NOT explain how you got the answer.

                Rules:
                - If the question is about identity, respond:
                "I am an AI assistant for EMR Chain, here to help answer questions about the company."
                - For all other questions:
                - Answer ONLY using the provided context
                - If the answer is not found, say: "I don't know based on the provided information"

                Style:
                - Be clear, direct, and professional
                - Keep answers short unless more detail is required
            """
            },
            {
                "role": "user",
                "content": f"""
                Context:
                {context}

                Question:
                {question}
                    """
            }
        ]
    )

    return response.choices[0].message.content