import os
from groq import Groq

groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)
def refine_query(question):

    prompt = f"""
    Rewrite the question into a clear standalone query.
    If it contains vague words like "it", "this", "they",
    replace them using context.

    Question:
    {question}

    Refined:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system", 
                "content": """
                you are a assisstan that refine user query.
                """
        
            },
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()