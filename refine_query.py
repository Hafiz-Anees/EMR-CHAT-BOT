def refine_query(question, client):

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
                "content": "You refine user queries into clear standalone questions."
            },
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()