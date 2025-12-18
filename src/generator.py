import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_draft(prompt, model="gpt-4o-mini"):
    """
    Generates the initial response that the human expert will critique.
    Using a smaller model (like 4o-mini) is often better for RLHF 
    because it makes more 'obvious' mistakes for you to fix.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst assistant. Be concise."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"