from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client=Groq(api_key=os.getenv("GROQ_API_KEY"))
model=os.getenv("MODEL_NAME")

TRADING_CONTEXT="""
You are an advanced AI trading assistant.
You only answer trading, stocks, crypto, investing, technical analysis, portfolio, risk management, candlestick, indicators, and financial market related questions.
Analyze the meaning of the user query first.
If unrelated, politely redirect the user toward trading-related topics.
Provide educational insights only.
"""

def ask_bot(question):
    response=client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":TRADING_CONTEXT},
            {"role":"user","content":question}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content