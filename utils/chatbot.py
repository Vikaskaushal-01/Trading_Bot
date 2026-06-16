from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

model = os.getenv("MODEL_NAME")


BASE_CONTEXT = """
You are the AI assistant of an AI Trading Platform.

You have access to:
- Wallet Balance
- Portfolio Holdings
- Trending Stocks
- Current Selected Stock
- Transaction History

Always use website data first before giving generic answers.

If user asks:
- trending stocks
- portfolio
- wallet
- holdings
- profit/loss

respond using the provided website data.

You only answer trading, investing, stocks, crypto,
technical analysis, portfolio management,
risk management and market related questions.

If the question is unrelated,
politely redirect the user toward trading topics.

Never say:
'I cannot access your website.'
'I cannot view your website.'
'I don't have access to your data.'

Because website data will be provided.
"""


def ask_bot(question, context=None):

    if context is None:
        context = {}

    wallet = context.get(
        "wallet",
        "Unknown"
    )

    portfolio = context.get(
        "portfolio",
        []
    )

    trending = context.get(
        "trending_stocks",
        []
    )

    selected_stock = context.get(
        "selected_stock",
        "None"
    )

    website_context = f"""
CURRENT WEBSITE DATA

Wallet Balance:
{wallet}

Selected Stock:
{selected_stock}

Portfolio:
{json.dumps(portfolio, indent=2)}

Trending Stocks:
{json.dumps(trending, indent=2)}
"""

    # FAST WEBSITE RESPONSES

    q = question.lower()

    if "wallet" in q:

        return f"""
Current wallet balance is ${wallet}
"""

    if "portfolio" in q:

        if len(portfolio) == 0:

            return "No stocks currently available in portfolio."

        response = "Current Portfolio:\n\n"

        for stock in portfolio:

            response += (
                f"• {stock.get('Asset')} "
                f"- Investment ${stock.get('Investment')}\n"
            )

        return response

    if "trending" in q:

        if len(trending) == 0:

            return (
                "No trending stocks available."
            )

        response = (
            "Current Trending Stocks:\n\n"
        )

        for stock in trending:

            response += (
                f"• {stock['name']} "
                f"(${stock['price']}) "
                f"{stock['change']}%\n"
            )

        return response

    try:

        response = client.chat.completions.create(

            model=model,

            messages=[

                {
                    "role": "system",
                    "content":
                    BASE_CONTEXT
                    + "\n\n"
                    + website_context
                },

                {
                    "role": "user",
                    "content": question
                }

            ],

            temperature=0.3,

            max_tokens=600

        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        return (
            f"Chatbot Error: {str(e)}"
        )