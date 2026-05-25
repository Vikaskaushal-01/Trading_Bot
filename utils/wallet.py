import json
import os

MEMORY_FILE = "data/trading_memory.json"

def load_memory():

    if not os.path.exists("data"):

        os.makedirs("data")

    if not os.path.exists(MEMORY_FILE):

        default_data = {
            "wallet": 500,
            "transactions": []
        }

        with open(MEMORY_FILE, "w") as f:

            json.dump(default_data, f, indent=4)

    with open(MEMORY_FILE, "r") as f:

        return json.load(f)

def save_memory(data):

    with open(MEMORY_FILE, "w") as f:

        json.dump(data, f, indent=4)

def add_money(amount):

    memory = load_memory()

    memory["wallet"] += amount

    save_memory(memory)

    return memory["wallet"]