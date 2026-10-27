import requests


# Telegram configuration:
# Create telegram_config.py in the project root with your own credentials.
# See the README for setup instructions.

try:
    from telegram_config import BOT_TOKEN, CHAT_ID
except ImportError:
    BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
    CHAT_ID = "YOUR_CHAT_ID_HERE"

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

data = {
    "chat_id": CHAT_ID,
    "text": "🚨 Telegram connection successful!\n\nAI Security Surveillance System is connected."
}

response = requests.post(url, data=data)

print(response.json())