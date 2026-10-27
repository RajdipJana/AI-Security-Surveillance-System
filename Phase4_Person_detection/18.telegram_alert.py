import os
import requests


# ============================================================
# TELEGRAM CONFIGURATION
# ============================================================


# Telegram configuration:
# Create telegram_config.py in the project root with your own credentials.
# See the README for setup instructions.

try:
    from telegram_config import BOT_TOKEN, CHAT_ID
except ImportError:
    BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
    CHAT_ID = "YOUR_CHAT_ID_HERE"


# Telegram API base URL
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"


# ============================================================
# SEND TEXT MESSAGE
# ============================================================

def send_telegram_message(message):

    url = f"{BASE_URL}/sendMessage"

    data = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(
            url,
            data=data,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        if result.get("ok"):
            print("[TELEGRAM] Message sent successfully")
            return True

        print("[TELEGRAM] Failed to send message")
        print(result)
        return False

    except Exception as error:
        print(f"[TELEGRAM] Error: {error}")
        return False


# ============================================================
# SEND PHOTO
# ============================================================

def send_telegram_photo(photo_path, caption=""):

    url = f"{BASE_URL}/sendPhoto"

    try:

        with open(photo_path, "rb") as photo:

            files = {
                "photo": photo
            }

            data = {
                "chat_id": CHAT_ID,
                "caption": caption
            }

            response = requests.post(
                url,
                data=data,
                files=files,
                timeout=20
            )

        response.raise_for_status()

        result = response.json()

        if result.get("ok"):
            print("[TELEGRAM] Photo sent successfully")
            return True

        print("[TELEGRAM] Failed to send photo")
        print(result)
        return False

    except Exception as error:
        print(f"[TELEGRAM] Photo error: {error}")
        return False


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Testing Telegram connection...")

    send_telegram_message(
        "🚨 AI Security Surveillance System\n\n"
        "Telegram alert module is working successfully!"
    )