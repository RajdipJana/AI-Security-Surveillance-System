# ==================================================
# TELEGRAM ALERT MODULE
# AI SECURITY SURVEILLANCE SYSTEM
# ==================================================

import requests


# ==================================================
# TELEGRAM CONFIGURATION
# ==================================================




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


# ==================================================
# SEND TEXT MESSAGE
# ==================================================

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
            timeout=5
        )

        response.raise_for_status()

        result = response.json()

        if result.get("ok"):

            print("[TELEGRAM] Message sent successfully")

            return True

        print("[TELEGRAM] Message failed")
        print(result)

        return False

    except Exception as error:

        print(
            f"[TELEGRAM] Message error: {error}"
        )

        return False


# ==================================================
# SEND PHOTO
# ==================================================

def send_telegram_photo(
    photo_path,
    caption=""
):

    url = f"{BASE_URL}/sendPhoto"

    try:

        with open(
            photo_path,
            "rb"
        ) as photo:

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
                timeout=15
            )

        response.raise_for_status()

        result = response.json()

        if result.get("ok"):

            print(
                "[TELEGRAM] Photo sent successfully"
            )

            return True

        print("[TELEGRAM] Photo failed")
        print(result)

        return False

    except Exception as error:

        print(
            f"[TELEGRAM] Photo error: {error}"
        )

        return False


# ==================================================
# WEAPON ALERT
# ==================================================

def send_weapon_alert(
    weapon_name,
    risk_level,
    risk_score
):

    message = (
        "🚨 WEAPON DETECTED\n\n"
        f"Weapon: {weapon_name}\n"
        f"Risk: {risk_level}\n"
        f"Risk Score: {risk_score}\n"
        f"Time: {get_current_time()}"
    )

    return send_telegram_message(
        message
    )


# ==================================================
# INTRUDER ALERT
# ==================================================

def send_intruder_alert(
    person_id,
    risk_level,
    risk_score
):

    message = (
        "🚨 INTRUDER ALERT\n\n"
        f"Person ID: {person_id}\n"
        "Restricted Area: ENTERED\n"
        f"Risk: {risk_level}\n"
        f"Risk Score: {risk_score}\n"
        f"Time: {get_current_time()}"
    )

    return send_telegram_message(
        message
    )


# ==================================================
# INTRUDER SNAPSHOT ALERT
# ==================================================

def send_intruder_snapshot(
    photo_path,
    person_id,
    risk_level,
    risk_score,
    weapon_name=""
):

    caption = (
        "🚨 SECURITY ALERT\n\n"
        f"Person ID: {person_id}\n"
        "Restricted Area: ENTERED\n"
        f"Risk: {risk_level}\n"
        f"Risk Score: {risk_score}\n"
    )

    if weapon_name:

        caption += (
            f"Weapon: {weapon_name}\n"
        )

    else:

        caption += (
            "Weapon: None detected\n"
        )

    caption += (
        f"Time: {get_current_time()}"
    )

    return send_telegram_photo(
        photo_path,
        caption
    )


# ==================================================
# CURRENT TIME
# ==================================================

def get_current_time():

    from datetime import datetime

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ==================================================
# TEST TELEGRAM CONNECTION
# ==================================================

if __name__ == "__main__":

    print(
        "Testing Telegram alert module..."
    )

    send_telegram_message(
        "🚨 AI Security Surveillance System\n\n"
        "Telegram alert module is working!"
    )