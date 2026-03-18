from __future__ import annotations

import argparse
import getpass
import json
import os
import urllib.error
import urllib.parse
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a Telegram Bot API message."
    )
    parser.add_argument("--message", required=True, help="Message text to send.")
    parser.add_argument("--bot-token", default="", help="Telegram bot token. Falls back to TELEGRAM_BOT_TOKEN.")
    parser.add_argument("--chat-id", default="", help="Telegram chat id. Falls back to TELEGRAM_CHAT_ID.")
    parser.add_argument("--parse-mode", default="", help="Optional Telegram parse mode, for example Markdown.")
    return parser.parse_args()


def send_message(bot_token: str, chat_id: str, message: str, parse_mode: str = "") -> dict:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    data = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    args = parse_args()
    bot_token = args.bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = args.chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token:
        bot_token = getpass.getpass("Telegram bot token: ").strip()
    if not chat_id:
        chat_id = input("Telegram chat id: ").strip()
    if not bot_token:
        raise ValueError("Telegram bot token is required.")
    if not chat_id:
        raise ValueError("Telegram chat id is required.")

    try:
        result = send_message(bot_token, chat_id, args.message, parse_mode=args.parse_mode)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Telegram API request failed with {error.code}: {body}") from error

    if not result.get("ok"):
        raise RuntimeError(f"Telegram API returned a failure response: {result}")

    print("Telegram message sent successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
