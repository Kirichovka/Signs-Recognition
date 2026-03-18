# Telegram Notifications

## Goal

This document explains how to connect the training launcher to Telegram notifications.

The project now supports Telegram alerts from:

- [`python/run_model_pipelines.py`](/D:/Integration-Game/gesture-trainer-web/python/run_model_pipelines.py)

and also includes a standalone test sender:

- [`python/send_telegram_message.py`](/D:/Integration-Game/gesture-trainer-web/python/send_telegram_message.py)

## What you need

You need two values:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

These can be passed either:

- as command-line flags
- or through environment variables

## Step 1. Create a bot

1. Open Telegram.
2. Search for **BotFather**.
3. Start a chat with BotFather.
4. Run:

```text
/newbot
```

5. Follow the prompts:
   - choose a bot name
   - choose a bot username ending in `bot`

6. BotFather will give you a token that looks like:

```text
123456789:AA...
```

That is your `TELEGRAM_BOT_TOKEN`.

## Step 2. Start a chat with your bot

Before the bot can send you a message:

1. Search for your bot in Telegram.
2. Open the bot chat.
3. Press **Start** or send any message.

Without this step, Telegram usually will not let the bot message you first.

## Step 3. Get your chat ID

The easiest way is to send a test message to the bot and then call Telegram's `getUpdates`.

Open this in a browser, replacing the token:

```text
https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
```

Look for a JSON block that contains:

```json
"chat": {
  "id": 123456789,
  ...
}
```

That `id` is your `TELEGRAM_CHAT_ID`.

## Step 4. Store the values

The repository includes:

- [`.env.example`](/D:/Integration-Game/gesture-trainer-web/.env.example)

You can keep the real values in your shell environment.

### PowerShell example

```powershell
$env:TELEGRAM_BOT_TOKEN="your_bot_token"
$env:TELEGRAM_CHAT_ID="your_chat_id"
```

### Or pass them directly

```powershell
python run_model_pipelines.py ^
  --mode all ^
  --word-dataset-root D:\Integration-Game\gesture-trainer-web\datasets\asl_citizen\ASL_Citizen ^
  --alphabet-dataset-root D:\Integration-Game\gesture-trainer-web\datasets\asl_semcom\ASL_SemCom ^
  --notify-telegram ^
  --telegram-bot-token "your_bot_token" ^
  --telegram-chat-id "your_chat_id"
```

## Step 5. Test message only

You can test Telegram separately before running a training pipeline:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
$env:TELEGRAM_BOT_TOKEN="your_bot_token"
$env:TELEGRAM_CHAT_ID="your_chat_id"
python send_telegram_message.py --message "Test message from gesture-trainer-web"
```

## Step 6. Use it with the unified launcher

Example:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
$env:TELEGRAM_BOT_TOKEN="your_bot_token"
$env:TELEGRAM_CHAT_ID="your_chat_id"

python run_model_pipelines.py ^
  --mode all ^
  --word-dataset-root D:\Integration-Game\gesture-trainer-web\datasets\asl_citizen\ASL_Citizen ^
  --alphabet-dataset-root D:\Integration-Game\gesture-trainer-web\datasets\asl_semcom\ASL_SemCom ^
  --export-web ^
  --notify-telegram
```

## What the notification includes

The launcher sends:

- success or failure status
- selected mode
- total duration
- run paths
- whether web export was enabled
- whether publish-to-models was enabled
- the error text if the pipeline failed

## Common issues

### The bot sends nothing

Check:

- you pressed **Start** in the bot chat
- the bot token is correct
- the chat ID is correct

### `getUpdates` returns nothing useful

Make sure:

- you already sent at least one message to the bot
- you are querying the correct bot token

### The launcher fails before sending a message

The launcher can only notify if:

- `--notify-telegram` is enabled
- a valid token is available
- a valid chat ID is available
