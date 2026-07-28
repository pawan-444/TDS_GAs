"""Telegram webhook parsing and replies."""
import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from agent import DataAgent
from json_formatter import telegram_json

LOGGER = logging.getLogger(__name__)


def build_telegram_application(token: str, agent: DataAgent) -> Application:
    application = Application.builder().token(token).updater(None).build()

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_message:
            await update.effective_message.reply_text("Send a question, optionally with a public dataset URL.")

    async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_message:
            await update.effective_message.reply_text("I analyse CSV, TSV, Excel, JSON, HTML-table, and ZIP dataset URLs.")

    async def message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_message or not update.effective_user or not update.effective_message.text:
            return
        answer, log_url = await agent.run(update.effective_user.id, update.effective_message.text)
        await update.effective_message.reply_text(telegram_json(answer, log_url))

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message))
    return application
