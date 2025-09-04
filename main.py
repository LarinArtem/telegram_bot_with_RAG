from typing import Final
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
import inspect
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    JobQueue,
    MessageHandler,
    filters
)
from modules.momentum import momentum_stocks
from modules.undervalued import undervalued_stocks
from modules.rddt import format_rankings_table, rankings_df
from modules.options_strategy import compute_recommendation_yf, format_recommendation_msg
from modules.ten_k_analysis import form_type




TOKEN: Final = "YOUR_BOT_TOKEN"
BOT_USERNAME: Final = "@YourBotUsername"




MOMENTUM, UNDERVALUED_STOCKS, BTN_IMPLIED_VOLATILITY, RDDT, TEN_K, TEN_Q = (
    "5 days momentum",
    "Undervalued stocks",
    "Implied volatility",
    "Reddit sentiment",
    "10-K Analysis",
    "10-Q Analysis",
)


CB_MOMENTUM = "momentum"
CB_UNDERVAL = "undervalued_stocks"
CB_IV = "btn_implied_volatility"
CB_RDDT = "rddt"
CB_TENK = "ten_k"
CB_TENQ = "ten_q"


STATE_KEY = "awaiting"          
STATE_FORM = "FORM_10K", "FORM_10Q"
STATE_OPTIONS = "OPTIONS"

BUTTONS = [
    [InlineKeyboardButton(MOMENTUM, callback_data=CB_MOMENTUM)],
    [InlineKeyboardButton(UNDERVALUED_STOCKS, callback_data=CB_UNDERVAL)],
    [InlineKeyboardButton(BTN_IMPLIED_VOLATILITY, callback_data=CB_IV)],
    [InlineKeyboardButton(RDDT, callback_data=CB_RDDT)],
    [InlineKeyboardButton(TEN_K, callback_data=CB_TENK)],
    [InlineKeyboardButton(TEN_Q, callback_data=CB_TENQ)],
]


async def maybe_await(fn, *args, **kwargs):
    res = fn(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data[STATE_KEY] = None
    await update.message.reply_text("Options:", reply_markup=InlineKeyboardMarkup(BUTTONS))


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Stopping the bot. Goodbye!")
    await context.application.stop()


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    context.user_data[STATE_KEY] = None

    if query.data == CB_MOMENTUM:
        await query.edit_message_text("Please wait, fetching top 10 stocks with 5-day momentum...")
        bot_response = momentum_stocks()
        await query.message.reply_text(bot_response, parse_mode="HTML")
        await query.message.reply_text("Options:", reply_markup=InlineKeyboardMarkup(BUTTONS))

    elif query.data == CB_UNDERVAL:
        await query.edit_message_text("Please wait, fetching top 10 stocks undervalued by Wall Street analysts...")
        bot_response = undervalued_stocks()
        await query.message.reply_text(bot_response, parse_mode="HTML")
        await query.message.reply_text("Options:", reply_markup=InlineKeyboardMarkup(BUTTONS))

    elif query.data == CB_RDDT:
        await query.edit_message_text("Please wait, fetching Reddit sentiment rankings...")
        bot_response = format_rankings_table(rankings_df)
        await query.message.reply_text(bot_response, parse_mode="HTML")
        await query.message.reply_text("Options:", reply_markup=InlineKeyboardMarkup(BUTTONS))

    elif query.data == CB_TENK:
        context.user_data[STATE_KEY] = "FORM_10K"
        await query.edit_message_text("Send a ticker (e.g., AAPL) for a 10-K analysis.")

    elif query.data == CB_TENQ:
        context.user_data[STATE_KEY] = "FORM_10Q"
        await query.edit_message_text("Send a ticker (e.g., AAPL) for a 10-Q analysis.")


    elif query.data == CB_IV:
        context.user_data[STATE_KEY] = STATE_OPTIONS
        await query.edit_message_text("Send a ticker (e.g., AAPL) to get option strategy results.")




async def on_ticker_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip().upper()
    if not text.isalnum() or len(text) > 10:
        await update.message.reply_text("Please send a valid ticker symbol (e.g., AAPL).")
        return

    mode = context.user_data.get(STATE_KEY)

    try:
        if mode == "FORM_10K":
            await update.message.reply_text(f"Analyzing 10-K for {text}...")
            result = form_type(text, "10-K")
            await update.message.reply_text(result, parse_mode="HTML")

        elif mode == "FORM_10Q":
            await update.message.reply_text(f"Analyzing 10-Q for {text}...")
            result = form_type(text, "10-Q")
            await update.message.reply_text(result, parse_mode="HTML")

        elif mode == STATE_OPTIONS:
            await update.message.reply_text(f"Computing option strategy for {text}...")
            result = await maybe_await(compute_recommendation_yf, text)
            msg = format_recommendation_msg(result)
            await update.message.reply_text(msg, parse_mode="HTML")

        else:
            await update.message.reply_text(f"No mode selected. Running default option strategy for {text}...")
            result = await maybe_await(compute_recommendation_yf, text)
            msg = format_recommendation_msg(result)
            await update.message.reply_text(msg, parse_mode="HTML")

    except Exception as e:
        await update.message.reply_text(f"⚠️ Error processing request for {text}: {e}")

    context.user_data[STATE_KEY] = None

    await update.message.reply_text("Options:", reply_markup=InlineKeyboardMarkup(BUTTONS))


def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN environment variable not set!")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_ticker_message))
    app.run_polling()


if __name__ == "__main__":
    main()
