import os
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton
)
from telegram.error import BadRequest, TimedOut, TelegramError
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from process_media import process_video

TOKEN = "8668673885:AAElLmg8nxLSlMO6ZCWg98NwDN6lWHGZ1DI"
MAX_VIDEO_MB = 20

DOWNLOAD_DIR = "bot_data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🎥 Обработать видео")],
        [KeyboardButton("ℹ️ О проекте")],
        [KeyboardButton("❌ Отмена")]
    ],
    resize_keyboard=True
)

USER_STATE = {}


def _extract_video_media(message):
    if message.video:
        return message.video, "video"
    doc = message.document
    if doc and (doc.mime_type or "").startswith("video/"):
        return doc, "document"

    return None, None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    USER_STATE[update.effective_user.id] = None
    await update.message.reply_text(
        "👋 Привет!\n\n"
        "Я бот для распознавания действий, эмоций и контекста человека.\n"
        "Выбери, что хочешь сделать:",
        reply_markup=MAIN_KEYBOARD
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🆘 Помощь\n\n"
        "/start — главное меню\n"
        "/help — помощь\n\n"
        "Как пользоваться:\n"
        "1️⃣ Выбери обработку видео\n"
        "2️⃣ Отправь видео\n"
        "3️⃣ Получи результат\n\n"
        "❌ Отмена — доступна всегда"
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    uid = update.effective_user.id

    if text == "🎥 Обработать видео":
        USER_STATE[uid] = "video"
        await update.message.reply_text(
            f"🎥 Отправь видео (лимит: до {MAX_VIDEO_MB} MB)."
        )

    elif text == "ℹ️ О проекте":
        await update.message.reply_text(
            "🔬 Проект:\n"
            "Реальное распознавание действий и намерений человека\n"
            "⚙️ Работает на CPU"
        )

    elif text == "❌ Отмена":
        USER_STATE[uid] = None
        await update.message.reply_text(
            "❌ Действие отменено",
            reply_markup=MAIN_KEYBOARD
        )

    else:
        await update.message.reply_text("❓ Используй кнопки или /help")

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if USER_STATE.get(uid) != "video":
        await update.message.reply_text("⚠️ Сначала выбери «Обработать видео»")
        return

    media, media_kind = _extract_video_media(update.message)
    if media is None:
        await update.message.reply_text("⚠️ Пришли именно видеофайл.")
        return

    size_bytes = int(getattr(media, "file_size", 0) or 0)
    size_mb = size_bytes / (1024 * 1024) if size_bytes else 0.0

    if size_mb > MAX_VIDEO_MB:
        await update.message.reply_text(
            f"⚠️ Файл слишком большой ({size_mb:.1f} MB). "
            f"Лимит бота: до {MAX_VIDEO_MB} MB."
        )
        USER_STATE[uid] = None
        return

    try:
        file = await media.get_file(
            read_timeout=120,
            write_timeout=120,
            connect_timeout=60,
            pool_timeout=60,
        )
    except BadRequest as e:
        if "file is too big" in str(e).lower():
            await update.message.reply_text(
                f"⚠️ Telegram вернул `File is too big`.\n"
                f"Отправь видео до {MAX_VIDEO_MB} MB."
            )
            USER_STATE[uid] = None
            return
        raise

    input_path = os.path.join(DOWNLOAD_DIR, f"{uid}_input.mp4")
    output_path = os.path.join(DOWNLOAD_DIR, f"{uid}_output.mp4")

    await file.download_to_drive(input_path)

    await update.message.reply_text("⏳ Обрабатываю видео...")
    process_video(
        input_path,
        output_path,
        output_max_width=960,
        target_fps=18,
    )

    try:
        with open(output_path, "rb") as video_file:
            await update.message.reply_video(
                video=video_file,
                caption=f"✅ Готово! ({media_kind})",
                reply_markup=MAIN_KEYBOARD,
                supports_streaming=True,
                read_timeout=300,
                write_timeout=300,
                connect_timeout=60,
                pool_timeout=60,
            )
    except TimedOut:
        await update.message.reply_text(
            "⚠️ Не успел отправить как видео (TimedOut). Пробую отправить как файл..."
        )
        try:
            with open(output_path, "rb") as video_file:
                await update.message.reply_document(
                    document=video_file,
                    filename=os.path.basename(output_path),
                    caption="✅ Готово (как файл)!",
                    reply_markup=MAIN_KEYBOARD,
                    read_timeout=300,
                    write_timeout=300,
                    connect_timeout=60,
                    pool_timeout=60,
                )
        except TelegramError as e:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            await update.message.reply_text(
                f"❌ Не удалось отправить результат ({type(e).__name__}). "
                f"Размер файла: {size_mb:.1f} MB. Попробуй видео короче."
            )
    except TelegramError as e:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        await update.message.reply_text(
            f"❌ Ошибка отправки ({type(e).__name__}). "
            f"Размер файла: {size_mb:.1f} MB."
        )
    finally:
        USER_STATE[uid] = None

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))
    app.add_handler(MessageHandler(filters.Document.VIDEO, handle_video))

    print("Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
