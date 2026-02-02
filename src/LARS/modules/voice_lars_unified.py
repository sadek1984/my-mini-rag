"""
voice_lars_unified.py
===============================
البوت الصوتي باستخدام أنماط النظام القديم الموحدة
"""
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from faster_whisper import WhisperModel
from io import BytesIO

# استيراد المحول
from legacy_patterns_adapter import LegacyPatternsAdapter


# ============================================================
# Configuration
# ============================================================

print("🔄 Loading Whisper model...")
stt_model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8",
    num_workers=2,
    cpu_threads=4
)

DB_PATH = "/Users/a12/Buraidah_lars/src/LARS/data/lars_data.duckdb"
adapter = LegacyPatternsAdapter(DB_PATH)


# ============================================================
# Telegram Handlers
# ============================================================

async def handle_voice_unified(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة الصوت باستخدام الأنماط القديمة"""
    voice_file = await update.message.voice.get_file()
    file_path = "temp_voice.ogg"
    await voice_file.download_to_drive(file_path)
    
    await update.message.reply_text("🎧 سمعتك، جاري التحليل...")
    
    try:
        # تحويل الصوت إلى نص
        segments, _ = stt_model.transcribe(
            file_path,
            beam_size=5,
            language="ar",
            initial_prompt="مبيدات حشرية، عينات، طماطم، خيار، هيل، توابل"
        )
        query = " ".join([s.text for s in segments])
        
        await update.message.reply_text(f"📝 **سؤالك:** {query}")
        await update.message.reply_text("🔍 جاري البحث باستخدام الأنماط الدقيقة...")
        
        # استخدام المحول للبحث
        response, df = adapter.match_and_execute(query)
        
        if response:
            # إرسال الرد
            await update.message.reply_text(response, parse_mode='Markdown')
            
            # إذا كانت النتائج كثيرة، إرسال ملف Excel
            if df is not None and len(df) > 10:
                excel_file = BytesIO()
                df.to_excel(excel_file, index=False, engine='openpyxl')
                excel_file.seek(0)
                await update.message.reply_document(
                    document=excel_file,
                    filename="lars_results.xlsx",
                    caption="📊 التقرير الكامل"
                )
        else:
            # إذا لم يتطابق مع أي نمط
            await update.message.reply_text(
                "⚠️ لم أتمكن من فهم سؤالك باستخدام الأنماط المعروفة.\n\n"
                "💡 جرب صياغته مثل:\n"
                "• 'ما عدد عينات الطماطم فوق الحد'\n"
                "• 'ابحث عن الفيبرونيل في الخيار'\n"
                "• 'المبيدات في الهيل'\n"
                "• 'عينات تحتوي على 3 مبيدات'",
                parse_mode='Markdown'
            )
    
    except Exception as e:
        await update.message.reply_text(f"⚠️ خطأ: {str(e)[:200]}")
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


async def handle_text_unified(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة النص باستخدام الأنماط القديمة"""
    query = update.message.text
    
    await update.message.reply_text("🔍 جاري البحث باستخدام الأنماط الدقيقة...")
    
    try:
        response, df = adapter.match_and_execute(query)
        
        if response:
            await update.message.reply_text(response, parse_mode='Markdown')
            
            if df is not None and len(df) > 10:
                excel_file = BytesIO()
                df.to_excel(excel_file, index=False, engine='openpyxl')
                excel_file.seek(0)
                await update.message.reply_document(
                    document=excel_file,
                    filename="lars_results.xlsx",
                    caption="📊 التقرير الكامل"
                )
        else:
            await update.message.reply_text(
                "⚠️ لم أتمكن من فهم سؤالك باستخدام الأنماط المعروفة.\n\n"
                "💡 جرب صياغته مثل الأمثلة السابقة.",
                parse_mode='Markdown'
            )
    
    except Exception as e:
        await update.message.reply_text(f"⚠️ خطأ: {str(e)[:200]}")


# ============================================================
# Run Bot
# ============================================================

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8393315665:AAG82MK45WQfIpn054HdoyxZK4rGtZLsL-A")


def main():
    app = ApplicationBuilder().token(TOKEN).build()
    
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_unified))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_unified))
    
    print("🚀 LARS Voice Assistant (Unified Patterns) is running...")
    print("✅ يستخدم نفس أنماط التطبيق النصي القديم!")
    print("")
    print("📝 الأنماط المدعومة:")
    print("   • Pattern 0: عد العينات فوق/تحت الحد")
    print("   • Pattern 0.5: المبيدات في الأحياء")
    print("   • Pattern 0.6: عينات تحتوي على مبيد")
    print("   • Pattern 1: عينات بعدد مبيدات محدد")
    print("   • Pattern 2: قراءات مبيد محدد")
    print("   • Pattern 3: قائمة المبيدات في عينة")
    print("   • Pattern 4: مبيدات في أحياء متعددة")
    
    app.run_polling()


if __name__ == '__main__':
    main()
