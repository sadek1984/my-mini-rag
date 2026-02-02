"""
voice_bot_enhanced.py
====================
بوت صوتي محسّن مع دعم كامل للأنماط
"""
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from faster_whisper import WhisperModel
from io import BytesIO
import pandas as pd

from enhanced_patterns_adapter import EnhancedPatternsAdapter


# ============================================================
# Configuration
# ============================================================

print("🔄 Loading Whisper large-v3 model...")
stt_model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8",
    num_workers=2,
    cpu_threads=4
)
print("✅ Whisper model loaded!")

DB_PATH = os.environ.get("DB_PATH", "/Users/a12/Buraidah_lars/src/LARS/data/lars_data.duckdb")
adapter = EnhancedPatternsAdapter(DB_PATH)

# Common pesticide names for STT prompting
PESTICIDE_NAMES_AR = """مبيدات حشرية: البايفنثرن، الفيبرونيل، الكلوربيريفوس، 
الايميداكلوبريد، البابروفيزن، الايثيون، دلتامثرين، سيبرمثرين"""


# ============================================================
# Handlers
# ============================================================

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة الرسائل الصوتية"""
    voice_file = await update.message.voice.get_file()
    file_path = "temp_voice.ogg"
    await voice_file.download_to_drive(file_path)
    
    await update.message.reply_text("🎧 سمعتك، جاري التحليل...")
    
    try:
        # Enhanced transcription
        segments, info = stt_model.transcribe(
            file_path,
            beam_size=5,
            language="ar",
            initial_prompt=PESTICIDE_NAMES_AR,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=True
        )
        
        query = " ".join([s.text for s in segments])
        confidence = info.language_probability
        
        # Show transcription
        await update.message.reply_text(
            f"📝 **سؤالك:** {query}\n🎯 الدقة: {confidence*100:.1f}%",
            parse_mode='Markdown'
        )
        
        # Process query
        await update.message.reply_text("🔍 جاري البحث...")
        
        response, df = adapter.match_and_execute(query)
        
        if response:
            # Send text response
            # Split if too long
            if len(response) > 4000:
                parts = [response[i:i+4000] for i in range(0, len(response), 4000)]
                for part in parts:
                    await update.message.reply_text(part, parse_mode='Markdown')
            else:
                await update.message.reply_text(response, parse_mode='Markdown')
            
            # Send Excel if results are large
            if df is not None and len(df) > 10:
                excel_file = BytesIO()
                df.to_excel(excel_file, index=False, engine='openpyxl')
                excel_file.seek(0)
                await update.message.reply_document(
                    document=excel_file,
                    filename="lars_results.xlsx",
                    caption="📎 التقرير الكامل"
                )
        else:
            await update.message.reply_text(
                "⚠️ **لم أتمكن من فهم سؤالك.**\n\n"
                "💡 **أمثلة على الأسئلة المدعومة:**\n\n"
                "**1. البحث عن مبيد:**\n"
                "• ابحث عن الفيبرونيل في الطماطم\n"
                "• Find bifenthrin in cucumber\n\n"
                "**2. عد العينات:**\n"
                "• ما عدد عينات الخيار فوق الحد؟\n"
                "• كم عينة طماطم و كوسة كل على حده؟\n\n"
                "**3. عينات بعدد مبيدات:**\n"
                "• ما العينات التي تحتوي على 6 مبيدات؟\n"
                "• عينات بمبيد واحد و مبيدين كل على حده\n\n"
                "**4. قائمة المبيدات:**\n"
                "• ماهي المبيدات في الهيل؟\n"
                "• المبيدات في الفلفل مع التكرار\n\n"
                "**5. المبيدات في أحياء:**\n"
                "• المبيدات في الإسكان والريان\n"
                "• المبيدات في حي النهضة",
                parse_mode='Markdown'
            )
    
    except Exception as e:
        await update.message.reply_text(f"⚠️ خطأ: {str(e)[:300]}")
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة الرسائل النصية"""
    query = update.message.text
    
    await update.message.reply_text("🔍 جاري البحث...")
    
    try:
        response, df = adapter.match_and_execute(query)
        
        if response:
            # Send response
            if len(response) > 4000:
                parts = [response[i:i+4000] for i in range(0, len(response), 4000)]
                for part in parts:
                    await update.message.reply_text(part, parse_mode='Markdown')
            else:
                await update.message.reply_text(response, parse_mode='Markdown')
            
            # Send Excel if needed
            if df is not None and len(df) > 10:
                excel_file = BytesIO()
                df.to_excel(excel_file, index=False, engine='openpyxl')
                excel_file.seek(0)
                await update.message.reply_document(
                    document=excel_file,
                    filename="lars_results.xlsx",
                    caption="📎 التقرير الكامل"
                )
        else:
            await update.message.reply_text(
                "⚠️ لم أتمكن من فهم سؤالك. جرب إعادة صياغته.",
                parse_mode='Markdown'
            )
    
    except Exception as e:
        await update.message.reply_text(f"⚠️ خطأ: {str(e)[:300]}")


# ============================================================
# Main
# ============================================================

def main():
    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8393315665:AAG82MK45WQfIpn054HdoyxZK4rGtZLsL-A")
    
    app = ApplicationBuilder().token(TOKEN).build()
    
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    print("=" * 60)
    print("🚀 LARS Enhanced Voice Assistant is running!")
    print("=" * 60) 
    print("\n✅ الأنماط المدعومة:")
    print("   1. البحث عن مبيد في عينة")
    print("   2. عد العينات فوق/تحت الحد")
    print("   3. عينات بعدد مبيدات محدد")
    print("   4. قائمة المبيدات في عينة")
    print("   5. المبيدات في أحياء")
    print("   6. إحصائيات (max, min, avg, median)")
    print("   7. عينات في منشأة/مستلم")
    print("   8. ترتيب الأحياء حسب المخالفات")
    print("\n📊 الميزات:")
    print("   • دعم كامل للغة العربية والإنجليزية")
    print("   • دعم خاصية 'كل على حده'")
    print("   • إرسال تلقائي لملفات Excel للنتائج الكبيرة")
    print("   • معالجة متقدمة للصوت (Whisper large-v3)")
    print("=" * 60)
    
    app.run_polling()


if __name__ == '__main__':
    main()