"""
voice_bot_unified.py
====================
البوت الصوتي باستخدام المعالج الهجين الموحد
"""
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from faster_whisper import WhisperModel
from io import BytesIO
import pandas as pd

# Try to import OpenAI client
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

from unified_query_processor import UnifiedQueryProcessor


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

# Database path
DB_PATH = os.environ.get("DB_PATH", "/Users/a12/Buraidah_lars/src/LARS/data/lars_data.duckdb")

# LLM client (optional)
llm_client = None
if HAS_OPENAI:
    try:
        llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        print("✅ Ollama LLM client connected (gemma3:4b)!")
    except Exception as e:
        print(f"⚠️ Ollama LLM client not available: {e}")

# Initialize unified processor
processor = UnifiedQueryProcessor(DB_PATH, llm_client)

# Pesticide names for STT prompting
PESTICIDE_NAMES_AR = """مبيدات حشرية: البايفنثرن، الفيبرونيل، الكلوربيريفوس، 
الايميداكلوبريد، البابروفيزن، الايثيون، دلتامثرين، سيبرمثرين.
عينات: طماطم، خيار، كوسة، فلفل، باذنجان، هيل، فستق."""


# ============================================================
# Telegram Handlers
# ============================================================

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة الرسائل الصوتية"""
    voice_file = await update.message.voice.get_file()
    file_path = "temp_voice.ogg"
    await voice_file.download_to_drive(file_path)
    
    await update.message.reply_text("🎧 سمعتك، جاري التحليل...")
    
    try:
        # Transcribe with enhanced settings
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
        
        await update.message.reply_text("🔍 جاري المعالجة عبر النظام الهجين...")
        
        # Process with unified processor
        response, df = processor.process(query)
        
        # Send response (split if too long)
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
    
    except Exception as e:
        await update.message.reply_text(f"⚠️ خطأ: {str(e)[:300]}")
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة الرسائل النصية"""
    query = update.message.text
    
    await update.message.reply_text("🔍 جاري المعالجة...")
    
    try:
        # Process with unified processor
        response, df = processor.process(query)
        
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
    
    except Exception as e:
        await update.message.reply_text(f"⚠️ خطأ: {str(e)[:300]}")


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """رسالة الترحيب"""
    welcome_msg = """
🌱 **مرحباً بك في مساعد LARS الصوتي!**

أنا مساعدك الذكي لتحليل بيانات المبيدات.

📝 **الأسئلة المدعومة:**

**1️⃣ البحث عن مبيد:**
• ابحث عن الفيبرونيل في الطماطم
• Find bifenthrin in cucumber

**2️⃣ عد العينات:**
• ما عدد عينات الخيار فوق الحد؟
• كم عينة طماطم وكوسة؟

**3️⃣ عينات بعدد مبيدات:**
• عينات تحتوي على 6 مبيدات
• عينات بمبيد واحد ومبيدين كل على حده

**4️⃣ قائمة المبيدات:**
• ماهي المبيدات في الهيل؟
• المبيدات في الفلفل

**5️⃣ تحليل الأحياء:**
• المبيدات في حي الإسكان
• ترتيب الأحياء حسب المخالفات

💬 أرسل سؤالك نصياً أو صوتياً!
"""
    await update.message.reply_text(welcome_msg, parse_mode='Markdown')


# ============================================================
# Main
# ============================================================

def main():
    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8393315665:AAG82MK45WQfIpn054HdoyxZK4rGtZLsL-A")
    
    app = ApplicationBuilder().token(TOKEN).build()
    
    # Add handlers
    from telegram.ext import CommandHandler
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    print("=" * 60)
    print("🚀 LARS Unified Voice Assistant (Hybrid Pipeline)")
    print("=" * 60)
    print("\n📊 **Architecture:**")
    print("   Layer 1: Quick Pattern Matching (90% of queries)")
    print("   Layer 2: Intent Classification + Entity Extraction")
    print("   Layer 3: SQL Generation (Template + LLM)")
    print("   Layer 4: Query Validation & Execution")
    print("   Layer 5: Response Generation")
    print("\n✅ **Supported Patterns:**")
    print("   • البحث عن مبيد في عينة")
    print("   • عد العينات فوق/تحت الحد")
    print("   • عينات بعدد مبيدات محدد")
    print("   • قائمة المبيدات في عينة")
    print("   • المبيدات في أحياء")
    print("   • ترتيب الأحياء حسب المخالفات")
    print("\n📈 **Expected Performance:**")
    print("   • Layer 1: 90% of queries, <50ms")
    print("   • Layer 2: 95% accuracy, <200ms")
    print("   • Layer 3+4: 98% success, <1s")
    print("=" * 60)
    
    app.run_polling()


if __name__ == '__main__':
    main()
