"""
voice_lars_unified.py
=====================
البوت الصوتي المحسن باستخدام النظام الموحد
يستخدم نفس الأنماط والكيانات مثل النظام النصي
"""

import os
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from faster_whisper import WhisperModel
from io import BytesIO

# استيراد النظام الموحد
from query_patterns_shared import correct_stt_text, SharedPatterns
from unified_router import UnifiedRouter

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

DB_PATH = "/Users/a12/Buraidah_lars/src/LARS/data/lars_data.duckdb"

# تهيئة الراوتر الموحد
router = UnifiedRouter(DB_PATH)

# إعدادات التلجرام
TOKEN = "8393315665:AAG82MK45WQfIpn054HdoyxZK4rGtZLsL-A"

# ============================================================
# Helper Functions
# ============================================================

def create_excel_report(df, filename="lars_report.xlsx"):
    """إنشاء ملف Excel من DataFrame"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='النتائج')
        
        worksheet = writer.sheets['النتائج']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    output.seek(0)
    return output


def transcribe_audio(file_path):
    """تحويل الصوت إلى نص مع تصحيح الأخطاء"""
    segments, info = stt_model.transcribe(
        file_path,
        beam_size=5,
        language="ar",
        initial_prompt="مبيدات حشرية: كلوربيريفوس، فبرونيل، ايميداكلوبريد، بوبروفيزين، دلتامثرن، طماطم، خيار، فاصوليا، هيل، كوسة",
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=True
    )
    
    raw_text = " ".join([s.text for s in segments])
    corrected_text = correct_stt_text(raw_text)
    
    return raw_text, corrected_text, info.language_probability


def detect_user_persona(text):
    """كشف شخصية المستخدم"""
    manager_keywords = ['كام', 'نسبة', 'إجمالي', 'اجمالي', 'ملخص', 'تقرير', 'كم', 'عدد']
    analyst_keywords = ['تحليل', 'مقارنة', 'اتجاه', 'trend', 'توزيع', 'إحصاء', 'تصنيف']
    
    text_lower = text.lower()
    
    if any(kw in text or kw in text_lower for kw in analyst_keywords):
        return 'analyst'
    elif any(kw in text or kw in text_lower for kw in manager_keywords):
        return 'manager'
    
    return 'technician'


# ============================================================
# Main Query Function (Using Unified Router)
# ============================================================

def answer_database_question(question, persona='manager'):
    """
    الإجابة على الاستعلام باستخدام النظام الموحد
    Returns: (text_response, excel_file, needs_dashboard)
    """
    # توجيه الاستعلام باستخدام الراوتر الموحد
    routing_result = router.route_query(question)
    
    print(f"🎯 النمط: {routing_result['pattern_name']}")
    print(f"🔍 الكيانات: عينات={routing_result['entities'].sample_types}, مبيدات={routing_result['entities'].pesticides}")
    print(f"📝 SQL: {routing_result['sql'][:100]}...")
    
    # تنفيذ الاستعلام
    df, error = router.execute_query(routing_result['sql'])
    
    if error:
        return f"⚠️ خطأ في تنفيذ الاستعلام:\n`{error}`\n\nSQL:\n`{routing_result['sql'][:200]}...`", None, False
    
    if df.empty:
        # معلومات مفيدة عند عدم وجود نتائج
        entities = routing_result['entities']
        msg = "لم أجد نتائج تطابق سؤالك."
        if entities.sample_types:
            msg += f"\n\n🔍 بحثت عن: {', '.join(entities.sample_types)}"
        if entities.pesticides:
            msg += f"\n🧪 المبيدات: {', '.join(entities.pesticides)}"
        return msg, None, False
    
    # تنسيق الرد
    text_response = router.format_response(df, routing_result)
    
    # تحديد إذا كان يحتاج ملف Excel
    should_send_excel = len(df) > 10 or routing_result['response_info']['should_generate_excel']
    excel_file = None
    
    if should_send_excel:
        excel_file = create_excel_report(df)
    
    # اقتراح لوحة التحكم للنتائج الكبيرة
    needs_dashboard = len(df) > 50
    
    return text_response, excel_file, needs_dashboard


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
        # تحويل الصوت إلى نص
        raw_text, corrected_text, confidence = transcribe_audio(file_path)
        
        # عرض النص المحول
        if corrected_text != raw_text:
            await update.message.reply_text(
                f"📝 النص المسموع: {raw_text}\n"
                f"✅ بعد التصحيح: {corrected_text}\n"
                f"🎯 الدقة: {confidence*100:.1f}%"
            )
        else:
            await update.message.reply_text(f"📝 النص: {corrected_text}")
        
        # كشف شخصية المستخدم
        persona = detect_user_persona(corrected_text)
        
        # البحث والإجابة
        await update.message.reply_text("🔍 جاري البحث...")
        
        text_response, excel_file, needs_dashboard = answer_database_question(
            corrected_text, 
            persona
        )
        
        # إرسال الرد النصي
        await update.message.reply_text(text_response, parse_mode='Markdown')
        
        # إرسال ملف Excel إذا كان موجوداً
        if excel_file:
            await update.message.reply_document(
                document=excel_file,
                filename="lars_report.xlsx",
                caption="📎 التقرير الكامل"
            )
        
        # اقتراح لوحة التحكم إذا لزم الأمر
        if needs_dashboard:
            await update.message.reply_text(
                "💡 **نصيحة:** لتحليل أعمق مع رسوم بيانية وخرائط، افتح لوحة التحكم في تطبيق LARS.",
                parse_mode='Markdown'
            )
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {error_details}")
        await update.message.reply_text(
            f"⚠️ عذراً، حدث خطأ:\n`{str(e)[:300]}`",
            parse_mode='Markdown'
        )
    
    finally:
        # تنظيف الملف المؤقت
        if os.path.exists(file_path):
            os.remove(file_path)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة الرسائل النصية"""
    text = update.message.text
    persona = detect_user_persona(text)
    
    await update.message.reply_text("🔍 جاري البحث...")
    
    try:
        text_response, excel_file, needs_dashboard = answer_database_question(text, persona)
        
        await update.message.reply_text(text_response, parse_mode='Markdown')
        
        if excel_file:
            await update.message.reply_document(
                document=excel_file,
                filename="lars_report.xlsx",
                caption="📎 التقرير الكامل"
            )
        
        if needs_dashboard:
            await update.message.reply_text(
                "💡 **نصيحة:** لتحليل أعمق مع رسوم بيانية وخرائط، افتح لوحة التحكم في تطبيق LARS.",
                parse_mode='Markdown'
            )
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {error_details}")
        await update.message.reply_text(
            f"⚠️ عذراً، حدث خطأ أثناء معالجة الاستعلام:\n"
            f"`{str(e)[:300]}`\n\n"
            f"الرجاء المحاولة مرة أخرى أو إعادة صياغة السؤال.",
            parse_mode='Markdown'
        )


# ============================================================
# Main
# ============================================================

def main():
    """الدالة الرئيسية"""
    print("=" * 60)
    print("🚀 LARS Voice Assistant - Unified System")
    print("=" * 60)
    print(f"📂 Database: {DB_PATH}")
    print(f"🤖 Bot Token: {TOKEN[:20]}...")
    print("✨ Mode: Unified Router (Voice + Text)")
    print("=" * 60)
    
    app = ApplicationBuilder().token(TOKEN).build()
    
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    print("🎤 Listening for messages...")
    app.run_polling()


if __name__ == '__main__':
    main()
