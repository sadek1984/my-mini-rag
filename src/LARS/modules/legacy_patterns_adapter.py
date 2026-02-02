"""
legacy_patterns_adapter.py
==========================
محول يستخدم نفس أنماط النظام القديم في البوت الصوتي
"""
import re
import duckdb
from typing import Dict, List, Optional, Tuple
import pandas as pd


class LegacyPatternsAdapter:
    """محول يستخدم الأنماط القديمة في النظام الجديد"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.patterns = self._load_legacy_patterns()

    def _load_legacy_patterns(self) -> Dict:
        """تحميل الأنماط من النظام القديم"""
        # هذه الأنماط مستخرجة مباشرة من handle_natural_language_query
        return {
            # === PATTERN 0 ===
            'count_samples_above_below': {
                'detect': lambda q: (
                    any(phrase in q.lower() for phrase in ['count', 'how many', 'عدد', 'كم', 'ماهي', 'ما هي'])
                    and any(phrase in q.lower() or phrase in q for phrase in ['فوق الحد', 'above limit', 'more than limit', 'exceed'])
                    and any(sample in q for sample in ['طماطم', 'خيار', 'كوسة', 'فاصوليا', 'فلفل', 'باذنجان'])
                ),
                'execute': self._execute_pattern_0,
                'description': 'عد العينات فوق/تحت الحد'
            },

            # === PATTERN 0.5 ===
            'pesticides_in_neighborhoods': {
                'detect': lambda q: (
                    ('المبيدات' in q or 'pesticides' in q.lower())
                    and ('حي' in q or 'neighborhood' in q.lower())
                ),
                'execute': self._execute_pattern_0_5,
                'description': 'المبيدات في الأحياء'
            },

            # === PATTERN 0.6 ===
            'samples_with_pesticide': {
                'detect': lambda q: (
                    ('عينات' in q or 'samples' in q.lower())
                    and ('تحتوي' in q or 'containing' in q.lower())
                    and ('مبيد' in q or 'pesticide' in q.lower())
                ),
                'execute': self._execute_pattern_0_6,
                'description': 'عينات تحتوي على مبيد'
            },

            # === PATTERN 1 ===
            'samples_with_n_pesticides': {
                'detect': lambda q: (
                    re.search(r'(\d+)\s*مبيد', q) or
                    re.search(r'mabid\s*(\d+)', q) or
                    re.search(r'(\d+)\s*pesticide', q.lower())
                ),
                'execute': self._execute_pattern_1,
                'description': 'عينات بعدد مبيدات محدد'
            },

            # === PATTERN 2 ===
            'pesticide_readings': {
                'detect': lambda q: (
                    ('ابحث عن' in q or 'find' in q.lower())
                    and ('مبيد' in q or 'pesticide' in q.lower())
                    and ('في' in q or 'in' in q.lower())
                ),
                'execute': self._execute_pattern_2,
                'description': 'قراءات مبيد محدد'
            },

            # === PATTERN 3 ===
            'list_pesticides_in_sample': {
                'detect': lambda q: (
                    ('المبيدات' in q and 'في' in q) or
                    ('المبيدات الموجوده' in q) or
                    ('ماهي المبيدات' in q) or
                    ('المبيدات التي ظهرت' in q) or
                    ('pesticides in' in q.lower())
                ),
                'execute': self._execute_pattern_3,
                'description': 'قائمة المبيدات في عينة'
            },

            # === PATTERN 4 ===
            'pesticides_multiple_neighborhoods': {
                'detect': lambda q: (
                    'المبيدات' in q and 'حي' in q and ('و' in q or 'and' in q.lower())
                ),
                'execute': self._execute_pattern_4,
                'description': 'المبيدات في أحياء متعددة'
            }
        }

    def match_and_execute(self, query: str) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
        """مطابقة الاستعلام وتنفيذه باستخدام الأنماط القديمة"""
        for pattern_name, pattern_info in self.patterns.items():
            if pattern_info['detect'](query):
                print(f"🎯 تطابق مع النمط: {pattern_info['description']}")
                return pattern_info['execute'](query)

        # إذا لم يتطابق مع أي نمط
        return None, None

    # ========== تنفيذ الأنماط ==========

    def _execute_pattern_0(self, query: str) -> Tuple[str, pd.DataFrame]:
        """تنفيذ Pattern 0: عد العينات فوق/تحت الحد"""
        # استخراج نوع العينة
        sample_types = ['طماطم', 'خيار', 'كوسة', 'فاصوليا', 'فلفل', 'باذنجان']
        detected_sample = next((st for st in sample_types if st in query), None)

        # تحديد فوق/تحت
        is_above = any(phrase in query.lower() or phrase in query 
                       for phrase in ['فوق', 'above', 'exceed', 'more than'])
        is_below = any(phrase in query.lower() or phrase in query 
                       for phrase in ['تحت', 'below', 'under', 'within'])

        if not detected_sample:
            return "⚠️ لم أتمكن من تحديد نوع العينة.", pd.DataFrame()

        sql = f"""
        SELECT 
            "اسم العينة" as نوع_العينة,
            COUNT(DISTINCT "كود العينة") as عدد_العينات,
            SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as فوق_الحد,
            SUM(CASE WHEN is_above_limit = 0 AND is_detected = 1 THEN 1 ELSE 0 END) as تحت_الحد
        FROM chemistry_tidy 
        WHERE is_detected = 1 
        AND "اسم العينة" LIKE '%{detected_sample}%'
        GROUP BY "اسم العينة"
        """

        df = self._execute_sql(sql)

        response = f"📊 **إحصائيات {detected_sample}:**\n\n"
        if not df.empty:
            total = df['عدد_العينات'].sum()
            above = df['فوق_الحد'].sum()
            below = df['تحت_الحد'].sum()

            response += f"• إجمالي العينات: {total}\n"
            if is_above:
                response += f"• فوق الحد: {above}\n"
            elif is_below:
                response += f"• تحت الحد: {below}\n"
            response += f"\n{df.to_markdown(index=False)}"

        return response, df

    def _execute_pattern_0_5(self, query: str) -> Tuple[str, pd.DataFrame]:
        """تنفيذ Pattern 0.5: مبيدات في أحياء"""
        # استخراج الأحياء
        neighborhoods = ['الإسكان', 'النهضة', 'الريان', 'الأخضر', 'الروضة', 'النخيل']
        detected_neighborhoods = [n for n in neighborhoods if n in query]

        if not detected_neighborhoods:
            return "⚠️ لم أتمكن من تحديد الأحياء.", pd.DataFrame()

        # بناء استعلام SQL
        neighborhood_conditions = []
        for n in detected_neighborhoods:
            neighborhood_conditions.append(f'"الحى" LIKE \'%{n}%\'')
        
        neighborhood_filter = f"({' OR '.join(neighborhood_conditions)})"

        sql = f"""
        SELECT 
            "الحى" as الحي,
            pesticide_name as المبيد,
            COUNT(*) as عدد_الكشف,
            SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) as عدد_المخالفات,
            ROUND(AVG(concentration), 4) as متوسط_التركيز
        FROM chemistry_tidy
        WHERE is_detected = 1 
        AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
        AND {neighborhood_filter}
        GROUP BY "الحى", pesticide_name
        ORDER BY "الحى", عدد_الكشف DESC
        """

        df = self._execute_sql(sql)

        response = f"🏘️ **المبيدات في الأحياء:**\n\n"
        if not df.empty:
            for neighborhood in detected_neighborhoods:
                neighborhood_df = df[df['الحي'].str.contains(neighborhood, na=False)]
                if not neighborhood_df.empty:
                    response += f"**📍 {neighborhood}:**\n"
                    response += neighborhood_df.head(5).to_markdown(index=False) + "\n\n"

        return response, df

    def _execute_pattern_0_6(self, query: str) -> Tuple[str, pd.DataFrame]:
        """تنفيذ Pattern 0.6: عينات تحتوي على مبيد"""
        # استخراج المبيد
        pesticides_ar = ['بيفنثرين', 'كلوربيريفوس', 'إيميداكلوبريد', 'فيبرونيل', 
                         'دلتامثرين', 'سيبرمثرين', 'كاربندازيم']
        pesticides_en = ['bifenthrin', 'chlorpyrifos', 'imidacloprid', 'fipronil', 
                         'deltamethrin', 'cypermethrin', 'carbendazim']
        
        detected_pesticide = None
        for pest in pesticides_ar:
            if pest in query:
                detected_pesticide = pest
                break
        
        if not detected_pesticide:
            for pest in pesticides_en:
                if pest in query.lower():
                    detected_pesticide = pest
                    break

        # استخراج نوع العينة
        sample_types = ['طماطم', 'خيار', 'كوسة', 'فاصوليا', 'فلفل', 'باذنجان', 'هيل']
        detected_sample = next((st for st in sample_types if st in query), None)

        if not detected_pesticide:
            return "⚠️ لم أتمكن من تحديد المبيد.", pd.DataFrame()

        sql = f"""
        SELECT 
            "كود العينة" as كود_العينة,
            "اسم العينة" as اسم_العينة,
            pesticide_name as المبيد,
            concentration as التركيز,
            limit_value as الحد_المسموح,
            CASE WHEN is_above_limit = 1 THEN 'فوق الحد' ELSE 'تحت الحد' END as الحالة
        FROM chemistry_tidy
        WHERE is_detected = 1 
        AND pesticide_name ILIKE '%{detected_pesticide}%'
        """
        
        if detected_sample:
            sql += f" AND \"اسم العينة\" LIKE '%{detected_sample}%'"
        
        sql += " ORDER BY concentration DESC LIMIT 20"

        df = self._execute_sql(sql)

        response = f"🔍 **عينات تحتوي على {detected_pesticide}:**\n\n"
        if not df.empty:
            response += df.to_markdown(index=False)
        else:
            response += "لم أجد عينات تحتوي على هذا المبيد."

        return response, df

    def _execute_pattern_1(self, query: str) -> Tuple[str, pd.DataFrame]:
        """تنفيذ Pattern 1: عينات بعدد مبيدات محدد"""
        # استخراج العدد
        match = re.search(r'(\d+)\s*مبيد', query)
        if not match:
            match = re.search(r'(\d+)\s*pesticide', query.lower())
        
        if not match:
            return "⚠️ لم أتمكن من تحديد عدد المبيدات.", pd.DataFrame()
        
        n_pesticides = int(match.group(1))

        # استخراج نوع العينة
        sample_types = ['طماطم', 'خيار', 'كوسة', 'فاصوليا', 'فلفل', 'باذنجان', 'هيل']
        detected_sample = next((st for st in sample_types if st in query), None)

        sql = f"""
        SELECT 
            "كود العينة" as sample_code,
            "اسم العينة" as sample_name,
            COUNT(*) as pesticide_count
        FROM chemistry_tidy
        WHERE is_detected = 1 
        AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
        """
        
        if detected_sample:
            sql += f" AND \"اسم العينة\" LIKE '%{detected_sample}%'"
        
        sql += f"""
        GROUP BY "كود العينة", "اسم العينة"
        HAVING COUNT(*) = {n_pesticides}
        ORDER BY "كود العينة" DESC
        LIMIT 20
        """

        df = self._execute_sql(sql)

        response = f"🔢 **عينات تحتوي على {n_pesticides} مبيدات:**\n\n"
        if not df.empty:
            response += f"✅ وجدت {len(df)} عينة.\n\n"
            response += df.to_markdown(index=False)
        else:
            response += "⚠️ لم أجد عينات بهذا العدد من المبيدات."

        return response, df

    def _execute_pattern_2(self, query: str) -> Tuple[str, pd.DataFrame]:
        """تنفيذ Pattern 2: قراءات مبيد محدد"""
        return self._execute_pattern_0_6(query)  # نفس المنطق تقريباً

    def _execute_pattern_3(self, query: str) -> Tuple[str, pd.DataFrame]:
        """تنفيذ Pattern 3: قائمة المبيدات في عينة"""
        # استخراج نوع العينة
        sample_types = ['طماطم', 'خيار', 'كوسة', 'فاصوليا', 'فلفل', 'باذنجان', 'هيل', 'هيل هندي']
        detected_sample = next((st for st in sample_types if st in query), None)

        if not detected_sample:
            return "⚠️ لم أتمكن من تحديد نوع العينة.", pd.DataFrame()

        sql = f"""
        SELECT 
            pesticide_name as المبيد,
            COUNT(*) as عدد_التكرار,
            ROUND(AVG(concentration), 3) as متوسط_التركيز,
            ROUND(MAX(concentration), 3) as أعلى_تركيز,
            SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as عدد_التجاوزات
        FROM chemistry_tidy
        WHERE "اسم العينة" LIKE '%{detected_sample}%'
        AND is_detected = 1
        GROUP BY pesticide_name
        ORDER BY عدد_التكرار DESC, عدد_التجاوزات DESC
        LIMIT 20
        """

        df = self._execute_sql(sql)

        response = f"🧪 **المبيدات في {detected_sample}:**\n\n"
        if not df.empty:
            total_detections = df['عدد_التكرار'].sum()
            total_violations = df['عدد_التجاوزات'].sum()
            unique_pesticides = len(df)

            response += f"📊 **الملخص:** {unique_pesticides} مبيد، {total_detections} اكتشاف"
            if total_violations > 0:
                response += f"، {total_violations} تجاوز\n\n"
            else:
                response += "\n\n"
            response += df.to_markdown(index=False)
        else:
            response += "⚠️ لم أجد أي مبيدات في هذه العينة."

        return response, df

    def _execute_pattern_4(self, query: str) -> Tuple[str, pd.DataFrame]:
        """تنفيذ Pattern 4: مبيدات في أحياء متعددة"""
        return self._execute_pattern_0_5(query)  # نفس المنطق

    def _execute_sql(self, sql: str) -> pd.DataFrame:
        """تنفيذ استعلام SQL"""
        try:
            con = duckdb.connect(self.db_path, read_only=True)
            df = con.execute(sql).df()
            con.close()
            return df
        except Exception as e:
            print(f"❌ خطأ SQL: {e}")
            return pd.DataFrame()
