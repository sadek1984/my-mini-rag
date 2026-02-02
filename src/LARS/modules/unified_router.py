"""
unified_router.py
=================
الراوتر الموحد الذي يربط النظام الصوتي بالنظام النصي القديم
أي تعديل على الأنماط هنا ينعكس تلقائياً على كلا النظامين
"""

import duckdb
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import re
import os

# استيراد من قاعدة الأنماط المشتركة
from query_patterns_shared import EntityExtractor, ExtractedEntities, SharedPatterns


class UnifiedRouter:
    """الراوتر الموحد لجميع الاستعلامات"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.extractor = EntityExtractor()
        
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        توجيه الاستعلام وإرجاع:
        - نوع النمط
        - الكيانات
        - SQL المناسب
        - معلومات إضافية للرد
        """
        # استخراج الكيانات
        entities = self.extractor.extract_all(query)
        
        # تحديد النمط المناسب
        pattern_info = self._determine_pattern(query, entities)
        
        # إنشاء SQL بناءً على النمط
        sql_query = self._generate_sql(pattern_info['pattern'], entities)
        
        # معلومات للرد
        response_info = self._prepare_response_info(pattern_info, entities)
        
        return {
            'pattern': pattern_info['pattern'],
            'pattern_name': pattern_info['name'],
            'entities': entities,
            'sql': sql_query,
            'response_info': response_info,
            'query_type': pattern_info.get('type', 'SQL_QUERY')
        }
    
    def _determine_pattern(self, query: str, entities: ExtractedEntities) -> Dict:
        """تحديد النمط المناسب للاستعلام"""
        query_lower = query.lower()
        
        # النمط 1: البحث عن مبيد في عينة
        if entities.pesticides and entities.sample_types:
            return {
                'pattern': 'pesticide_in_sample',
                'name': 'بحث عن مبيد في عينة',
                'type': 'SQL_QUERY'
            }
        
        # النمط 2: عد العينات حسب النوع
        if entities.aggregation_type == 'count' and entities.sample_types:
            return {
                'pattern': 'count_samples',
                'name': 'عد العينات',
                'type': 'SQL_QUERY'
            }
        
        # النمط 3: عينات بعدد مبيدات محدد
        if entities.pesticide_count is not None:
            return {
                'pattern': 'pesticide_count',
                'name': 'عينات بعدد مبيدات',
                'type': 'SQL_QUERY'
            }
        
        # النمط 4: مبيدات في أحياء
        if entities.pesticides and entities.neighborhoods:
            return {
                'pattern': 'pesticide_in_neighborhood',
                'name': 'مبيدات في أحياء',
                'type': 'SQL_QUERY'
            }
        
        # النمط 5: إحصائيات تركيز
        if entities.aggregation_type == 'stats' and entities.pesticides:
            return {
                'pattern': 'concentration_stats',
                'name': 'إحصائيات تركيز',
                'type': 'SQL_QUERY'
            }
        
        # النمط 6: عينات غير مطابقة
        if entities.limit_condition == 'above':
            return {
                'pattern': 'non_compliant',
                'name': 'عينات غير مطابقة',
                'type': 'SQL_QUERY'
            }
        
        # النمط 7: تصنيف كيميائي
        if entities.needs_classification:
            return {
                'pattern': 'chemical_classification',
                'name': 'تصنيف كيميائي',
                'type': 'SQL_QUERY'
            }
        
        # النمط 8: عينات فريدة
        if entities.unique_samples:
            return {
                'pattern': 'unique_samples',
                'name': 'عينات فريدة',
                'type': 'SQL_QUERY'
            }
        
        # النمط 9: بحث حسب المستلم
        if entities.receivers:
            return {
                'pattern': 'receiver_search',
                'name': 'بحث حسب المستلم',
                'type': 'SQL_QUERY'
            }
        
        # النمط 10: بحث حسب المنشأة
        if entities.facilities:
            return {
                'pattern': 'facility_search',
                'name': 'بحث حسب المنشأة',
                'type': 'SQL_QUERY'
            }
        
        # النمط 11: بحث عن مبيد فقط
        if entities.pesticides:
            return {
                'pattern': 'pesticide_only',
                'name': 'بحث عن مبيد',
                'type': 'SQL_QUERY'
            }
        
        # النمط 12: بحث عن نوع عينة فقط
        if entities.sample_types:
            return {
                'pattern': 'sample_type_only',
                'name': 'بحث عن نوع عينة',
                'type': 'SQL_QUERY'
            }
        
        # النمط 13: بحث في حي
        if entities.neighborhoods:
            return {
                'pattern': 'neighborhood_search',
                'name': 'بحث في حي',
                'type': 'SQL_QUERY'
            }
        
        # النمط الافتراضي: استعلام عام
        return {
            'pattern': 'general_query',
            'name': 'استعلام عام',
            'type': 'SQL_QUERY'
        }
    
    def _generate_sql(self, pattern: str, entities: ExtractedEntities) -> str:
        """إنشاء SQL بناءً على النمط والكيانات"""
        sql_generators = {
            'pesticide_in_sample': self._sql_pesticide_in_sample,
            'count_samples': self._sql_count_samples,
            'pesticide_count': self._sql_pesticide_count,
            'pesticide_in_neighborhood': self._sql_pesticide_in_neighborhood,
            'concentration_stats': self._sql_concentration_stats,
            'non_compliant': self._sql_non_compliant,
            'chemical_classification': self._sql_chemical_classification,
            'unique_samples': self._sql_unique_samples,
            'receiver_search': self._sql_receiver_search,
            'facility_search': self._sql_facility_search,
            'pesticide_only': self._sql_pesticide_only,
            'sample_type_only': self._sql_sample_type_only,
            'neighborhood_search': self._sql_neighborhood_search,
            'general_query': self._sql_general_query,
        }
        
        generator = sql_generators.get(pattern, self._sql_general_query)
        return generator(entities)
    
    def _sql_pesticide_in_sample(self, entities: ExtractedEntities) -> str:
        """SQL للبحث عن مبيد في عينة"""
        pesticides = entities.pesticides
        sample_types = entities.sample_types
        limit_condition = entities.limit_condition
        
        where_clauses = ["is_detected = 1"]
        
        if pesticides:
            pest_conditions = [f"pesticide_name ILIKE '%{p}%'" for p in pesticides]
            where_clauses.append(f"({' OR '.join(pest_conditions)})")
        
        if sample_types:
            sample_conditions = [f'"اسم العينة" LIKE \'%{s}%\'' for s in sample_types]
            where_clauses.append(f"({' OR '.join(sample_conditions)})")
        
        if limit_condition == 'above':
            where_clauses.append("is_above_limit = 1")
        elif limit_condition == 'below':
            where_clauses.append("is_above_limit = 0")
        
        where_sql = " AND ".join(where_clauses)
        
        return f"""
SELECT 
    "كود العينة" as كود_العينة,
    "اسم العينة" as اسم_العينة,
    "الحى" as الحي,
    pesticide_name as المبيد,
    concentration as التركيز,
    limit_value as الحد_المسموح,
    ROUND(exceedance_ratio, 2) as نسبة_التجاوز,
    CASE WHEN is_above_limit = 1 THEN 'فوق الحد' ELSE 'تحت الحد' END as الحالة
FROM chemistry_tidy
WHERE {where_sql}
ORDER BY concentration DESC
LIMIT 100
"""
    
    def _sql_count_samples(self, entities: ExtractedEntities) -> str:
        """SQL لعد العينات"""
        sample_types = entities.sample_types
        neighborhoods = entities.neighborhoods
        limit_condition = entities.limit_condition
        
        where_clauses = ["is_detected = 1"]
        
        if sample_types:
            sample_conditions = [f'"اسم العينة" LIKE \'%{s}%\'' for s in sample_types]
            where_clauses.append(f"({' OR '.join(sample_conditions)})")
        
        if neighborhoods:
            neighborhood_conditions = [f'"الحى" LIKE \'%{n}%\'' for n in neighborhoods]
            where_clauses.append(f"({' OR '.join(neighborhood_conditions)})")
        
        if limit_condition == 'above':
            where_clauses.append("is_above_limit = 1")
        elif limit_condition == 'below':
            where_clauses.append("is_above_limit = 0")
        
        where_sql = " AND ".join(where_clauses)
        
        return f"""
SELECT 
    "اسم العينة" as نوع_العينة,
    COUNT(DISTINCT "كود العينة") as عدد_العينات,
    SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as فوق_الحد,
    SUM(CASE WHEN is_above_limit = 0 AND is_detected = 1 THEN 1 ELSE 0 END) as تحت_الحد
FROM chemistry_tidy
WHERE {where_sql}
GROUP BY "اسم العينة"
ORDER BY عدد_العينات DESC
"""
    
    def _sql_pesticide_count(self, entities: ExtractedEntities) -> str:
        """SQL للعينات بعدد مبيدات محدد"""
        count = entities.pesticide_count
        sample_types = entities.sample_types
        needs_classification = entities.needs_classification
        
        where_clauses = [
            "is_detected = 1",
            "pesticide_name NOT IN ('NO DETECTION', 'NO DATA')"
        ]
        
        if sample_types:
            sample_conditions = [f'"اسم العينة" LIKE \'%{s}%\'' for s in sample_types]
            where_clauses.append(f"({' OR '.join(sample_conditions)})")
        
        where_sql = " AND ".join(where_clauses)
        
        if needs_classification:
            # مع التصنيف الكيميائي
            return f"""
WITH sample_pesticides AS (
    SELECT 
        "كود العينة" as sample_code,
        "اسم العينة" as sample_name,
        pesticide_name,
        COUNT(*) OVER (PARTITION BY "كود العينة") as total_pesticides
    FROM chemistry_tidy
    WHERE {where_sql}
),
filtered_samples AS (
    SELECT DISTINCT sample_code, sample_name, total_pesticides
    FROM sample_pesticides
    WHERE total_pesticides = {count}
)
SELECT 
    fs.sample_code as كود_العينة,
    fs.sample_name as اسم_العينة,
    fs.total_pesticides as عدد_المبيدات,
    STRING_AGG(sp.pesticide_name, ', ') as المبيدات
FROM filtered_samples fs
JOIN sample_pesticides sp ON fs.sample_code = sp.sample_code
GROUP BY fs.sample_code, fs.sample_name, fs.total_pesticides
ORDER BY fs.sample_code DESC
"""
        else:
            return f"""
SELECT 
    "كود العينة" as كود_العينة,
    "اسم العينة" as اسم_العينة,
    COUNT(*) as عدد_المبيدات
FROM chemistry_tidy
WHERE {where_sql}
GROUP BY "كود العينة", "اسم العينة"
HAVING COUNT(*) = {count}
ORDER BY "كود العينة" DESC
"""
    
    def _sql_pesticide_in_neighborhood(self, entities: ExtractedEntities) -> str:
        """SQL للمبيدات في أحياء"""
        pesticides = entities.pesticides
        neighborhoods = entities.neighborhoods
        
        where_clauses = [
            "is_detected = 1",
            "pesticide_name NOT IN ('NO DETECTION', 'NO DATA')"
        ]
        
        if pesticides:
            pest_conditions = [f"pesticide_name ILIKE '%{p}%'" for p in pesticides]
            where_clauses.append(f"({' OR '.join(pest_conditions)})")
        
        if neighborhoods:
            neighborhood_conditions = [f'"الحى" LIKE \'%{n}%\'' for n in neighborhoods]
            where_clauses.append(f"({' OR '.join(neighborhood_conditions)})")
        
        where_sql = " AND ".join(where_clauses)
        
        return f"""
SELECT 
    "الحى" as الحي,
    pesticide_name as المبيد,
    COUNT(*) as عدد_الكشف,
    SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) as عدد_المخالفات,
    ROUND(AVG(concentration), 4) as متوسط_التركيز
FROM chemistry_tidy
WHERE {where_sql}
GROUP BY "الحى", pesticide_name
ORDER BY "الحى", عدد_الكشف DESC
"""
    
    def _sql_concentration_stats(self, entities: ExtractedEntities) -> str:
        """SQL لإحصائيات التركيز"""
        pesticides = entities.pesticides
        sample_types = entities.sample_types
        
        where_clauses = ["is_detected = 1"]
        
        if pesticides:
            pest_conditions = [f"pesticide_name ILIKE '%{p}%'" for p in pesticides]
            where_clauses.append(f"({' OR '.join(pest_conditions)})")
        
        if sample_types:
            sample_conditions = [f'"اسم العينة" LIKE \'%{s}%\'' for s in sample_types]
            where_clauses.append(f"({' OR '.join(sample_conditions)})")
        
        where_sql = " AND ".join(where_clauses)
        
        return f"""
SELECT 
    pesticide_name as المبيد,
    COUNT(*) as عدد_القياسات,
    ROUND(MIN(concentration), 4) as أقل_تركيز,
    ROUND(MAX(concentration), 4) as أعلى_تركيز,
    ROUND(AVG(concentration), 4) as متوسط_التركيز,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY concentration), 4) as الوسيط
FROM chemistry_tidy
WHERE {where_sql}
GROUP BY pesticide_name
ORDER BY أعلى_تركيز DESC
"""
    
    def _sql_non_compliant(self, entities: ExtractedEntities) -> str:
        """SQL للعينات غير المطابقة"""
        sample_types = entities.sample_types
        neighborhoods = entities.neighborhoods
        
        where_clauses = ["is_compliant = 0", "is_detected = 1"]
        
        if sample_types:
            sample_conditions = [f'"اسم العينة" LIKE \'%{s}%\'' for s in sample_types]
            where_clauses.append(f"({' OR '.join(sample_conditions)})")
        
        if neighborhoods:
            neighborhood_conditions = [f'"الحى" LIKE \'%{n}%\'' for n in neighborhoods]
            where_clauses.append(f"({' OR '.join(neighborhood_conditions)})")
        
        where_sql = " AND ".join(where_clauses)
        
        return f"""
SELECT 
    "كود العينة" as كود_العينة,
    "اسم العينة" as اسم_العينة,
    "الحى" as الحي,
    pesticide_name as المبيد,
    concentration as التركيز,
    limit_value as الحد_المسموح,
    ROUND(exceedance_ratio, 2) as نسبة_التجاوز
FROM chemistry_tidy
WHERE {where_sql}
ORDER BY exceedance_ratio DESC
LIMIT 100
"""
    
    def _sql_chemical_classification(self, entities: ExtractedEntities) -> str:
        """SQL للتصنيف الكيميائي"""
        sample_types = entities.sample_types
        pesticide_count = entities.pesticide_count
        
        where_clauses = ["is_detected = 1"]
        
        if sample_types:
            sample_conditions = [f'"اسم العينة" LIKE \'%{s}%\'' for s in sample_types]
            where_clauses.append(f"({' OR '.join(sample_conditions)})")
        
        where_sql = " AND ".join(where_clauses)
        
        having_clause = ""
        if pesticide_count:
            having_clause = f"HAVING COUNT(DISTINCT pesticide_name) = {pesticide_count}"
        
        return f"""
SELECT 
    "كود العينة" as كود_العينة,
    "اسم العينة" as اسم_العينة,
    COUNT(DISTINCT pesticide_name) as عدد_المبيدات,
    STRING_AGG(DISTINCT pesticide_name, ', ') as المبيدات
FROM chemistry_tidy
WHERE {where_sql}
GROUP BY "كود العينة", "اسم العينة"
{having_clause}
ORDER BY عدد_المبيدات DESC
LIMIT 50
"""
    
    def _sql_unique_samples(self, entities: ExtractedEntities) -> str:
        """SQL للعينات الفريدة"""
        sample_types = entities.sample_types
        limit_condition = entities.limit_condition
        
        where_clauses = ["is_detected = 1"]
        
        if sample_types:
            sample_conditions = [f'"اسم العينة" LIKE \'%{s}%\'' for s in sample_types]
            where_clauses.append(f"({' OR '.join(sample_conditions)})")
        
        if limit_condition == 'above':
            where_clauses.append("is_above_limit = 1")
        elif limit_condition == 'below':
            where_clauses.append("is_above_limit = 0")
        
        where_sql = " AND ".join(where_clauses)
        
        return f"""
SELECT DISTINCT
    "كود العينة" as كود_العينة,
    "اسم العينة" as اسم_العينة,
    "الحى" as الحي,
    "اسم المنشاة" as المنشأة,
    "اسم المستلم" as المستلم
FROM chemistry_tidy
WHERE {where_sql}
ORDER BY "كود العينة" DESC
"""
    
    def _sql_receiver_search(self, entities: ExtractedEntities) -> str:
        """SQL للبحث حسب المستلم"""
        receivers = entities.receivers
        limit_condition = entities.limit_condition
        
        where_clauses = ["is_detected = 1"]
        
        if receivers:
            receiver_conditions = []
            for receiver in receivers:
                name_parts = receiver.split()
                for part in name_parts:
                    receiver_conditions.append(f'"اسم المستلم" LIKE \'%{part}%\'')
            where_clauses.append(f"({' OR '.join(receiver_conditions)})")
        
        if limit_condition == 'above':
            where_clauses.append("is_above_limit = 1")
        elif limit_condition == 'below':
            where_clauses.append("is_above_limit = 0")
        
        where_sql = " AND ".join(where_clauses)
        
        return f"""
SELECT DISTINCT
    "كود العينة" as كود_العينة,
    "اسم العينة" as اسم_العينة,
    "الحى" as الحي,
    "اسم المنشاة" as المنشأة,
    "اسم المستلم" as المستلم,
    COUNT(*) OVER (PARTITION BY "كود العينة") as عدد_المبيدات
FROM chemistry_tidy
WHERE {where_sql}
ORDER BY "كود العينة" DESC
"""
    
    def _sql_facility_search(self, entities: ExtractedEntities) -> str:
        """SQL للبحث حسب المنشأة"""
        facilities = entities.facilities
        
        where_clauses = ["is_detected = 1"]
        
        if facilities:
            facility_conditions = [f'"اسم المنشاة" LIKE \'%{f}%\'' for f in facilities]
            where_clauses.append(f"({' OR '.join(facility_conditions)})")
        
        where_sql = " AND ".join(where_clauses)
        
        return f"""
SELECT DISTINCT
    "كود العينة" as كود_العينة,
    "اسم العينة" as اسم_العينة,
    "الحى" as الحي,
    "اسم المنشاة" as المنشأة,
    COUNT(*) OVER (PARTITION BY "كود العينة") as عدد_المبيدات
FROM chemistry_tidy
WHERE {where_sql}
ORDER BY "كود العينة" DESC
"""
    
    def _sql_pesticide_only(self, entities: ExtractedEntities) -> str:
        """SQL للبحث عن مبيد فقط"""
        pesticides = entities.pesticides
        
        where_clauses = ["is_detected = 1"]
        
        if pesticides:
            pest_conditions = [f"pesticide_name ILIKE '%{p}%'" for p in pesticides]
            where_clauses.append(f"({' OR '.join(pest_conditions)})")
        
        where_sql = " AND ".join(where_clauses)
        
        return f"""
SELECT 
    "كود العينة" as كود_العينة,
    "اسم العينة" as اسم_العينة,
    "الحى" as الحي,
    pesticide_name as المبيد,
    concentration as التركيز,
    limit_value as الحد_المسموح,
    CASE WHEN is_above_limit = 1 THEN 'فوق الحد' ELSE 'تحت الحد' END as الحالة
FROM chemistry_tidy
WHERE {where_sql}
ORDER BY concentration DESC
LIMIT 100
"""
    
    def _sql_sample_type_only(self, entities: ExtractedEntities) -> str:
        """SQL لبحث عن نوع عينة"""
        sample_types = entities.sample_types
        
        where_clauses = ["is_detected = 1"]
        
        if sample_types:
            sample_conditions = [f'"اسم العينة" LIKE \'%{s}%\'' for s in sample_types]
            where_clauses.append(f"({' OR '.join(sample_conditions)})")
        
        where_sql = " AND ".join(where_clauses)
        
        return f"""
SELECT 
    "اسم العينة" as نوع_العينة,
    COUNT(DISTINCT "كود العينة") as عدد_العينات,
    COUNT(DISTINCT pesticide_name) as عدد_المبيدات_المختلفة,
    SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as فوق_الحد,
    SUM(CASE WHEN is_above_limit = 0 THEN 1 ELSE 0 END) as تحت_الحد
FROM chemistry_tidy
WHERE {where_sql}
GROUP BY "اسم العينة"
ORDER BY عدد_العينات DESC
"""
    
    def _sql_neighborhood_search(self, entities: ExtractedEntities) -> str:
        """SQL للبحث في حي"""
        neighborhoods = entities.neighborhoods
        
        where_clauses = ["is_detected = 1"]
        
        if neighborhoods:
            neighborhood_conditions = [f'"الحى" LIKE \'%{n}%\'' for n in neighborhoods]
            where_clauses.append(f"({' OR '.join(neighborhood_conditions)})")
        
        where_sql = " AND ".join(where_clauses)
        
        return f"""
SELECT 
    "الحى" as الحي,
    "اسم العينة" as نوع_العينة,
    COUNT(DISTINCT "كود العينة") as عدد_العينات,
    SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) as عدد_المخالفات
FROM chemistry_tidy
WHERE {where_sql}
GROUP BY "الحى", "اسم العينة"
ORDER BY عدد_العينات DESC
"""
    
    def _sql_general_query(self, entities: ExtractedEntities) -> str:
        """SQL للاستعلام العام"""
        return """
SELECT 
    "كود العينة" as كود_العينة,
    "اسم العينة" as اسم_العينة,
    "الحى" as الحي,
    pesticide_name as المبيد,
    concentration as التركيز,
    limit_value as الحد_المسموح,
    ROUND(exceedance_ratio, 2) as نسبة_التجاوز,
    CASE WHEN is_compliant = 1 THEN 'مطابق' ELSE 'غير مطابق' END as الحالة
FROM chemistry_tidy
WHERE is_detected = 1
ORDER BY "كود العينة" DESC
LIMIT 50
"""
    
    def _prepare_response_info(self, pattern_info: Dict, entities: ExtractedEntities) -> Dict:
        """تحضير معلومات إضافية للرد"""
        response_info = {
            'should_generate_excel': False,
            'excel_filename': 'lars_report.xlsx',
            'summary_type': 'detailed',
            'persona_hint': 'manager'
        }
        
        # تحديد إذا كان يحتاج ملف Excel
        if pattern_info['pattern'] in ['chemical_classification', 'pesticide_in_neighborhood', 'concentration_stats']:
            response_info['should_generate_excel'] = True
        
        # تحديد نوع الملخص
        if pattern_info['pattern'] in ['count_samples', 'non_compliant']:
            response_info['summary_type'] = 'statistical'
        elif pattern_info['pattern'] in ['chemical_classification', 'concentration_stats']:
            response_info['summary_type'] = 'analytical'
        
        # تلميحة الشخصية
        if pattern_info['pattern'] in ['count_samples', 'non_compliant']:
            response_info['persona_hint'] = 'manager'
        elif pattern_info['pattern'] in ['chemical_classification', 'concentration_stats']:
            response_info['persona_hint'] = 'analyst'
        elif pattern_info['pattern'] in ['pesticide_in_sample', 'general_query']:
            response_info['persona_hint'] = 'technician'
        
        return response_info
    
    def execute_query(self, sql: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """تنفيذ الاستعلام وإرجاع النتيجة"""
        try:
            con = duckdb.connect(self.db_path, read_only=True)
            df = con.execute(sql).df()
            con.close()
            return df, None
        except Exception as e:
            return pd.DataFrame(), str(e)
    
    def format_response(self, df: pd.DataFrame, routing_result: Dict) -> str:
        """تنسيق الرد بناءً على نتيجة التوجيه"""
        if df.empty:
            return "لم أجد نتائج تطابق سؤالك."
        
        row_count = len(df)
        pattern = routing_result['pattern']
        
        # بناء الرد المناسب لكل نمط
        if pattern == 'count_samples':
            total_samples = df['عدد_العينات'].sum() if 'عدد_العينات' in df.columns else row_count
            return f"📊 **إجمالي العينات:** {total_samples} عينة\n\n{df.to_markdown(index=False)}"
        
        elif pattern == 'pesticide_in_sample':
            if row_count <= 10:
                return f"🔍 **النتائج ({row_count} سجل):**\n\n{df.to_markdown(index=False)}"
            else:
                summary = df.head(5).to_markdown(index=False)
                return f"🔍 **وجدت {row_count} نتيجة. أول 5:**\n\n{summary}\n\n💡 *سأرسل التقرير الكامل كملف Excel.*"
        
        elif pattern == 'chemical_classification':
            return f"🧪 **التصنيف الكيميائي:**\n\n{df.to_markdown(index=False)}"
        
        elif pattern == 'concentration_stats':
            return f"📈 **إحصائيات التركيز:**\n\n{df.to_markdown(index=False)}"
        
        elif pattern == 'non_compliant':
            violations = row_count
            return f"⚠️ **المخالفات:** {violations} مخالفة\n\n{df.to_markdown(index=False)}"
        
        elif pattern == 'pesticide_count':
            return f"🧪 **عينات بعدد مبيدات محدد ({row_count} عينة):**\n\n{df.to_markdown(index=False)}"
        
        else:
            if row_count <= 15:
                return f"📋 **النتائج ({row_count} سجل):**\n\n{df.to_markdown(index=False)}"
            else:
                summary = df.head(10).to_markdown(index=False)
                return f"📋 **وجدت {row_count} نتيجة. أول 10:**\n\n{summary}\n\n💡 *لمشاهدة النتائج كاملة، سأرسل ملف Excel.*"


# للاختبار
if __name__ == "__main__":
    DB_PATH = "/Users/a12/Buraidah_lars/src/LARS/data/lars_data.duckdb"
    
    if not os.path.exists(DB_PATH):
        print(f"❌ قاعدة البيانات غير موجودة: {DB_PATH}")
    else:
        router = UnifiedRouter(DB_PATH)
        
        test_queries = [
            "ما عدد عينات الطماطم",
            "ابحث عن الفيبرونيل في الخيار",
            "ما هي العينات التي تحتوي على 10 مبيدات",
            "عينات غير مطابقة في حي النهضة",
            "ما هي المبيدات في الهيل",
        ]
        
        print("=" * 60)
        print("اختبار الراوتر الموحد")
        print("=" * 60)
        
        for query in test_queries:
            print(f"\n📝 الاستعلام: {query}")
            result = router.route_query(query)
            print(f"   🎯 النمط: {result['pattern_name']}")
            print(f"   📊 الكيانات: {result['entities'].sample_types}, {result['entities'].pesticides}")
            
            # تنفيذ الاستعلام
            df, error = router.execute_query(result['sql'])
            if error:
                print(f"   ❌ خطأ: {error}")
            else:
                print(f"   ✅ النتائج: {len(df)} سجل")
