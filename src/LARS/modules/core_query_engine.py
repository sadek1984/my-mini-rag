"""
core_query_engine.py
====================
محرك استعلامات أساسي خالٍ من Streamlit
يمكن استخدامه من:
- التطبيق الرئيسي (ai_assistant.py)
- البوت الصوتي (voice_bot_enhanced.py)
"""
import re
import duckdb
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "lars_data.duckdb"


class CoreQueryEngine:
    """محرك استعلامات مركزي - بدون Streamlit"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self._load_mappings()
    
    def _load_mappings(self):
        """تحميل الخرائط"""
        # Sample types
        self.sample_types = {
            'خيار': 'خيار', 'طماطم': 'طماطم', 'باذنجان': 'باذنجان',
            'فلفل': 'فلفل', 'كوسة': 'كوسة', 'جزر': 'جزر',
            'خس': 'خس', 'بقدونس': 'بقدونس', 'سبانخ': 'سبانخ',
            'بطاطس': 'بطاطس', 'فاصوليا': 'فاصوليا', 'فاصووليا': 'فاصوليا',
            'بامية': 'بامية', 'بروكلي': 'بروكلي', 'بصل': 'بصل',
            'جرجير': 'جرجير', 'زهرة': 'زهرة', 'شبت': 'شبت',
            'كزبرة': 'كزبرة', 'ملوخية': 'ملوخية', 'نعناع': 'نعناع',
            'فراولة': 'فراولة', 'عنب': 'عنب', 'تفاح': 'تفاح',
            'برتقال': 'برتقال', 'رمان': 'رمان', 'كمثرى': 'كمثرى',
            'ليمون': 'ليمون', 'توت': 'توت', 'تمر': 'تمر',
            'قمح': 'قمح', 'رز': 'رز', 'ذرة': 'ذرة',
            'عدس': 'عدس', 'دقيق': 'دقيق', 'طحين': 'طحين',
            'شوفان': 'شوفان', 'كمون': 'كمون', 'زعتر': 'زعتر',
            'بهارات': 'بهارات', 'توابل': 'توابل', 'بهار': 'توابل',
            'البهارات': 'توابل', 'قرنفل': 'قرنفل', 'هيل': 'هيل',
            'يانسون': 'يانسون', 'شمر': 'شمر', 'محلب': 'محلب',
            'لوز': 'لوز', 'فستق': 'فستق', 'كاجو': 'كاجو',
            'بندق': 'بندق', 'بيكان': 'بيكان', 'فول سوداني': 'فول سوداني',
            'سمسم': 'سمسم',
        }
        
        # English to Arabic sample types
        self.english_sample_types = {
            'cucumber': 'خيار', 'tomato': 'طماطم', 'eggplant': 'باذنجان',
            'aubergine': 'باذنجان', 'pepper': 'فلفل', 'zucchini': 'كوسة',
            'squash': 'كوسة', 'carrot': 'جزر', 'lettuce': 'خس',
            'parsley': 'بقدونس', 'spinach': 'سبانخ', 'potato': 'بطاطس',
            'beans': 'فاصوليا', 'okra': 'بامية', 'broccoli': 'بروكلي',
            'onion': 'بصل', 'arugula': 'جرجير', 'rocket': 'جرجير',
            'cauliflower': 'زهرة', 'dill': 'شبت', 'coriander': 'كزبرة',
            'cilantro': 'كزبرة', 'molokhia': 'ملوخية', 'mint': 'نعناع',
            'strawberry': 'فراولة', 'grape': 'عنب', 'apple': 'تفاح',
            'orange': 'برتقال', 'pomegranate': 'رمان', 'pear': 'كمثرى',
            'lemon': 'ليمون', 'lime': 'ليمون', 'berry': 'توت',
            'date': 'تمر', 'dates': 'تمر', 'wheat': 'قمح',
            'rice': 'رز', 'corn': 'ذرة', 'lentil': 'عدس',
            'flour': 'دقيق', 'oat': 'شوفان', 'oats': 'شوفان',
            'cumin': 'كمون', 'thyme': 'زعتر', 'spices': 'بهارات',
            'clove': 'قرنفل', 'cardamom': 'هيل', 'anise': 'يانسون',
            'fennel': 'شمر', 'almond': 'لوز', 'pistachio': 'فستق',
            'cashew': 'كاجو', 'hazelnut': 'بندق', 'pecan': 'بيكان',
            'peanut': 'فول سوداني', 'peanuts': 'فول سوداني', 'sesame': 'سمسم',
        }
        
        # Neighborhoods
        self.neighborhood_patterns = {
            'الاسكان': 'الإسكان', 'الإسكان': 'الإسكان', 'اسكان': 'الإسكان',
            'الصفراء': 'الصفراء', 'صفراء': 'الصفراء',
            'الموطأ': 'الموطأ', 'موطأ': 'الموطأ',
            'الريان': 'الريان', 'النخيل': 'النخيل',
            'الخليج': 'الخليج', 'الفيصلية': 'الفيصلية',
            'النهضة': 'النهضة', 'السلام': 'السلام',
            'الشفاء': 'الشفاء', 'المنتزه': 'المنتزه',
            'العليا': 'العليا', 'الضاحي': 'الضاحي',
            'الورود': 'الورود', 'المروج': 'المروج',
            'البساتين': 'البساتين', 'الهلال': 'الهلال',
            'الأخضر': 'الأخضر', 'الاخضر': 'الأخضر',
            'الروضة': 'الروضة', 'الفلاح': 'الفلاح',
            'الجردة': 'الجردة', 'الأفق': 'الأفق', 'الافق': 'الأفق',
            'السالمية': 'السالمية', 'الفايزية': 'الفايزية',
            'الشقة': 'الشقة', 'المنار': 'المنار',
        }
        
        # Pesticide mappings
        self.arabic_pesticide_map = {
            'بابروفيزن': 'bupro', 'بايفنثرين': 'bifenthrin',
            'كلوربيريفوس': 'chlorpyrifos', 'إيميداكلوبريد': 'imidacloprid',
            'اميداكلوبريد': 'imidacloprid', 'اسيتاميبريد': 'acetamiprid',
            'ازوكسيستروبين': 'azoxystrobin', 'دلتاميثرين': 'deltamethrin',
            'سايبرمثرين': 'cypermethrin', 'ثياميثوكسام': 'thiamethoxam',
            'بروفينوفوس': 'profenofos', 'ميتالاكسيل': 'metalaxyl',
            'كاربندازيم': 'carbendazim', 'ابامكتين': 'abamectin',
            'الفيبرونيل': 'fipronil', 'فيبرونيل': 'fipronil',
            'الايثيون': 'ethion', 'الايثون': 'ethion',
            'لامدا': 'lambda-cyhalothrin', 'سايهالوثرين': 'lambda-cyhalothrin',
            'ديميثويت': 'dimethoate', 'ميثوميل': 'methomyl',
            'بوبروفيزين': 'buprofezin', 'بروكونازول': 'propiconazole',
            'تيبوكونازول': 'tebuconazole',
        }
    
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get database connection"""
        return duckdb.connect(self.db_path, read_only=True)
    
    def _normalize_query(self, query: str) -> str:
        """تطبيع الاستعلام"""
        # Arabic numerals
        num_map = {
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
            'واحد': '1', 'اثنين': '2', 'ثلاثة': '3', 'اربعة': '4', 'خمسة': '5',
            'ستة': '6', 'سبعة': '7', 'ثمانية': '8', 'تسعة': '9', 'عشرة': '10'
        }
        for ar, en in num_map.items():
            query = query.replace(ar, en)
        return query
    
    def _detect_sample_types(self, query: str) -> List[str]:
        """كشف أنواع العينات"""
        detected = []
        for ar_name, pattern in self.sample_types.items():
            if ar_name in query:
                detected.append(pattern)
        
        if not detected:
            query_lower = query.lower()
            for en_name, ar_pattern in self.english_sample_types.items():
                if en_name in query_lower:
                    detected.append(ar_pattern)
        
        return list(set(detected))
    
    def _detect_neighborhoods(self, query: str) -> List[str]:
        """كشف الأحياء"""
        detected = []
        for ar_name, pattern in self.neighborhood_patterns.items():
            if ar_name in query:
                detected.append(pattern)
        return list(set(detected))
    
    def _detect_pesticide(self, query: str) -> Optional[str]:
        """كشف المبيد"""
        for ar_name, en_pattern in self.arabic_pesticide_map.items():
            if ar_name in query:
                return en_pattern
        return None
    
    def process(self, query: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        معالجة الاستعلام وإرجاع (نص الاستجابة, DataFrame)
        """
        query_normalized = self._normalize_query(query)
        query_lower = query_normalized.lower()
        
        detected_samples = self._detect_sample_types(query)
        detected_neighborhoods = self._detect_neighborhoods(query)
        detected_pesticide = self._detect_pesticide(query)
        
        # ============================================================
        # Pattern 1: Samples with N pesticides
        # "عدد العينات التي تحتوي على 6 مبيدات"
        # ============================================================
        pesticide_count_pattern = r'(?:عدد|عينات|samples?|تحتوي|contain).*?(\d+)\s*(?:مبيد|مبيدات|pesticide)'
        match = re.search(pesticide_count_pattern, query_normalized, re.IGNORECASE)
        
        if match:
            n_pesticides = int(match.group(1))
            return self._handle_n_pesticides(n_pesticides, detected_samples)
        
        # ============================================================
        # Pattern 2: Count samples above/below limit
        # "عدد عينات الطماطم فوق الحد"
        # ============================================================
        count_phrases = ['count', 'how many', 'عدد', 'كم', 'ماهي', 'ما هي', 'what is']
        above_limit_phrases = ['more than limit', 'exceeding', 'above limit', 'over limit',
                               'reading more', 'exceed', 'فوق', 'تجاوز', 'مخالف', 'راسب']
        below_limit_phrases = ['below limit', 'under limit', 'within limit', 'تحت', 'ناجح', 'سليم', 'مطابق']
        
        is_count_query = any(phrase in query_lower for phrase in count_phrases)
        is_above_limit = any(phrase in query_lower for phrase in above_limit_phrases)
        is_below_limit = any(phrase in query_lower for phrase in below_limit_phrases)
        
        if is_count_query and detected_samples and (is_above_limit or is_below_limit):
            return self._handle_count_samples_limit(detected_samples, detected_neighborhoods, is_above_limit)
        
        # ============================================================
        # Pattern 3: List pesticides in sample type
        # "ماهي المبيدات في الطماطم"
        # ============================================================
        pesticide_list_patterns = ['المبيدات الموجوده', 'المبيدات في عينة', 'تكرار المبيدات', 
                                   'المبيدات المكتشفة', 'ماهي المبيدات', 'المبيدات في']
        if any(p in query for p in pesticide_list_patterns) and detected_samples:
            return self._handle_list_pesticides(detected_samples)
        
        # ============================================================
        # Pattern 4: Pesticides in neighborhoods
        # "ماهي المبيدات في حي الإسكان"
        # ============================================================
        if detected_neighborhoods and ('المبيدات' in query or 'مبيدات' in query):
            show_separately = any(phrase in query for phrase in ['كل علي حده', 'كل على حده', 'بشكل منفصل'])
            return self._handle_neighborhood_pesticides(detected_neighborhoods, show_separately)
        
        # ============================================================
        # Pattern 5: Find samples containing pesticide
        # "عينات الطماطم التي تحتوي على البايفنثرين"
        # ============================================================
        if detected_pesticide and detected_samples:
            return self._handle_find_pesticide_in_sample(detected_pesticide, detected_samples)
        
        # ============================================================
        # Pattern 6: Neighborhood ranking
        # "ترتيب الأحياء حسب المخالفات"
        # ============================================================
        if any(kw in query for kw in ['ترتيب', 'ranking', 'الاكثر', 'الأكثر']) and \
           any(kw in query for kw in ['حي', 'أحياء', 'احياء']):
            return self._handle_neighborhood_ranking()
        
        # ============================================================
        # Pattern 7: Comprehensive analysis
        # "تحليل شامل للطماطم"
        # ============================================================
        comprehensive_keywords = ['تحليل شامل', 'تقرير شامل', 'احصائيات', 'تفصيل', 'ملخص شامل']
        if any(kw in query for kw in comprehensive_keywords) and detected_samples:
            return self._handle_comprehensive_analysis(detected_samples)
        
        # Default: Unknown query
        return self._handle_unknown_query(query), None
    
    # ============================================================
    # Handlers
    # ============================================================
    
    def _handle_n_pesticides(self, n: int, samples: List[str]) -> Tuple[str, pd.DataFrame]:
        """عينات بعدد N مبيدات"""
        con = self._get_connection()
        
        sample_filter = ""
        if samples:
            conditions = [f"\"اسم العينة\" LIKE '%{s}%'" for s in samples]
            sample_filter = f"AND ({' OR '.join(conditions)})"
        
        sql = f"""
        SELECT 
            "كود العينة" as sample_code,
            "اسم العينة" as sample_name,
            COUNT(*) as pesticide_count
        FROM chemistry_tidy
        WHERE is_detected = 1 AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
        {sample_filter}
        GROUP BY "كود العينة", "اسم العينة"
        HAVING COUNT(*) = {n}
        ORDER BY "كود العينة" DESC
        LIMIT 100
        """
        
        df = con.execute(sql).df()
        con.close()
        
        sample_display = ' + '.join(samples) if samples else "جميع الأنواع"
        
        if not df.empty:
            # Build breakdown
            normalized_names = df['sample_name'].astype(str).str.strip()
            normalized_names = normalized_names.str.replace('[أإآ]', 'ا', regex=True).str.replace('ة', 'ه')
            counts = normalized_names.value_counts()
            breakdown = " + ".join([f"{count} {name}" for name, count in counts.items()])
            
            response = f"🔢 **عينات {sample_display} بـ {n} مبيدات:**\n\n"
            response += f"✅ وجدت **{len(df)}** عينة\n\n"
            response += f"📋 **ملخص الأنواع:** {breakdown}\n\n"
            response += df.head(20).to_markdown(index=False)
            
            if len(df) > 20:
                response += f"\n\n💾 *الملف المرفق يحتوي على جميع الـ {len(df)} نتيجة*"
        else:
            response = f"⚠️ لم أجد عينات {sample_display} بـ {n} مبيدات"
        
        return response, df
    
    def _handle_count_samples_limit(self, samples: List[str], neighborhoods: List[str], 
                                     is_above: bool) -> Tuple[str, pd.DataFrame]:
        """عد العينات فوق/تحت الحد"""
        con = self._get_connection()
        
        # Sample filter
        sample_conditions = []
        for t in samples:
            sample_conditions.append(f"\"اسم العينة\" LIKE '%{t}%'")
            sample_conditions.append(f"\"نوع العينة\" LIKE '%{t}%'")
        sample_filter = f"AND ({' OR '.join(sample_conditions)})"
        
        # Neighborhood filter
        neighborhood_filter = ""
        if neighborhoods:
            hood_conditions = []
            for n in neighborhoods:
                variants = [n]
                if 'ا' in n:
                    variants.append(n.replace('ا', 'إ'))
                    variants.append(n.replace('ا', 'أ'))
                if 'إ' in n:
                    variants.append(n.replace('إ', 'ا'))
                if 'أ' in n:
                    variants.append(n.replace('أ', 'ا'))
                for v in set(variants):
                    hood_conditions.append(f"\"الحى\" LIKE '%{v}%'")
            neighborhood_filter = f"AND ({' OR '.join(hood_conditions)})"
        
        sql = f"""
        SELECT 
            "اسم العينة" as نوع_العينة,
            COUNT(DISTINCT "كود العينة") as عدد_العينات,
            SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as فوق_الحد,
            SUM(CASE WHEN is_above_limit = 0 AND is_detected = 1 THEN 1 ELSE 0 END) as تحت_الحد
        FROM chemistry_tidy
        WHERE is_detected = 1
        {sample_filter}
        {neighborhood_filter}
        GROUP BY "اسم العينة"
        ORDER BY عدد_العينات DESC
        """
        
        df = con.execute(sql).df()
        con.close()
        
        type_display = " + ".join(samples)
        if neighborhoods:
            type_display += f" في {' + '.join(neighborhoods)}"
        
        if not df.empty:
            total_samples = int(df['عدد_العينات'].sum())
            total_above = int(df['فوق_الحد'].sum())
            total_below = int(df['تحت_الحد'].sum())
            
            response = f"📊 **نتائج عينات {type_display}:**\n\n"
            response += f"✅ إجمالي: **{total_samples}** عينة\n"
            response += f"🔴 فوق الحد: **{total_above}**\n"
            response += f"🟢 تحت الحد: **{total_below}**\n\n"
            response += df.to_markdown(index=False)
        else:
            response = f"⚠️ لم يتم العثور على عينات {type_display}"
        
        return response, df
    
    def _handle_list_pesticides(self, samples: List[str]) -> Tuple[str, pd.DataFrame]:
        """قائمة المبيدات في عينة"""
        con = self._get_connection()
        
        conditions = [f"\"اسم العينة\" LIKE '%{s}%'" for s in samples]
        sample_filter = f"({' OR '.join(conditions)})"
        
        sql = f"""
        SELECT 
            pesticide_name as المبيد,
            COUNT(*) as عدد_التكرار,
            ROUND(AVG(concentration), 4) as متوسط_التركيز,
            ROUND(MAX(concentration), 4) as أعلى_تركيز,
            SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as عدد_المخالفات
        FROM chemistry_tidy
        WHERE is_detected = 1
        AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
        AND {sample_filter}
        GROUP BY pesticide_name
        ORDER BY عدد_التكرار DESC
        LIMIT 50
        """
        
        df = con.execute(sql).df()
        con.close()
        
        sample_display = ' + '.join(samples)
        
        if not df.empty:
            total_detections = int(df['عدد_التكرار'].sum())
            total_violations = int(df['عدد_المخالفات'].sum())
            unique_pesticides = len(df)
            
            response = f"🧪 **المبيدات في {sample_display}:**\n\n"
            response += f"📊 **الملخص:**\n"
            response += f"• عدد المبيدات المختلفة: **{unique_pesticides}**\n"
            response += f"• إجمالي الاكتشافات: **{total_detections}**\n"
            response += f"• إجمالي المخالفات: **{total_violations}**\n\n"
            response += df.to_markdown(index=False)
        else:
            response = f"⚠️ لم يتم العثور على مبيدات في {sample_display}"
        
        return response, df
    
    def _handle_neighborhood_pesticides(self, neighborhoods: List[str], 
                                         show_separately: bool) -> Tuple[str, pd.DataFrame]:
        """المبيدات في الأحياء"""
        con = self._get_connection()
        
        all_dfs = []
        all_responses = []
        
        if show_separately:
            for n in neighborhoods:
                sql = f"""
                SELECT 
                    pesticide_name as المبيد,
                    COUNT(*) as عدد_التكرار,
                    ROUND(AVG(concentration), 4) as متوسط_التركيز,
                    SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as فوق_الحد
                FROM chemistry_tidy
                WHERE is_detected = 1
                AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
                AND "الحى" LIKE '%{n}%'
                GROUP BY pesticide_name
                ORDER BY عدد_التكرار DESC
                LIMIT 20
                """
                df = con.execute(sql).df()
                all_dfs.append(df)
                
                if not df.empty:
                    total = int(df['عدد_التكرار'].sum())
                    violations = int(df['فوق_الحد'].sum())
                    all_responses.append(f"📍 **حي {n}:** {len(df)} مبيد، {total} اكتشاف، {violations} تجاوز\n\n{df.to_markdown(index=False)}")
                else:
                    all_responses.append(f"📍 **حي {n}:** لم يتم العثور على مبيدات")
            
            response = "\n\n---\n\n".join(all_responses)
            combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        else:
            # Combined view
            conditions = [f"\"الحى\" LIKE '%{n}%'" for n in neighborhoods]
            neighborhood_filter = f"({' OR '.join(conditions)})"
            
            sql = f"""
            SELECT 
                "الحى" as الحي,
                pesticide_name as المبيد,
                COUNT(*) as عدد_الكشف,
                SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) as عدد_المخالفات
            FROM chemistry_tidy
            WHERE is_detected = 1 
            AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
            AND {neighborhood_filter}
            GROUP BY "الحى", pesticide_name
            ORDER BY "الحى", عدد_الكشف DESC
            LIMIT 50
            """
            
            combined_df = con.execute(sql).df()
            
            if not combined_df.empty:
                response = f"🏘️ **المبيدات في {' + '.join(neighborhoods)}:**\n\n"
                response += f"✅ وجدت {len(combined_df)} سجل\n\n"
                response += combined_df.to_markdown(index=False)
                response += "\n\n💡 أضف 'كل علي حده' للسؤال لعرض كل حي بشكل منفصل"
            else:
                response = "⚠️ لم يتم العثور على مبيدات في الأحياء المحددة"
        
        con.close()
        return response, combined_df
    
    def _handle_find_pesticide_in_sample(self, pesticide: str, 
                                          samples: List[str]) -> Tuple[str, pd.DataFrame]:
        """البحث عن مبيد في عينة"""
        con = self._get_connection()
        
        conditions = [f"\"اسم العينة\" LIKE '%{s}%'" for s in samples]
        sample_filter = f"({' OR '.join(conditions)})"
        
        sql = f"""
        SELECT 
            "كود العينة" as كود_العينة,
            "اسم العينة" as اسم_العينة,
            pesticide_name as المبيد,
            concentration as التركيز,
            limit_value as الحد,
            ROUND(exceedance_ratio, 2) as النسبة,
            CASE WHEN is_compliant = 1 THEN 'مطابق' ELSE 'غير مطابق' END as الحالة
        FROM chemistry_tidy
        WHERE is_detected = 1 
        AND pesticide_name ILIKE '%{pesticide}%'
        AND {sample_filter}
        ORDER BY concentration DESC
        LIMIT 50
        """
        
        df = con.execute(sql).df()
        con.close()
        
        samples_display = ' + '.join(samples)
        
        if not df.empty:
            compliant = len(df[df['الحالة'] == 'مطابق'])
            non_compliant = len(df[df['الحالة'] == 'غير مطابق'])
            
            response = f"🔍 **{pesticide} في {samples_display}:**\n\n"
            response += f"✅ وجدت **{len(df)}** سجل\n"
            response += f"📊 {compliant} مطابق | {non_compliant} غير مطابق\n\n"
            response += df.to_markdown(index=False)
        else:
            response = f"⚠️ لم يتم العثور على {pesticide} في {samples_display}"
        
        return response, df
    
    def _handle_neighborhood_ranking(self) -> Tuple[str, pd.DataFrame]:
        """ترتيب الأحياء حسب المخالفات"""
        con = self._get_connection()
        
        sql = """
        SELECT 
            "الحى" as الحي,
            COUNT(DISTINCT "كود العينة") as عدد_العينات,
            SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) as عدد_المخالفات,
            ROUND(SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) * 100.0 / 
                  NULLIF(COUNT(DISTINCT "كود العينة"), 0), 2) as نسبة_المخالفات
        FROM chemistry_tidy
        WHERE "الحى" IS NOT NULL AND "الحى" != ''
        GROUP BY "الحى"
        ORDER BY عدد_المخالفات DESC
        LIMIT 20
        """
        
        df = con.execute(sql).df()
        con.close()
        
        response = "🏘️ **ترتيب الأحياء حسب المخالفات (تنازلي):**\n\n"
        
        if not df.empty:
            response += df.to_markdown(index=False)
        else:
            response += "⚠️ لم أجد بيانات"
        
        return response, df
    
    def _handle_comprehensive_analysis(self, samples: List[str]) -> Tuple[str, pd.DataFrame]:
        """تحليل شامل لنوع عينة"""
        con = self._get_connection()
        
        conditions = [f"\"اسم العينة\" LIKE '%{s}%'" for s in samples]
        sample_filter = f"({' OR '.join(conditions)})"
        type_display = " + ".join(samples)
        
        # Distribution query
        distribution_sql = f"""
        WITH sample_pesticide_counts AS (
            SELECT 
                "كود العينة" as sample_code,
                COUNT(CASE WHEN is_detected = 1 AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA') THEN 1 END) as pesticide_count,
                SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as failing_count
            FROM chemistry_tidy
            WHERE {sample_filter}
            GROUP BY "كود العينة"
        )
        SELECT 
            pesticide_count as عدد_المبيدات,
            COUNT(*) as عدد_العينات,
            SUM(CASE WHEN failing_count = 0 THEN 1 ELSE 0 END) as لم_تتجاوز,
            SUM(CASE WHEN failing_count > 0 THEN 1 ELSE 0 END) as تجاوزت
        FROM sample_pesticide_counts
        GROUP BY pesticide_count
        ORDER BY pesticide_count ASC
        """
        
        distribution_df = con.execute(distribution_sql).df()
        
        # Pesticide stats query
        pesticide_stats_sql = f"""
        SELECT 
            pesticide_name as المبيد,
            COUNT(*) as التكرار,
            ROUND(MAX(limit_value), 4) as MRL,
            SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as تجاوزات,
            ROUND(MEDIAN(concentration), 4) as median,
            ROUND(MIN(concentration), 4) as min,
            ROUND(MAX(concentration), 4) as max
        FROM chemistry_tidy
        WHERE {sample_filter}
        AND is_detected = 1
        AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
        GROUP BY pesticide_name
        ORDER BY التكرار DESC
        """
        
        pesticide_stats_df = con.execute(pesticide_stats_sql).df()
        con.close()
        
        response = f"📊 **تحليل شامل لعينات {type_display}**\n\n"
        
        if not distribution_df.empty:
            total = int(distribution_df['عدد_العينات'].sum())
            total_passed = int(distribution_df['لم_تتجاوز'].sum())
            total_failed = int(distribution_df['تجاوزت'].sum())
            
            response += f"**📋 الإحصائيات العامة:**\n"
            response += f"• إجمالي العينات: **{total}**\n"
            response += f"• لم تتجاوز: **{total_passed}**\n"
            response += f"• تجاوزت: **{total_failed}**\n\n"
            
            response += "**📊 توزيع العينات حسب عدد المبيدات:**\n"
            response += distribution_df.to_markdown(index=False)
            response += "\n\n"
        
        if not pesticide_stats_df.empty:
            failing_pesticides = pesticide_stats_df[pesticide_stats_df['تجاوزات'] > 0]
            
            response += "**🧪 إحصائيات المبيدات المكتشفة:**\n"
            response += pesticide_stats_df.head(15).to_markdown(index=False)
            
            if not failing_pesticides.empty:
                response += f"\n\n🔴 **المبيدات المتسببة في الرسوب ({len(failing_pesticides)} نوع):** "
                response += ", ".join(failing_pesticides['المبيد'].tolist())
        
        # Combine for return
        return response, distribution_df
    
    def _handle_unknown_query(self, query: str) -> str:
        """معالجة الاستعلامات غير المعروفة"""
        response = "⚠️ **عذراً، لم أتمكن من فهم سؤالك بشكل كامل.**\n\n"
        response += "📝 **أمثلة على الأسئلة المدعومة:**\n"
        response += "• ما هو عدد العينات التي تحتوي على 6 مبيدات؟\n"
        response += "• ما عدد عينات الطماطم فوق الحد؟\n"
        response += "• ماهي المبيدات في الخيار؟\n"
        response += "• المبيدات في حي الإسكان والنهضة\n"
        response += "• ترتيب الأحياء حسب المخالفات\n"
        response += "• تحليل شامل للطماطم\n"
        return response


# Create singleton instance
_engine = None

def get_query_engine(db_path: str = None) -> CoreQueryEngine:
    """Get query engine singleton"""
    global _engine
    if _engine is None:
        _engine = CoreQueryEngine(db_path)
    return _engine
