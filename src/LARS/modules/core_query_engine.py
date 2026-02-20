"""
core_query_engine.py
====================
محرك استعلامات موحد خالٍ من Streamlit
يمكن استخدامه من:
- التطبيق الرئيسي (ai_assistant.py)
- البوت الصوتي (voice_bot_enhanced.py)

Features:
- Pattern matching for common queries
- LLM fallback for unknown queries
- Egyptian & Saudi Arabic dialect support
"""
import re
import duckdb
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import os
import logging

# ── Centralized mappings (replaces local dictionaries) ──
from modules.mappings import (
    SAMPLE_CORRECTIONS,
    SAMPLE_EN_TO_AR,
    NEIGHBORHOOD_CORRECTIONS,
    PESTICIDE_AR_TO_EN,
    normalize_arabic_query,
    get_pesticide_variants,
)

# Semantic pattern recognizer (optional - graceful fallback if not available)
try:
    from semantic_pattern_recognizer import get_semantic_recognizer
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False
    get_semantic_recognizer = lambda: None

# Intent-based query router (optional - graceful fallback if not available)
try:
    from modules.intent_router import IntentRouter, Intent, QueryEntities
    HAS_INTENT_ROUTER = True
except ImportError:
    HAS_INTENT_ROUTER = False
    IntentRouter = None
    Intent = None
    QueryEntities = None

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "lars_data.duckdb"

# LLM Configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b-it-qat")



class CoreQueryEngine:
    """
    محرك استعلامات مركزي موحد - بدون Streamlit
    Unified Query Engine - Streamlit-free
    
    Supports:
    - All query patterns from ai_assistant.py
    - LLM fallback for unknown queries
    - Dialect normalization (Egyptian + Saudi Arabic)
    """
    
    def __init__(self, db_path: str = None, llm_client=None, enable_llm_fallback: bool = True):
        """
        Initialize the query engine.
        
        Args:
            db_path: Path to DuckDB database
            llm_client: Optional pre-configured LLM client
            enable_llm_fallback: Whether to use LLM for unknown queries
        """
        self.db_path = db_path or str(DB_PATH)
        self.llm_client = llm_client
        self.enable_llm_fallback = enable_llm_fallback
        
        # Initialize LLM if enabled and not provided
        if enable_llm_fallback and llm_client is None:
            self._init_llm()
        
        self._load_mappings()
        self._load_schema_info()
        
        # Initialize semantic pattern recognizer (optional)
        self.semantic_recognizer = get_semantic_recognizer() if HAS_SEMANTIC else None
        if self.semantic_recognizer:
            print("✅ Semantic Pattern Recognizer loaded!")
        else:
            print("ℹ️ Semantic Pattern Recognizer not available, using keyword matching only")
        
        # Initialize intent-based query router (optional)
        if HAS_INTENT_ROUTER:
            self.router = IntentRouter(dialect_synonyms=self.dialect_synonyms)
            print("✅ Intent Router loaded!")
        else:
            self.router = None
            print("ℹ️ Intent Router not available, using pattern matching only")
    
    def _init_llm(self):
        """Initialize LLM client for fallback queries"""
        try:
            from ollama import Client
            self.llm_client = Client(host=OLLAMA_URL)
            # Test connection
            self.llm_client.list()
        except Exception as e:
            print(f"⚠️ LLM initialization failed: {e}")
            self.llm_client = None
    
    def _load_schema_info(self):
        """Load database schema for LLM context"""
        try:
            con = self._get_connection()
            # Get column names from chemistry_tidy
            cols_df = con.execute("DESCRIBE chemistry_tidy").df()
            self.schema_columns = cols_df['column_name'].tolist()
            
            # Get sample types
            sample_types_df = con.execute('''
                SELECT DISTINCT "اسم العينة" FROM chemistry_tidy 
                WHERE "اسم العينة" IS NOT NULL LIMIT 50
            ''').df()
            self.available_sample_types = sample_types_df['اسم العينة'].tolist()
            
            # Get neighborhoods
            neighborhoods_df = con.execute('''
                SELECT DISTINCT "الحى" FROM chemistry_tidy 
                WHERE "الحى" IS NOT NULL LIMIT 50
            ''').df()
            self.available_neighborhoods = neighborhoods_df['الحى'].tolist()
            
            con.close()
        except Exception as e:
            print(f"⚠️ Schema loading failed: {e}")
            self.schema_columns = []
            self.available_sample_types = []
            self.available_neighborhoods = []
    
    def _load_mappings(self):
        """Load query mappings.

        Dialect synonyms and action/limit keywords are defined here
        because they are specific to the query engine. All entity
        dictionaries (samples, pesticides, neighborhoods) come from
        the centralized ``modules.mappings`` module.
        """
        # ============================================================
        # Egyptian & Saudi Arabic Dialect Synonyms (KEEP — engine-specific)
        # ============================================================
        self.dialect_synonyms = {
            # Egyptian
            'اديني': 'أعطني', 'ادني': 'أعطني', 'هاتلي': 'أعطني', 'الديني': 'أعطني',
            'عايز': 'أريد', 'عاوز': 'أريد', 'عاوزه': 'أريد',
            'ازاي': 'كيف', 'ازي': 'كيف', 'يعني ايه': 'ما معنى',
            'فين': 'أين', 'ايه': 'ما', 'إيه': 'ما',
            'كام': 'كم', 'قد ايه': 'كم', 'كده': 'هكذا',
            'دور على': 'ابحث عن', 'دوري على': 'ابحث عن',
            'ورني': 'أرني', 'وريني': 'أرني',
            'طلعلي': 'أخرج لي', 'جيبلي': 'أحضر لي',
            'مش فاهم': 'لا أفهم', 'مفهمتش': 'لم أفهم',
            # Saudi
            'ابغى': 'أريد', 'ابغي': 'أريد', 'ابي': 'أريد', 'أبي': 'أريد',
            'ابيها': 'أريدها', 'ابغاها': 'أريدها', 'ابيه': 'أريده',
            'وش': 'ما', 'ويش': 'ما', 'ايش': 'ما', 'إيش': 'ما',
            'وين': 'أين', 'فيين': 'أين',
            'يعني وش': 'ما معنى', 'شلون': 'كيف',
            'طيب': 'حسناً', 'زين': 'جيد', 'تمام': 'حسناً',
            'اعطني': 'أعطني', 'عطني': 'أعطني', 'اعطيني': 'أعطني',
            'ورني': 'أرني', 'وريني': 'أرني',
            'دور لي': 'ابحث عن', 'بحث': 'ابحث', 'لقني': 'ابحث عن',
            # Common verbs/actions
            'اوجد': 'ابحث عن', 'أوجد': 'ابحث عن', 'اعثر': 'ابحث عن',
            'احصل': 'ابحث عن', 'طلع': 'اعرض', 'اظهر': 'اعرض', 'أظهر': 'اعرض',
            'بين': 'اعرض', 'بينلي': 'اعرض لي',
            # Query keywords
            'كل على حده': 'كل علي حده', 'كل واحد لوحده': 'كل علي حده',
            'بشكل منفصل': 'كل علي حده', 'each separately': 'كل علي حده',
            'separately': 'كل علي حده',
        }

        # ============================================================
        # Action keywords (KEEP — engine-specific)
        # ============================================================
        self.action_keywords = {
            'search': ['ابحث', 'دور', 'اوجد', 'اعثر', 'find', 'search', 'locate'],
            'show': ['اعرض', 'اظهر', 'ورني', 'طلع', 'show', 'display', 'list'],
            'give': ['اديني', 'اعطني', 'عطني', 'هاتلي', 'جيبلي', 'give', 'provide'],
            'want': ['عايز', 'عاوز', 'ابغى', 'ابي', 'اريد', 'want', 'need'],
            'count': ['عدد', 'كم', 'كام', 'count', 'how many', 'number'],
            'what': ['ماهي', 'ما هي', 'ماهو', 'ما هو', 'ايه', 'ايش', 'وش', 'what', 'which'],
        }

        # ============================================================
        # Limit/Compliance keywords (KEEP — engine-specific)
        # ============================================================
        self.above_limit_keywords = [
            'فوق الحد', 'تجاوز', 'مخالف', 'راسب', 'راسبة', 'غير مطابق',
            'مخالفة', 'المخالفات', 'الراسبة', 'الراسبين', 'متجاوز', 'متجاوزة',
            'above limit', 'exceeding', 'over limit', 'violation',
            'non-compliant', 'الغير مطابقة',
        ]
        self.below_limit_keywords = [
            'تحت الحد', 'ضمن الحد', 'مطابق', 'مطابقة', 'ناجح', 'سليم',
            'below limit', 'within limit', 'compliant', 'passing', 'safe',
        ]

        # ============================================================
        # Statistics keywords (KEEP — engine-specific)
        # ============================================================
        self.stats_keywords = {
            'max': ['اعلى', 'أعلى', 'اكبر', 'أكبر', 'max', 'maximum', 'highest'],
            'min': ['اقل', 'أقل', 'اصغر', 'أصغر', 'min', 'minimum', 'lowest'],
            'avg': ['متوسط', 'المتوسط', 'معدل', 'average', 'mean', 'avg'],
            'median': ['الوسيط', 'وسيط', 'median'],
            'range': ['المدى', 'المدي', 'نطاق', 'range', 'spread'],
            'count': ['عدد', 'تكرار', 'count', 'frequency'],
            'unique': ['فريدة', 'فريد', 'مميزة', 'unique', 'distinct'],
        }

        # ============================================================
        # Entity dictionaries — FROM CENTRALIZED MAPPINGS
        # ============================================================
        self.sample_types = SAMPLE_CORRECTIONS
        self.english_sample_types = SAMPLE_EN_TO_AR
        self.neighborhood_patterns = NEIGHBORHOOD_CORRECTIONS
        self.arabic_pesticide_map = PESTICIDE_AR_TO_EN
    
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get database connection"""
        return duckdb.connect(self.db_path, read_only=True)
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query: apply dialect synonyms then convert Arabic numerals.

        Dialect synonyms are engine-specific (defined in _load_mappings).
        Numeral conversion uses the centralized normalize_arabic_query().
        """
        # Step 1: Apply dialect synonyms (engine-specific)
        for dialect, standard in self.dialect_synonyms.items():
            query = query.replace(dialect, standard)
        # Step 2: Convert Arabic numerals & number words (centralized)
        return normalize_arabic_query(query)
    
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
    
    def _handle_facility_search(self, query: str) -> Tuple[str, pd.DataFrame]:
        """البحث عن العينات في منشأة معينة"""
        con = self._get_connection()
        
        # استخراج اسم المنشأة من الاستعلام
        # إزالة الكلمات المفتاحية الشائعة
        facility_keywords = ['ابحث', 'عن', 'العينات', 'في', 'منشأة', 'منشاة', 'مصنع', 'مطعم', 
                            'محل', 'متجر', 'عينات', 'بحث', 'search', 'facility', 'samples']
        
        words = query.split()
        facility_name_parts = []
        
        for word in words:
            if word.lower() not in facility_keywords and len(word) > 2:
                facility_name_parts.append(word)
        
        facility_name = ' '.join(facility_name_parts)
        
        if not facility_name:
            return "⚠️ لم أتمكن من استخراج اسم المنشأة من السؤال. يرجى تحديد اسم المنشأة.", pd.DataFrame()
        
        # البحث في قاعدة البيانات
        sql = f"""
        SELECT 
            "كود العينة" as كود_العينة,
            "اسم المنشاة" as اسم_المنشأة,
            "اسم العينة" as نوع_العينة,
            "التاريخ" as التاريخ,
            "الحى" as الحي,
            pesticide_name as المبيد,
            concentration as التركيز,
            limit_value as الحد_المسموح,
            CASE WHEN is_above_limit = 1 THEN 'فوق الحد' ELSE 'ضمن الحد' END as الحالة
        FROM chemistry_tidy
        WHERE is_detected = 1
        AND "اسم المنشاة" LIKE '%{facility_name}%'
        ORDER BY "التاريخ" DESC
        LIMIT 50
        """
        
        df = con.execute(sql).df()
        con.close()
        
        if not df.empty:
            unique_samples = df['كود_العينة'].nunique()
            above_limit = len(df[df['الحالة'] == 'فوق الحد'])
            
            response = f"🏭 **نتائج البحث في منشأة: {facility_name}**\n\n"
            response += f"✅ عدد العينات: **{unique_samples}** عينة\n"
            response += f"🔬 عدد الاكتشافات: **{len(df)}** مبيد\n"
            response += f"🔴 فوق الحد: **{above_limit}**\n"
            response += f"🟢 ضمن الحد: **{len(df) - above_limit}**\n\n"
            response += df.to_markdown(index=False)
        else:
            response = f"⚠️ لم يتم العثور على عينات في منشأة تحتوي على '{facility_name}'"
        
        return response, df
    
    def _handle_comprehensive_neighborhood(self, query: str, detected_samples: List[str], 
                                          detected_neighborhoods: List[str]) -> Tuple[str, pd.DataFrame]:
        """تحليل شامل للعينات في حي معين مع عرض الأنواع والعدد وفوق/تحت الحد"""
        con = self._get_connection()
        
        # Build neighborhood filter (mandatory)
        hood_conditions = []
        for n in detected_neighborhoods:
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
        
        # Build category filter based on keywords
        category_filter = ""
        category_name = "العينات"
        
        if 'توابل' in query or 'التوابل' in query or 'بهارات' in query:
            category_filter = """AND ("تصنيف العينة" LIKE '%توابل%' OR "اسم العينة" IN ('كمون', 'هيل', 'زعتر', 'كركم', 'بهارات', 'فلفل اسود', 'قرفة', 'يانسون', 'كزبرة', 'شمر', 'قرنفل'))"""
            category_name = "التوابل"
        elif 'خضار' in query or 'الخضار' in query or 'خضروات' in query:
            category_filter = """AND ("تصنيف العينة" LIKE '%خضار%' OR "اسم العينة" IN ('طماطم', 'خيار', 'كوسة', 'باذنجان', 'فلفل', 'بامية', 'فاصوليا', 'زهرة', 'بروكلي', 'ملفوف', 'خس', 'جرجير', 'بقدونس', 'كزبرة', 'شبت'))"""
            category_name = "الخضار"
        elif 'فاكهة' in query or 'فواكه' in query:
            category_filter = """AND ("تصنيف العينة" LIKE '%فاكهة%' OR "اسم العينة" IN ('تفاح', 'برتقال', 'عنب', 'مانجو', 'توت', 'رمان', 'ليمون', 'يوسف افندي', 'كمثرى', 'فراولة', 'خوخ', 'مشمش'))"""
            category_name = "الفواكه"
        elif 'ورقيات' in query or 'الورقيات' in query:
            category_filter = """AND ("تصنيف العينة" LIKE '%ورقيات%' OR "اسم العينة" IN ('خس', 'جرجير', 'بقدونس', 'كزبرة', 'شبت', 'نعناع', 'سبانخ', 'ملوخية', 'رجلة', 'سلق'))"""
            category_name = "الورقيات"
        elif detected_samples:
            # Specific samples requested
            conditions = [f"\"اسم العينة\" LIKE '%{s}%'" for s in detected_samples]
            category_filter = f"AND ({' OR '.join(conditions)})"
            category_name = " + ".join(detected_samples)
        
        hood_display = " + ".join(detected_neighborhoods)
        logging.info(f"📊 تحليل شامل لـ {category_name} في حي {hood_display}")
        
        # Comprehensive SQL
        sql = f"""
        WITH sample_status AS (
            SELECT 
                "كود العينة" as sample_code,
                "اسم العينة" as sample_name,
                MAX(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as has_violation,
                MAX(CASE WHEN is_detected = 1 THEN 1 ELSE 0 END) as has_detection,
                COUNT(CASE WHEN is_detected = 1 THEN 1 END) as pesticide_count
            FROM chemistry_tidy
            WHERE 1=1
            {neighborhood_filter}
            {category_filter}
            GROUP BY "كود العينة", "اسم العينة"
        )
        SELECT 
            sample_name as نوع_العينة,
            COUNT(*) as العدد_الكلي,
            SUM(has_violation) as فوق_الحد,
            SUM(CASE WHEN has_violation = 0 THEN 1 ELSE 0 END) as تحت_الحد,
            SUM(CASE WHEN has_detection = 0 THEN 1 ELSE 0 END) as خالية_من_المبيدات
        FROM sample_status
        GROUP BY sample_name
        ORDER BY العدد_الكلي DESC
        """
        
        df = con.execute(sql).df()
        con.close()
        
        if not df.empty:
            total = int(df['العدد_الكلي'].sum())
            total_above = int(df['فوق_الحد'].sum())
            total_below = int(df['تحت_الحد'].sum())
            total_clean = int(df['خالية_من_المبيدات'].sum())
            
            response = f"📊 **تحليل شامل لـ {category_name} في حي {hood_display}**\n\n"
            response += f"✅ **إجمالي العينات:** {total} عينة\n"
            response += f"🔴 **فوق الحد:** {total_above} عينة (تحتوي على مبيدات مخالفة)\n"
            response += f"🟢 **تحت الحد:** {total_below} عينة (سليمة)\n"
            response += f"   • منها **{total_clean}** عينة خالية تماماً من المبيدات\n"
            response += f"   • ومنها **{total_below - total_clean}** عينة تحتوي على مبيدات ضمن الحدود المسموحة\n\n"
            response += df.to_markdown(index=False)
            
            return response, df
        else:
            response = f"⚠️ لم أجد عينات {category_name} في {hood_display}"
            return response, pd.DataFrame()
    
    def _route_by_semantic_pattern(self, pattern_type: str, query: str, query_normalized: str,
                                   query_lower: str, detected_samples: List[str],
                                   detected_neighborhoods: List[str], 
                                   detected_pesticide: Optional[str]) -> Optional[Tuple[str, pd.DataFrame]]:
        """
        Route query to appropriate handler based on semantic pattern type.
        Returns None if pattern cannot be handled (falls back to keyword matching).
        """
        try:
            # Check for 'كل على حده' (show separately)
            show_separately = any(phrase in query for phrase in ['كل علي حده', 'كل على حده', 'بشكل منفصل'])
            
            # Determine above/below limit from query
            is_above = None  # None = both
            above_keywords = ['فوق', 'above', 'exceed', 'تجاوز', 'غير مطابق', 'راسب', 'مخالف', 'فاشل']
            below_keywords = ['تحت', 'below', 'within', 'مطابق', 'ناجح', 'سليم']
            
            if any(kw in query for kw in above_keywords) and not any(kw in query_lower for kw in ['مطابق', 'تحت']):
                is_above = True
            elif any(kw in query for kw in below_keywords) and not any(kw in query for kw in above_keywords):
                is_above = False
            
            # Route based on pattern type
            if pattern_type == 'count_above_limit':
                # _handle_count_samples_limit(samples: List, neighborhoods: List, is_above: bool)
                samples = detected_samples if detected_samples else ['']
                return self._handle_count_samples_limit(samples, detected_neighborhoods, True)
            
            elif pattern_type == 'count_below_limit':
                samples = detected_samples if detected_samples else ['']
                return self._handle_count_samples_limit(samples, detected_neighborhoods, False)
            
            elif pattern_type == 'simple_count':
                if detected_samples:
                    # Return both above and below (is_above=None means show both)
                    return self._handle_count_samples_limit(detected_samples, detected_neighborhoods, None)
            
            elif pattern_type == 'comprehensive_analysis':
                if detected_samples:
                    return self._handle_comprehensive_analysis(detected_samples[0])
            
            elif pattern_type == 'pesticide_specific':
                if detected_pesticide and detected_samples:
                    # Check if it actually contains limit/compliance keywords (misclassified pattern)
                    limit_keywords = ['فوق', 'الحد', 'تجاوز', 'مخالف', 'غير مطابق', 'راسب', 'above', 'limit']
                    if any(kw in query for kw in limit_keywords):
                        # Redirect to _handle_sample_pesticide_limit logic
                        non_compliant_keywords = ['غير مطابق', 'الغير مطابقة', 'راسب', 'مخالف', 'non-compliant']
                        is_non_compliant = any(kw in query for kw in non_compliant_keywords)
                        
                        if is_non_compliant:
                            limit_filter = "AND sample_result LIKE '%non-compliant%'"
                            limit_desc = "غير مطابقة"
                        else:
                            limit_filter = None
                            limit_desc = None
                            
                        return self._handle_sample_pesticide_limit(
                            detected_samples, detected_pesticide, True,
                            limit_filter=limit_filter, limit_desc=limit_desc
                        )
                    
                    return self._handle_find_pesticide_in_sample(detected_pesticide, detected_samples)
            
            elif pattern_type == 'pesticide_limit_specific':
                if detected_pesticide and detected_samples:
                    # Check for non-compliant keywords
                    # Check for non-compliant keywords vs above limit
                    non_compliant_keywords = ['غير مطابق', 'الغير مطابقة', 'راسب', 'مخالف', 'non-compliant']
                    is_non_compliant = any(kw in query for kw in non_compliant_keywords)
                    
                    if is_non_compliant:
                        # Use sample_result logic
                        limit_filter = "AND sample_result LIKE '%non-compliant%'"
                        limit_desc = "غير مطابقة"
                        is_above = True
                    else:
                        # Default to above limit logic (is_above=True)
                        limit_filter = None
                        limit_desc = None
                        is_above = True
                        
                    return self._handle_sample_pesticide_limit(
                        detected_samples, detected_pesticide, is_above,
                        limit_filter=limit_filter, limit_desc=limit_desc
                    )
            
            elif pattern_type == 'list_pesticides':
                if detected_samples:
                    return self._handle_list_pesticides(detected_samples)
            
            elif pattern_type == 'neighborhood_pesticides':
                if detected_neighborhoods:
                    return self._handle_neighborhood_pesticides(detected_neighborhoods, show_separately)
            
            elif pattern_type == 'samples_with_n_pesticides':
                # Extract number from query
                import re
                numbers = re.findall(r'\d+', query)
                if numbers:
                    n_pesticides = int(numbers[0])
                    if 1 <= n_pesticides <= 50:
                        return self._handle_n_pesticides(n_pesticides, detected_samples)
            
            elif pattern_type == 'statistics':
                if detected_samples:
                    return self._handle_statistics(detected_samples[0], query)
            
            elif pattern_type == 'comprehensive_neighborhood_analysis':
                # This pattern handles queries like "ما هي انواع التوابل في حي الاسكان وما عددها وفوق/تحت الحد"
                # It requires neighborhood detection and will analyze all sample types in that neighborhood
                if detected_neighborhoods:
                    # Call the comprehensive neighborhood handler directly
                    # (it's implemented starting at line 603 in process())
                    return self._handle_comprehensive_neighborhood(query, detected_samples, detected_neighborhoods)
            
            elif pattern_type == 'facility_search':
                # Extract facility name from query
                return self._handle_facility_search(query)
            
            # Pattern recognized but couldn't be handled - fall back to keyword matching
            return None
            
        except Exception as e:
            logging.warning(f"Error in semantic routing: {e}")
            return None
    
    def _dispatch_by_intent(self, intent, entities) -> Optional[Tuple[str, Optional[pd.DataFrame]]]:
        """Route an intent + entities to the correct handler.
        
        Args:
            intent: Intent enum from IntentRouter
            entities: QueryEntities object with extracted information
        
        Returns:
            (response_text, dataframe) tuple or None if intent is UNKNOWN
        
        The handlers are the EXISTING methods — no changes needed to them.
        This dispatcher just translates intent+entities into method calls.
        """
        if not HAS_INTENT_ROUTER or intent is None:
            return None
        
        try:
            query = entities.normalized_query
            
            # ── Neighborhood + Category analysis ──
            if intent == Intent.COMPREHENSIVE_NEIGHBORHOOD or intent == Intent.LIST_TYPES_IN_NEIGHBORHOOD:
                return self._handle_comprehensive_neighborhood(
                    query, entities.samples, entities.neighborhoods
                )
            
            if intent == Intent.NEIGHBORHOOD_PESTICIDES:
                return self._handle_neighborhood_pesticides(
                    entities.neighborhoods, entities.wants_separately
                )
            
            if intent == Intent.NEIGHBORHOOD_RANKING:
                return self._handle_neighborhood_ranking()
            
            # ── N pesticides ──
            if intent == Intent.COUNT_N_PESTICIDES and entities.n_pesticides:
                if len(entities.n_pesticides) > 1 and entities.wants_separately:
                    return self._handle_multiple_n_pesticides(entities.n_pesticides, entities.samples)
                return self._handle_n_pesticides(entities.n_pesticides[-1], entities.samples)
            
            # ── Pesticide queries ──
            if intent == Intent.FIND_PESTICIDE_IN_SAMPLE and entities.pesticide and entities.samples:
                if entities.wants_limit_breakdown:
                    return self._handle_sample_pesticide_limit(
                        entities.samples, entities.pesticide, 
                        entities.is_above_limit if entities.is_above_limit is not None else True
                    )
                return self._handle_find_pesticide_in_sample(entities.pesticide, entities.samples)
            
            if intent == Intent.FIND_PESTICIDE_ALL and entities.pesticide:
                return self._handle_find_pesticide_all(entities.pesticide)
            
            if intent == Intent.LIST_PESTICIDES_IN_SAMPLE and entities.samples:
                return self._handle_list_pesticides(entities.samples)
            
            if intent == Intent.PESTICIDE_STATISTICS and entities.pesticide:
                return self._handle_pesticide_stats(
                    entities.pesticide, entities.samples, entities.stat_types
                )
            
            # ── Sample counting ──
            if intent == Intent.COUNT_SAMPLES_LIMIT and entities.samples:
                return self._handle_count_samples_limit(
                    entities.samples, entities.neighborhoods, entities.is_above_limit
                )
            
            if intent == Intent.COUNT_SAMPLES_SIMPLE and entities.samples:
                return self._handle_simple_sample_count(entities.samples, entities.neighborhoods)
            
            if intent == Intent.UNIQUE_COUNT and entities.samples:
                return self._handle_unique_samples_count(entities.samples, entities.neighborhoods)
            
            # ── Other ──
            if intent == Intent.COMPREHENSIVE_ANALYSIS and entities.samples:
                return self._handle_comprehensive_analysis(entities.samples)
            
            if intent == Intent.FACILITY_SEARCH:
                return self._handle_facility_search(query)
            
            if intent == Intent.RECIPIENT_SEARCH:
                return self._handle_facility_search(query)  # Same handler for now
        
        except Exception as ex:
            logging.warning(f"Intent dispatch error for {intent}: {ex}")
        
        return None  # UNKNOWN → falls through to patterns/LLM
    
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
        # NEW: Semantic Pattern Recognition (First Priority)
        # Uses embeddings to understand query intent semantically
        # Falls back to keyword matching if confidence is low
        # ============================================================
        if self.semantic_recognizer:
            semantic_result = self.semantic_recognizer.recognize(query)
            if semantic_result and semantic_result['confidence'] >= 0.75:
                pattern_type = semantic_result['pattern_type']
                logging.debug(f"Semantic match: {pattern_type} ({semantic_result['confidence']:.2f})")
                
                # Route to appropriate handler based on semantic pattern
                result = self._route_by_semantic_pattern(
                    pattern_type, query, query_normalized, query_lower,
                    detected_samples, detected_neighborhoods, detected_pesticide
                )
                if result is not None:
                    return result
        
        # ============================================================
        # NEW: Intent-Based Router (Second Priority)
        # Flexible intent+entity matching that handles query variations.
        # Falls back to pattern matching if intent is UNKNOWN.
        # ============================================================
        if self.router:
            intent, entities = self.router.analyze(query)
            
            if intent != Intent.UNKNOWN:
                logging.info(f"🎯 Intent: {intent.name} | samples={entities.samples} "
                             f"neighborhoods={entities.neighborhoods} category={entities.category} "
                             f"pesticide={entities.pesticide}")
                
                result = self._dispatch_by_intent(intent, entities)
                if result is not None:
                    return result
        
        
        # ============================================================
        # Pattern 0: Comprehensive neighborhood + sample type analysis
        # "ما هي انواع التوابل في حي الاسكان وما عددها وفوق/تحت الحد"
        # This includes samples with NO pesticides detected (MUST BE FIRST)
        # ============================================================
        category_keywords = ['انواع', 'أنواع', 'types', 'الموجودة', 'موجوده', 'what are']
        limit_keywords = ['فوق الحد', 'تحت الحد', 'above', 'below', 'limit', 'المسموح', 'مخالف', 'سليم']
        
        # Check if it's a comprehensive query asking about types + counts + limits + neighborhood
        is_comprehensive_hood_query = (
            any(kw in query for kw in category_keywords) and 
            detected_neighborhoods and
            (any(kw in query for kw in limit_keywords) or 'عددها' in query or 'ماعددها' in query)
        )
        
        if is_comprehensive_hood_query:
            return self._handle_comprehensive_neighborhood(query, detected_samples, detected_neighborhoods)
        
        # ============================================================
        # Pattern 0B: Category/types in neighborhood (WITHOUT limit keywords)
        # "ما هي انواع التوابل في حي الاسكان"
        # "التوابل في حي الريان"  
        # "انواع الخضار في الاسكان"
        # This catches queries that Pattern 0 misses because they
        # don't mention limits. We route to the SAME handler — it
        # already shows types + counts + limits in its output.
        # ============================================================
        category_names = ['توابل', 'التوابل', 'بهارات', 'خضار', 'الخضار', 'خضروات',
                         'فاكهة', 'فواكه', 'ورقيات', 'الورقيات', 'مكسرات', 'المكسرات',
                         'حبوب', 'الحبوب']
        has_category = any(cat in query for cat in category_names)
        has_type_question = any(kw in query for kw in category_keywords)  # "انواع", "الموجودة", etc.
        
        if (has_category or (has_type_question and detected_samples)) and detected_neighborhoods:
            return self._handle_comprehensive_neighborhood(query, detected_samples, detected_neighborhoods)
        
        # Continue with other patterns if not comprehensive
        con = self._get_connection()
        
        # ============================================================
        # Pattern 1: Samples with N pesticides (supports multiple counts)
        # "عدد العينات التي تحتوي على 6 مبيدات"
        # "عدد العينات التي تحتوي على 1 و 2 و 3 مبيدات كل على حده"
        # "العينات الخالية من المبيدات والتي تحتوي على مبيد واحد وثلاث مبيدات"
        # ============================================================
        pesticide_count_keywords = ['مبيد', 'مبيدات', 'pesticide', 'pesticides']
        zero_pesticide_keywords = ['خالية', 'الخالية', 'بدون مبيدات', 'لا تحتوي',
                                   'خاليه', 'نظيفة', 'سليمة تماما', 'zero pesticide']
        has_zero_request = any(kw in query for kw in zero_pesticide_keywords)
        
        if any(kw in query for kw in pesticide_count_keywords) or has_zero_request:
            # Extract all numbers from the query
            all_numbers = re.findall(r'(\d+)', query_normalized)
            
            # ★ إذا طلب "الخالية" أضف 0 للقائمة (بعد تعريف all_numbers)
            if has_zero_request and '0' not in all_numbers:
                all_numbers.insert(0, '0')
            
            if all_numbers:
                # Check if it's asking for multiple counts ("كل علي حده" or "و")
                is_multiple = any(kw in query for kw in ['كل علي حده', 'كل على حده', 'كل على حدة', 
                                                          'each', 'separately', 'بشكل منفصل'])
                
                # Check if query structure suggests comparison (e.g., "1 و 2 و 3")
                has_multiple_with_and = len(all_numbers) > 1 and ('و' in query or 'and' in query.lower())
                
                if (is_multiple or has_multiple_with_and) and len(all_numbers) > 1:
                    # Handle multiple pesticide counts
                    pesticide_counts = [int(n) for n in all_numbers if int(n) <= 50]
                    if pesticide_counts:
                        return self._handle_multiple_n_pesticides(pesticide_counts, detected_samples)
                else:
                    # Single count - get the most relevant number
                    n_pesticides = int(all_numbers[-1])
                    if 0 <= n_pesticides <= 50:  # ★ كان 1 <= أصبح 0 <=
                        return self._handle_n_pesticides(n_pesticides, detected_samples)
        
        # ============================================================
        # Pattern 2: Count samples above/below limit (pesticide concentration)
        # "عدد عينات الطماطم فوق الحد"
        # This checks if ANY pesticide in the sample exceeds its limit
        # ============================================================
        count_phrases = ['count', 'how many', 'عدد', 'كم', 'ماهي', 'ما هي', 'what is']
        # Expanded keywords for ABOVE LIMIT (non-compliant/failing)
        above_limit_phrases = [
            'more than limit', 'exceeding', 'above limit', 'over limit',
            'reading more', 'exceed', 'فوق الحد', 'تجاوز الحد', 'متجاوزة',
            'مخالف', 'مخالفة', 'غير مطابق', 'غير مطابقة', 'الغير مطابقة', 'الغير مطابق',
            'راسب', 'راسبة', 'فاشل', 'فاشلة', 'failing', 'violated', 'violation', 'non-compliant'
        ]
        # Expanded keywords for BELOW LIMIT (compliant/passing)
        below_limit_phrases = [
            'below limit', 'under limit', 'within limit', 'تحت الحد', 
            'ضمن الحد', 'لم تتجاوز',
            'سليم', 'سليمة', 'مطابق', 'مطابقة', 'ناجح', 'ناجحة',
            'compliant', 'passing', 'safe'
        ]
        
        is_count_query = any(phrase in query_lower for phrase in count_phrases)
        is_above_limit = any(phrase in query for phrase in above_limit_phrases)
        is_below_limit = any(phrase in query for phrase in below_limit_phrases)
        
        # IMPORTANT DISTINCTION:
        # - "غير مطابقة" = sample_result (chemist's final decision)
        # - "فوق الحد" = is_above_limit (technical exceedance)
        # These are DIFFERENT! A sample can be above limit but still compliant.
        
        # Keywords for SAMPLE RESULT (chemist decision)
        non_compliant_result_keywords = [
            'غير مطابق', 'غير مطابقة', 'الغير مطابقة', 'الغير مطابق',
            'non-compliant', 'non compliant', 'راسب', 'راسبة'
        ]
        compliant_result_keywords_for_pesticide = [
            'مطابق', 'مطابقة', 'المطابقة', 'ناجح', 'ناجحة'
        ]
        
        # Check if asking about compliance (sample_result) vs limit (is_above_limit)
        is_non_compliant_query = any(kw in query for kw in non_compliant_result_keywords)
        # Only consider compliant query if no limit keywords are present
        is_compliant_query = (any(kw in query for kw in compliant_result_keywords_for_pesticide) 
                              and not is_non_compliant_query
                              and not is_above_limit  # Don't treat "فوق الحد" as compliant query
                              and not is_below_limit)  # Don't treat "تحت الحد" as compliant query
        use_sample_result = is_non_compliant_query or is_compliant_query
        
        # Check for explicit "both" request (فوق الحد و تحت الحد)
        # Note: Use full phrases to avoid matching 'تحت' inside 'تحتوي'
        explicit_both = (('فوق الحد' in query and 'تحت الحد' in query) or 
                        ('above limit' in query_lower and 'below limit' in query_lower))
        explicit_both = explicit_both and not is_non_compliant_query
        
        if is_count_query and detected_samples and detected_pesticide and (is_above_limit or is_below_limit or use_sample_result):
             # This is a specific query: Sample + Pesticide + Limit/Compliance
             if use_sample_result:
                 # Use sample_result (chemist's decision)
                 if is_non_compliant_query:
                     limit_filter = "AND sample_result LIKE '%non-compliant%'"
                     limit_desc = "غير مطابقة"
                 else:
                     limit_filter = "AND sample_result LIKE '%compliant%' AND sample_result NOT LIKE '%non-compliant%'"
                     limit_desc = "مطابقة"
                 return self._handle_sample_pesticide_limit(detected_samples, detected_pesticide, is_above_limit, 
                                                            limit_filter=limit_filter, limit_desc=limit_desc)
             elif explicit_both:
                 return self._handle_sample_pesticide_limit(detected_samples, detected_pesticide, is_above_limit, both=True)
             else:
                 return self._handle_sample_pesticide_limit(detected_samples, detected_pesticide, is_above_limit)

        if is_count_query and detected_samples and (is_above_limit or is_below_limit):
            return self._handle_count_samples_limit(detected_samples, detected_neighborhoods, is_above_limit)
        
        # ============================================================
        # Pattern 2B: Count compliant/non-compliant samples (sample_result)
        # "عدد عينات الخيار غير المطابقة" or "عينات الخيار المطابقة"
        # This uses the official sample_result column, NOT pesticide limits
        # ============================================================
        compliant_keywords = ['مطابقة', 'المطابقة', 'مطابق', 'compliant', 'passed']
        non_compliant_keywords = ['غير مطابقة', 'غير المطابقة', 'الغير مطابقة', 'الغير المطابقة', 
                                   'non-compliant', 'non compliant', 'failed', 'مخالف', 'راسب', 'راسبة']
        
        # Check if asking about compliance status (not limit)
        is_non_compliant_query = any(kw in query for kw in non_compliant_keywords)
        is_compliant_query = any(kw in query for kw in compliant_keywords) and not is_non_compliant_query
        
        if detected_samples and (is_non_compliant_query or is_compliant_query):
            # Check it's not a limit query
            if not is_above_limit and not is_below_limit:
                return self._handle_count_samples_compliance(detected_samples, detected_neighborhoods, 
                                                              is_non_compliant_query)
        
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
        
        # ============================================================
        # Pattern 8: Pesticide statistics (max, min, range, median, average)
        # "ماهو اعلي تركيز و اقل تركيز لمبيد الايميداكلوبرايد"
        # "ماهو المدي لتركيزات الايميداكلوبرايد في الطماطم"
        # "ماهو المتوسط و الوسيط لمبيد الايميداكلوبرايد في الطماطم"
        # ============================================================
        stats_keywords_found = []
        for stat_type, keywords in self.stats_keywords.items():
            if any(kw in query_lower or kw in query for kw in keywords):
                stats_keywords_found.append(stat_type)
        
        if stats_keywords_found and detected_pesticide:
            return self._handle_pesticide_stats(detected_pesticide, detected_samples, stats_keywords_found)
        
        # ============================================================
        # Pattern 9: Search for samples by establishment/recipient
        # "ابحث عن العينات في منشأة محامص مذاق ضيافة الخير"
        # "ماهي عينات مشعل الرشيدي المطابقة الفريدة"
        # "ماهي عينات المكسرات للمستلم أحمد الفايز"
        # ============================================================
        recipient_keywords = ['المستلم', 'مستلم', 'recipient']
        establishment_keywords = ['منشأة', 'المنشأة', 'محل', 'المحل', 'establishment', 'facility']
        
        # Extract recipient or establishment name
        recipient_match = re.search(r'(?:المستلم|مستلم|للمستلم)\s+([^\s]+(?:\s+[^\s]+)?)', query)
        establishment_match = re.search(r'(?:منشأة|المنشأة|محل)\s+(.+?)(?:\s+(?:و|وما|ماهي|المطابقة|الفريدة|$))', query)
        
        if recipient_match or establishment_match or any(kw in query for kw in recipient_keywords + establishment_keywords):
            name = None
            search_type = None
            
            if recipient_match:
                name = recipient_match.group(1).strip()
                search_type = 'recipient'
            elif establishment_match:
                name = establishment_match.group(1).strip()
                search_type = 'establishment'
            else:
                # Try to extract any name mentioned
                name_match = re.search(r'(?:عينات|samples)\s+([^\s]+(?:\s+[^\s]+)?(?:\s+[^\s]+)?)', query)
                if name_match:
                    name = name_match.group(1).strip()
                    search_type = 'any'
            
            if name:
                is_unique = any(kw in query for kw in ['فريدة', 'فريد', 'unique', 'distinct'])
                is_compliant = any(kw in query for kw in self.below_limit_keywords)
                is_non_compliant = any(kw in query for kw in self.above_limit_keywords)
                return self._handle_recipient_establishment_query(
                    name, search_type, detected_samples, is_unique, is_compliant, is_non_compliant
                )
        
        # ============================================================
        # Pattern 10: Samples containing specific pesticide (without sample type filter)
        # "كم عدد العينات التي تحتوي علي مبيد الايثيون"
        # ============================================================
        if detected_pesticide and not detected_samples:
            # Check if it's asking about samples with this pesticide
            if any(kw in query for kw in ['عينات', 'العينات', 'samples', 'عدد', 'كم']):
                return self._handle_samples_with_pesticide(detected_pesticide)
        
        # ============================================================
        # Pattern 11: Unique sample count
        # "كم عدد عينات الخيار الفريدة بناءً على كود العينة"
        # ============================================================
        if detected_samples and any(kw in query for kw in ['فريدة', 'فريد', 'unique', 'distinct', 'كود العينة']):
            return self._handle_unique_samples_count(detected_samples, detected_neighborhoods)
        
        # Pattern 12: Just search for pesticide (no sample filter)
        # "ابحث عن الفيبرونيل" or "Find fipronil"
        # ============================================================
        search_keywords = ['ابحث', 'دور', 'اوجد', 'find', 'search', 'locate']
        if detected_pesticide and any(kw in query_lower for kw in search_keywords):
            return self._handle_find_pesticide_all(detected_pesticide)
        
        # ============================================================
        # Pattern 14: Simple sample count (NO CONDITIONS)
        # "ما عدد عينات الطماطم و الخيار و الكوسة"
        # "كم عينة خيار"
        # This MUST be after limit/compliance patterns to avoid conflicts
        # ============================================================
        count_keywords = ['عدد', 'كم', 'count', 'how many', 'عينات', 'العينات']
        if detected_samples and any(kw in query for kw in count_keywords):
            # Only trigger if NOT a limit/compliance query (those are handled above)
            return self._handle_simple_sample_count(detected_samples, detected_neighborhoods)
        
        # ============================================================
        # Pattern 14: LLM Fallback for unknown queries
        # If we have LLM configured, try to generate SQL
        # ============================================================
        if self.llm_client is not None:
            llm_response = self._handle_llm_query(query, detected_samples, detected_neighborhoods, detected_pesticide)
            if llm_response[0]:
                return llm_response
        
        # Default: Unknown query
        return self._handle_unknown_query(query), None
    
    # ============================================================
    # Handlers
    # ============================================================
    
    def _handle_n_pesticides(self, n: int, samples: List[str]) -> Tuple[str, pd.DataFrame]:
        """عينات بعدد N مبيدات"""
        if n == 0:
            return self._handle_multiple_n_pesticides([0], samples)
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
    def _handle_multiple_n_pesticides(self, counts: List[int], samples: List[str]) -> Tuple[str, pd.DataFrame]:
    """
    عينات بأعداد مختلفة من المبيدات (يشمل 0 = خالية)
    يدعم: "العينات الخالية من المبيدات و التي تحتوي علي مبيد واحد و ثلاث مبيدات"
    """
    con = self._get_connection()
    
    sample_filter = ""
    if samples:
        conditions = [f'"اسم العينة" LIKE \'%{s}%\'' for s in samples]
        sample_filter = f"AND ({' OR '.join(conditions)})"
    
    has_zero = 0 in counts
    non_zero_counts = [c for c in counts if c > 0]
    
    all_dfs = []
    summary_rows = []
    details_responses = []
    
    # ── Part A: Zero-pesticide samples ──
    if has_zero:
        zero_sql = f"""
        WITH detected_samples AS (
            SELECT DISTINCT "كود العينة"
            FROM chemistry_tidy
            WHERE is_detected = 1
            AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
            {sample_filter}
        )
        SELECT 
            "كود العينة" as sample_code,
            "اسم العينة" as sample_name,
            0 as pesticide_count
        FROM chemistry_tidy
        WHERE "كود العينة" NOT IN (SELECT "كود العينة" FROM detected_samples)
        {sample_filter}
        GROUP BY "كود العينة", "اسم العينة"
        ORDER BY "كود العينة" DESC
        """
        zero_df = con.execute(zero_sql).df()
        all_dfs.append(zero_df)
        
        if not zero_df.empty:
            type_counts = zero_df['sample_name'].value_counts()
            breakdown_parts = [f"{count} {name}" for name, count in type_counts.head(10).items()]
            breakdown = "، ".join(breakdown_parts)
            if len(type_counts) > 10:
                breakdown += f"، و {len(type_counts) - 10} أنواع أخرى"
            
            summary_rows.append({"عدد المبيدات": "0 (خالية)", "عدد العينات": len(zero_df)})
            details_responses.append(
                f"#### 🛡️ العينات الخالية من المبيدات (0 مبيد)\n\n"
                f"- ✅ **العدد:** {len(zero_df)} عينة\n"
                f"- 📋 **الأنواع:** {breakdown}\n"
            )
        else:
            details_responses.append("#### 🛡️ العينات الخالية\n\nلم أجد عينات خالية من المبيدات")
    
    # ── Part B: Non-zero pesticide counts ──
    for n in sorted(non_zero_counts):
        n_sql = f"""
        SELECT 
            "كود العينة" as sample_code,
            "اسم العينة" as sample_name,
            COUNT(*) as pesticide_count
        FROM chemistry_tidy
        WHERE is_detected = 1 
        AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
        {sample_filter}
        GROUP BY "كود العينة", "اسم العينة"
        HAVING COUNT(*) = {n}
        ORDER BY "كود العينة" DESC
        """
        n_df = con.execute(n_sql).df()
        all_dfs.append(n_df)
        
        label = "مبيد واحد" if n == 1 else f"{n} مبيدات"
        icon = "🔹" if n == 1 else "🔸"
        
        if not n_df.empty:
            type_counts = n_df['sample_name'].value_counts()
            breakdown_parts = [f"{count} {name}" for name, count in type_counts.head(10).items()]
            breakdown = "، ".join(breakdown_parts)
            if len(type_counts) > 10:
                breakdown += f"، و {len(type_counts) - 10} أنواع أخرى"
            
            summary_rows.append({"عدد المبيدات": n, "عدد العينات": len(n_df)})
            details_responses.append(
                f"#### {icon} العينات بـ {label}\n\n"
                f"- ✅ **العدد:** {len(n_df)} عينة\n"
                f"- 📋 **الأنواع:** {breakdown}\n"
            )
        else:
            details_responses.append(f"#### {icon} العينات بـ {label}\n\nلم أجد عينات")
    
    con.close()
    
    # ── Summary table ──
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="عدد المبيدات" if "عدد المبيدات" in summary_df.columns else 0)
    
    # ── Build final response ──
    counts_display = '، '.join(
        ["الخالية (0)" if c == 0 else f"{c} مبيد" if c == 1 else f"{c} مبيدات" for c in sorted(counts)]
    )
    sample_display = ' + '.join(samples) if samples else "جميع الأنواع"
    
    response = f"## 📊 توزيع عينات {sample_display} حسب عدد المبيدات\n\n"
    response += f"🔢 **المطلوب:** {counts_display}\n\n"
    
    if not summary_df.empty:
        response += "### 📋 جدول ملخص\n\n"
        response += summary_df.to_markdown(index=False)
        response += "\n\n---\n\n"
    
    response += "### 🔍 تفاصيل كل فئة\n\n"
    response += "\n\n---\n\n".join(details_responses)
    
    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    
    return response, combined_df
    
    def _handle_count_samples_limit(self, samples: List[str], neighborhoods: List[str], 
                                     is_above: bool) -> Tuple[str, pd.DataFrame]:
        """عد العينات فوق/تحت الحد - بناءً على كود العينة الفريد"""
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
        
        # Use CTE to correctly count UNIQUE SAMPLES based on sample code
        # A sample is "above limit" if it has AT LEAST ONE pesticide above limit
        sql = f"""
        WITH sample_status AS (
            SELECT 
                "كود العينة" as sample_code,
                "اسم العينة" as sample_name,
                MAX(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as has_violation
            FROM chemistry_tidy
            WHERE is_detected = 1
            {sample_filter}
            {neighborhood_filter}
            GROUP BY "كود العينة", "اسم العينة"
        )
        SELECT 
            sample_name as نوع_العينة,
            COUNT(*) as عدد_العينات,
            SUM(has_violation) as عينات_فوق_الحد,
            SUM(CASE WHEN has_violation = 0 THEN 1 ELSE 0 END) as عينات_تحت_الحد
        FROM sample_status
        GROUP BY sample_name
        ORDER BY عدد_العينات DESC
        """
        
        df = con.execute(sql).df()
        con.close()
        
        type_display = " + ".join(samples)
        if neighborhoods:
            type_display += f" في {' + '.join(neighborhoods)}"
        
        if not df.empty:
            total_samples = int(df['عدد_العينات'].sum())
            total_above = int(df['عينات_فوق_الحد'].sum())
            total_below = int(df['عينات_تحت_الحد'].sum())
            
            response = f"📊 **نتائج عينات {type_display}:**\n\n"
            response += f"✅ إجمالي العينات الفريدة: **{total_samples}** عينة\n"
            response += f"🔴 عينات فوق الحد: **{total_above}** (لديها مبيد واحد على الأقل متجاوز)\n"
            response += f"🟢 عينات تحت الحد: **{total_below}** (جميع المبيدات ضمن الحد)\n\n"
            response += df.to_markdown(index=False)
        else:
            response = f"⚠️ لم يتم العثور على عينات {type_display}"
        
        return response, df
    
    def _handle_count_samples_compliance(self, samples: List[str], neighborhoods: List[str], 
                                          is_non_compliant: bool) -> Tuple[str, pd.DataFrame]:
        """عد العينات المطابقة/غير المطابقة - بناءً على عمود نتيجة العينة (sample_result)"""
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
        
        # Query using sample_result column
        sql = f"""
        SELECT 
            "اسم العينة" as نوع_العينة,
            sample_result as نتيجة_العينة,
            COUNT(DISTINCT "كود العينة") as عدد_العينات
        FROM chemistry_tidy
        WHERE 1=1
        {sample_filter}
        {neighborhood_filter}
        GROUP BY "اسم العينة", sample_result
        ORDER BY عدد_العينات DESC
        """
        
        df = con.execute(sql).df()
        con.close()
        
        type_display = " + ".join(samples)
        if neighborhoods:
            type_display += f" في {' + '.join(neighborhoods)}"
        
        if not df.empty:
            # Summarize by compliance status (use regex=False to avoid regex issues)
            compliant = df[df['نتيجة_العينة'].str.contains('compliant', na=False, regex=False) & 
                          ~df['نتيجة_العينة'].str.contains('non-compliant', na=False, regex=False)]['عدد_العينات'].sum()
            non_compliant = df[df['نتيجة_العينة'].str.contains('non-compliant', na=False, regex=False)]['عدد_العينات'].sum()
            unknown = df[df['نتيجة_العينة'].isna()]['عدد_العينات'].sum()
            total = int(df['عدد_العينات'].sum())
            
            status_text = "غير المطابقة" if is_non_compliant else "المطابقة"
            target_count = int(non_compliant) if is_non_compliant else int(compliant)
            
            response = f"📊 **نتائج عينات {type_display} {status_text}:**\n\n"
            response += f"✅ إجمالي العينات الفريدة: **{total}** عينة\n"
            response += f"🟢 مطابقة: **{int(compliant)}**\n"
            response += f"🔴 غير مطابقة: **{int(non_compliant)}**\n"
            if unknown > 0:
                response += f"⚪ غير محدد: **{int(unknown)}**\n"
            response += f"\n📌 **العينات {status_text}: {target_count}**\n\n"
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
        AND pesticide_name IN ({', '.join([f"'{v}'" for v in get_pesticide_variants(pesticide)])})
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
    
    def _handle_pesticide_stats(self, pesticide: str, samples: List[str], 
                                 stats_requested: List[str]) -> Tuple[str, pd.DataFrame]:
        """إحصائيات مبيد محدد (أعلى، أقل، مدى، متوسط، وسيط)"""
        con = self._get_connection()
        
        sample_filter = ""
        if samples:
            conditions = [f"\"اسم العينة\" LIKE '%{s}%'" for s in samples]
            sample_filter = f"AND ({' OR '.join(conditions)})"
        
        sql = f"""
        SELECT 
            pesticide_name as المبيد,
            COUNT(*) as عدد_الاكتشافات,
            ROUND(MIN(concentration), 4) as أقل_تركيز,
            ROUND(MAX(concentration), 4) as أعلى_تركيز,
            ROUND(AVG(concentration), 4) as المتوسط,
            ROUND(MEDIAN(concentration), 4) as الوسيط,
            ROUND(MAX(concentration) - MIN(concentration), 4) as المدى,
            ROUND(MAX(limit_value), 4) as MRL,
            SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as التجاوزات
        FROM chemistry_tidy
        WHERE is_detected = 1
        AND pesticide_name IN ({', '.join([f"'{v}'" for v in get_pesticide_variants(pesticide)])})
        {sample_filter}
        GROUP BY pesticide_name
        """
        
        df = con.execute(sql).df()
        con.close()
        
        sample_display = ' + '.join(samples) if samples else "جميع العينات"
        
        if not df.empty:
            response = f"📊 **إحصائيات مبيد {pesticide}**\n"
            response += f"🧪 في: {sample_display}\n\n"
            
            row = df.iloc[0]
            response += f"📈 **الإحصائيات:**\n"
            response += f"• عدد الاكتشافات: **{int(row['عدد_الاكتشافات'])}**\n"
            response += f"• أقل تركيز: **{row['أقل_تركيز']}** mg/kg\n"
            response += f"• أعلى تركيز: **{row['أعلى_تركيز']}** mg/kg\n"
            response += f"• المتوسط: **{row['المتوسط']}** mg/kg\n"
            response += f"• الوسيط: **{row['الوسيط']}** mg/kg\n"
            response += f"• المدى: **{row['المدى']}** mg/kg\n"
            response += f"• الحد المسموح (MRL): **{row['MRL']}** mg/kg\n"
            response += f"• عدد التجاوزات: **{int(row['التجاوزات'])}**\n"
        else:
            response = f"⚠️ لم أجد مبيد {pesticide} في {sample_display}"
        
        return response, df
    
    def _handle_recipient_establishment_query(self, name: str, search_type: str,
                                               samples: List[str], is_unique: bool,
                                               is_compliant: bool, is_non_compliant: bool) -> Tuple[str, pd.DataFrame]:
        """البحث عن العينات حسب المستلم أو المنشأة"""
        con = self._get_connection()
        
        # Build filters
        filters = []
        
        # Name filter (search in multiple columns)
        name_conditions = [
            f"\"اسم المستلم\" LIKE '%{name}%'",
            f"\"اسم المستلم\" ILIKE '%{name}%'",
        ]
        # Also try to search in establishment if column exists
        filters.append(f"({' OR '.join(name_conditions)})")
        
        # Sample type filter
        if samples:
            sample_conditions = [f"\"اسم العينة\" LIKE '%{s}%'" for s in samples]
            filters.append(f"({' OR '.join(sample_conditions)})")
        
        # Compliance filter
        if is_compliant and not is_non_compliant:
            filters.append("is_compliant = 1")
        elif is_non_compliant and not is_compliant:
            filters.append("is_compliant = 0")
        
        where_clause = " AND ".join(filters)
        
        if is_unique:
            sql = f"""
            SELECT DISTINCT
                "كود العينة" as كود_العينة,
                "اسم العينة" as اسم_العينة,
                "اسم المستلم" as المستلم,
                MAX(CASE WHEN is_compliant = 1 THEN 'مطابق' ELSE 'غير مطابق' END) as الحالة
            FROM chemistry_tidy
            WHERE {where_clause}
            GROUP BY "كود العينة", "اسم العينة", "اسم المستلم"
            ORDER BY "كود العينة" DESC
            LIMIT 100
            """
        else:
            sql = f"""
            SELECT 
                "كود العينة" as كود_العينة,
                "اسم العينة" as اسم_العينة,
                "اسم المستلم" as المستلم,
                pesticide_name as المبيد,
                concentration as التركيز,
                CASE WHEN is_compliant = 1 THEN 'مطابق' ELSE 'غير مطابق' END as الحالة
            FROM chemistry_tidy
            WHERE {where_clause}
            AND is_detected = 1
            ORDER BY "كود العينة" DESC
            LIMIT 100
            """
        
        df = con.execute(sql).df()
        con.close()
        
        status_desc = ""
        if is_compliant:
            status_desc = " المطابقة"
        elif is_non_compliant:
            status_desc = " غير المطابقة (المخالفة)"
        
        unique_desc = " الفريدة" if is_unique else ""
        
        if not df.empty:
            response = f"🔍 **عينات{unique_desc}{status_desc} لـ {name}:**\n\n"
            response += f"✅ وجدت **{len(df)}** سجل\n\n"
            response += df.to_markdown(index=False)
        else:
            response = f"⚠️ لم أجد عينات لـ {name}"
        
        return response, df
    
    def _handle_samples_with_pesticide(self, pesticide: str) -> Tuple[str, pd.DataFrame]:
        """عينات تحتوي على مبيد محدد (بدون فلترة نوع العينة)"""
        con = self._get_connection()
        
        sql = f"""
        SELECT 
            "اسم العينة" as نوع_العينة,
            COUNT(DISTINCT "كود العينة") as عدد_العينات,
            COUNT(*) as عدد_الاكتشافات,
            SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as فوق_الحد,
            ROUND(AVG(concentration), 4) as متوسط_التركيز,
            ROUND(MAX(concentration), 4) as أعلى_تركيز
        FROM chemistry_tidy
        WHERE is_detected = 1
        AND pesticide_name IN ({', '.join([f"'{v}'" for v in get_pesticide_variants(pesticide)])})
        GROUP BY "اسم العينة"
        ORDER BY عدد_الاكتشافات DESC
        LIMIT 50
        """
        
        df = con.execute(sql).df()
        con.close()
        
        if not df.empty:
            total_samples = int(df['عدد_العينات'].sum())
            total_detections = int(df['عدد_الاكتشافات'].sum())
            total_above = int(df['فوق_الحد'].sum())
            
            response = f"🧪 **العينات التي تحتوي على {pesticide}:**\n\n"
            response += f"📊 **الملخص:**\n"
            response += f"• إجمالي العينات: **{total_samples}**\n"
            response += f"• إجمالي الاكتشافات: **{total_detections}**\n"
            response += f"• التجاوزات: **{total_above}**\n\n"
            response += df.to_markdown(index=False)
        else:
            response = f"⚠️ لم أجد عينات تحتوي على {pesticide}"
        
        return response, df
    
    def _handle_unique_samples_count(self, samples: List[str], 
                                      neighborhoods: List[str]) -> Tuple[str, pd.DataFrame]:
        """عدد العينات الفريدة"""
        con = self._get_connection()
        
        conditions = [f"\"اسم العينة\" LIKE '%{s}%'" for s in samples]
        sample_filter = f"({' OR '.join(conditions)})"
        
        neighborhood_filter = ""
        if neighborhoods:
            hood_conditions = [f"\"الحى\" LIKE '%{n}%'" for n in neighborhoods]
            neighborhood_filter = f"AND ({' OR '.join(hood_conditions)})"
        
        sql = f"""
        SELECT 
            "اسم العينة" as نوع_العينة,
            COUNT(DISTINCT "كود العينة") as العينات_الفريدة,
            COUNT(*) as إجمالي_السجلات,
            SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as سجلات_فوق_الحد
        FROM chemistry_tidy
        WHERE {sample_filter}
        {neighborhood_filter}
        GROUP BY "اسم العينة"
        ORDER BY العينات_الفريدة DESC
        """
        
        df = con.execute(sql).df()
        con.close()
        
        sample_display = ' + '.join(samples)
        location_desc = f" في {' + '.join(neighborhoods)}" if neighborhoods else ""
        
        if not df.empty:
            total_unique = int(df['العينات_الفريدة'].sum())
            
            response = f"📊 **العينات الفريدة لـ {sample_display}{location_desc}:**\n\n"
            response += f"✅ إجمالي العينات الفريدة: **{total_unique}**\n\n"
            response += df.to_markdown(index=False)
        else:
            response = f"⚠️ لم أجد عينات {sample_display}"
        
        return response, df
    
    def _handle_find_pesticide_all(self, pesticide: str) -> Tuple[str, pd.DataFrame]:
        """البحث عن مبيد في جميع العينات"""
        con = self._get_connection()
        
        sql = f"""
        SELECT 
            "كود العينة" as كود_العينة,
            "اسم العينة" as نوع_العينة,
            "الحى" as الحي,
            pesticide_name as المبيد,
            concentration as التركيز,
            limit_value as الحد,
            ROUND(exceedance_ratio, 2) as النسبة,
            CASE WHEN is_compliant = 1 THEN 'مطابق' ELSE 'غير مطابق' END as الحالة
        FROM chemistry_tidy
        WHERE is_detected = 1
        AND pesticide_name IN ({', '.join([f"'{v}'" for v in get_pesticide_variants(pesticide)])})
        ORDER BY concentration DESC
        LIMIT 100
        """
        
        df = con.execute(sql).df()
        con.close()
        
        if not df.empty:
            compliant = len(df[df['الحالة'] == 'مطابق'])
            non_compliant = len(df[df['الحالة'] == 'غير مطابق'])
            
            response = f"🔍 **نتائج البحث عن {pesticide}:**\n\n"
            response += f"✅ وجدت **{len(df)}** سجل\n"
            response += f"📊 {compliant} مطابق | {non_compliant} غير مطابق\n\n"
            response += df.to_markdown(index=False)
        else:
            response = f"⚠️ لم أجد مبيد {pesticide} في قاعدة البيانات"
        
        return response, df
    
    def _handle_comprehensive_neighborhood_samples(self, samples: List[str], 
                                                    neighborhoods: List[str], 
                                                    query: str) -> Tuple[str, pd.DataFrame]:
        """
        تحليل شامل للعينات في حي معين (يشمل العينات السليمة والخالية)
        "ما هي انواع التوابل في حي الاسكان وما عددها وفوق/تحت الحد"
        """
        con = self._get_connection()
        
        # Build neighborhood filter (mandatory)
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
        
        # Build category filter if applicable
        category_filter = ""
        category_name = "العينات"
        
        if 'توابل' in query or 'التوابل' in query or 'بهارات' in query:
            category_filter = """AND ("تصنيف العينة" LIKE '%توابل%' OR "اسم العينة" IN ('كمون', 'هيل', 'زعتر', 'كركم', 'بهارات', 'فلفل اسود', 'قرفة', 'يانسون', 'كزبرة', 'شمر', 'قرنفل'))"""
            category_name = "التوابل"
        elif 'خضار' in query or 'الخضار' in query or 'خضروات' in query:
             category_filter = """AND ("تصنيف العينة" LIKE '%خضار%' OR "اسم العينة" IN ('طماطم', 'خيار', 'كوسة', 'باذنجان', 'فلفل', 'بامية', 'فاصوليا', 'زهرة', 'بروكلي', 'ملفوف', 'خس', 'جرجير', 'بقدونس', 'كزبرة', 'شبت'))"""
             category_name = "الخضار"
        elif 'فاكهة' in query or 'فواكه' in query:
             category_filter = """AND ("تصنيف العينة" LIKE '%فاكهة%' OR "اسم العينة" IN ('تفاح', 'برتقال', 'عنب', 'مانجو', 'توت', 'رمان', 'ليمون', 'يوسف افندي', 'كمثرى', 'فراولة', 'خوخ', 'مشمش'))"""
             category_name = "الفواكه"
        elif 'ورقيات' in query or 'الورقيات' in query:
             category_filter = """AND ("تصنيف العينة" LIKE '%ورقيات%' OR "اسم العينة" IN ('خس', 'جرجير', 'بقدونس', 'كزبرة', 'شبت', 'نعناع', 'سبانخ', 'ملوخية', 'رجلة', 'سلق'))"""
             category_name = "الورقيات"
        elif samples:
            # Specific samples requested
            conditions = [f"\"اسم العينة\" LIKE '%{s}%'" for s in samples]
            category_filter = f"AND ({' OR '.join(conditions)})"
            category_name = " + ".join(samples)
            
        # Comprehensive SQL to get ALL samples (including empty ones)
        # Using MAX(is_above_limit) per sample to detect violations
        # Using MAX(is_detected) to see if ANY pesticide was found
        sql = f"""
        WITH sample_status AS (
            SELECT 
                "كود العينة" as sample_code,
                "اسم العينة" as sample_name,
                MAX(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as has_violation,
                MAX(CASE WHEN is_detected = 1 THEN 1 ELSE 0 END) as has_detection,
                COUNT(CASE WHEN is_detected = 1 THEN 1 END) as pesticide_count
            FROM chemistry_tidy
            WHERE 1=1
            {neighborhood_filter}
            {category_filter}
            GROUP BY "كود العينة", "اسم العينة"
        )
        SELECT 
            sample_name as نوع_العينة,
            COUNT(*) as العدد_الكلي,
            SUM(has_violation) as فوق_الحد,
            SUM(CASE WHEN has_violation = 0 THEN 1 ELSE 0 END) as تحت_الحد,
            SUM(CASE WHEN has_detection = 0 THEN 1 ELSE 0 END) as خالية_من_المبيدات
        FROM sample_status
        GROUP BY sample_name
        ORDER BY العدد_الكلي DESC
        """
        
        df = con.execute(sql).df()
        con.close()
        
        location_desc = f" في {' + '.join(neighborhoods)}"
        
        if not df.empty:
            total = int(df['العدد_الكلي'].sum())
            total_above = int(df['فوق_الحد'].sum())
            total_below = int(df['تحت_الحد'].sum())
            total_clean = int(df['خالية_من_المبيدات'].sum())
            
            response = f"📊 **تحليل {category_name}{location_desc}:**\n\n"
            response += f"✅ إجمالي العينات: **{total}** عينة\n"
            response += f"🔴 فوق الحد: **{total_above}**\n"
            response += f"🟢 تحت الحد: **{total_below}** (منها {total_clean} خالية تماماً)\n\n"
            response += df.to_markdown(index=False)
        else:
            response = f"⚠️ لم أجد عينات {category_name}{location_desc}"
        
        return response, df
    
    
    def _handle_sample_pesticide_limit(self, samples: List[str], pesticide: str, is_above: bool, 
                                        both: bool = False, limit_filter: str = None, limit_desc: str = None) -> Tuple[str, pd.DataFrame]:
        """
        عينات تحتوي على مبيد X فوق/تحت الحد أو غير مطابقة
        "عينات الطماطم التي تحتوي على مبيد البابروفيزن فوق الحد"
        "عينات الطماطم التي تحتوي على مبيد البابروفيزن غير مطابقة" (uses sample_result)
        """
        con = self._get_connection()
        
        # Build filters
        sample_conditions = [f"\"اسم العينة\" LIKE '%{s}%'" for s in samples]
        sample_filter = f"({' OR '.join(sample_conditions)})"
        
        # Clean pesticide name (remove 'AL' prefix if present)
        if pesticide.startswith('ال') and len(pesticide) > 4:
            pesticide = pesticide[2:]
        
        # Determine limit filter and description
        if limit_filter is not None and limit_desc is not None:
            # Custom filter provided (e.g., for sample_result queries)
            pass
        elif both:
            limit_desc = "فوق الحد و تحت الحد"
            limit_filter = ""  # No filter - show all
        else:
            limit_desc = "فوق الحد" if is_above else "تحت الحد"
            limit_filter = f"AND is_above_limit = {1 if is_above else 0}"
        
        # Smart pesticide matching logic to handle typos
        pesticide_lower = pesticide.lower()
        fuzzy_condition = ""
        
        if len(pesticide_lower) > 4:
            # Match first 4 chars to catch typos like 'buprofuzin' vs 'buprofezin'
            prefix = pesticide_lower[:4]
            fuzzy_condition = f"OR LOWER(pesticide_name) LIKE '%{prefix}%'"
        
        sql = f"""
        SELECT 
            "كود العينة" as sample_code,
            "اسم العينة" as sample_name,
            pesticide_name as اسم_المبيد,
            concentration as التركيز,
            limit_value as الحد_المسموح,
            is_above_limit as فوق_الحد,
            sample_result as نتيجة_العينة,
            "الحى" as neighborhood
        FROM chemistry_tidy
        WHERE {sample_filter}
        AND pesticide_name IN ({', '.join([f"'{v}'" for v in get_pesticide_variants(pesticide)])})
        AND is_detected = 1
        {limit_filter}
        ORDER BY is_above_limit DESC, concentration DESC
        LIMIT 50
        """
        
        try:
             df = con.execute(sql).df()
        except Exception:
             # Fallback query
             sql = f"""
            SELECT 
                "كود العينة" as sample_code,
                "اسم العينة" as sample_name,
                pesticide_name,
                concentration,
                limit_value,
                is_above_limit,
                "الحى" as neighborhood
            FROM chemistry_tidy
            WHERE {sample_filter}
            AND pesticide_name IN ({', '.join([f"'{v}'" for v in get_pesticide_variants(pesticide)])})
            AND is_detected = 1
            {limit_filter}
            ORDER BY is_above_limit DESC, concentration DESC
            LIMIT 50
            """
             df = con.execute(sql).df()
            
        con.close()
        
        sample_display = ' + '.join(samples)
        
        if not df.empty:
            count = len(df)
            
            if both:
                # Count above and below - handle both column name possibilities
                above_col = 'فوق_الحد' if 'فوق_الحد' in df.columns else 'is_above_limit'
                above_count = len(df[df[above_col] == 1]) if above_col in df.columns else 0
                below_count = len(df[df[above_col] == 0]) if above_col in df.columns else 0
                response = f"📊 **عينات {sample_display} التي تحتوي على مبيد '{pesticide}' ({limit_desc}):**\n\n"
                response += f"✅ وجدت **{count}** عينة: {above_count} فوق الحد، {below_count} تحت الحد\n\n"
            else:
                response = f"📊 **عينات {sample_display} التي تحتوي على مبيد '{pesticide}' {limit_desc}:**\n\n"
                response += f"✅ وجدت **{count}** عينة\n\n"
            
            response += df.to_markdown(index=False)
        else:
            response = f"⚠️ لم أجد عينات {sample_display} تحتوي على مبيد '{pesticide}' {limit_desc}"
            
        return response, df
    
    def _handle_simple_sample_count(self, samples: List[str], 
                                     neighborhoods: List[str]) -> Tuple[str, pd.DataFrame]:
        """
        عد العينات الفريدة (بدون شروط)
        Simple sample count without limit/compliance conditions
        "ما عدد عينات الطماطم و الخيار و الكوسة"
        """
        con = self._get_connection()
        
        # Build sample filter
        sample_conditions = [f"\"اسم العينة\" LIKE '%{s}%'" for s in samples]
        sample_filter = f"({' OR '.join(sample_conditions)})"
        
        # Build neighborhood filter
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
        
        # Count unique samples by sample code
        sql = f"""
        WITH sample_info AS (
            SELECT 
                "كود العينة" as sample_code,
                "اسم العينة" as sample_name,
                MAX(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as has_violation,
                COUNT(*) as pesticide_count
            FROM chemistry_tidy
            WHERE {sample_filter}
            {neighborhood_filter}
            GROUP BY "كود العينة", "اسم العينة"
        )
        SELECT 
            sample_name as نوع_العينة,
            COUNT(*) as عدد_العينات_الفريدة,
            SUM(has_violation) as عينات_متجاوزة,
            SUM(CASE WHEN has_violation = 0 THEN 1 ELSE 0 END) as عينات_سليمة
        FROM sample_info
        GROUP BY sample_name
        ORDER BY عدد_العينات_الفريدة DESC
        """
        
        df = con.execute(sql).df()
        con.close()
        
        sample_display = ' + '.join(samples)
        location_desc = f" في {' + '.join(neighborhoods)}" if neighborhoods else ""
        
        if not df.empty:
            total = int(df['عدد_العينات_الفريدة'].sum())
            total_violated = int(df['عينات_متجاوزة'].sum())
            total_ok = int(df['عينات_سليمة'].sum())
            
            response = f"📊 **عدد عينات {sample_display}{location_desc}:**\n\n"
            response += f"✅ إجمالي العينات الفريدة: **{total}** عينة\n"
            response += f"🔴 عينات متجاوزة: **{total_violated}**\n"
            response += f"🟢 عينات سليمة: **{total_ok}**\n\n"
            response += df.to_markdown(index=False)
        else:
            response = f"⚠️ لم أجد عينات {sample_display}"
        
        return response, df
    
    def _handle_llm_query(self, query: str, samples: List[str], 
                           neighborhoods: List[str], pesticide: Optional[str]) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        LLM Fallback - Generate SQL using LLM
        Used when no pattern matches
        """
        if self.llm_client is None:
            return None, None
        
        try:
            # Build context
            schema_context = f"""
Available columns in chemistry_tidy table:
{', '.join(self.schema_columns[:30])}

Available sample types: {', '.join(self.available_sample_types[:15])}
Available neighborhoods: {', '.join(self.available_neighborhoods[:15])}

Key columns:
- "كود العينة": Sample code (unique identifier)
- "اسم العينة": Sample name/type
- "الحى": Neighborhood
- pesticide_name: Pesticide name (English)
- concentration: Pesticide concentration
- limit_value: MRL limit value
- is_detected: 1 if detected, 0 otherwise
- is_above_limit: 1 if concentration > limit
- sample_result: 'compliant (مطابقة)' or 'non-compliant (غير مطابقة)'
"""
            
            # Build prompt
            prompt = f"""You are a SQL expert for a pesticide analysis database.
Generate a DuckDB SQL query for this question:

Question: {query}

{schema_context}

Detected entities:
- Sample types: {samples if samples else 'None'}
- Neighborhoods: {neighborhoods if neighborhoods else 'None'}
- Pesticide: {pesticide if pesticide else 'None'}

Rules:
1. Use COUNT(DISTINCT "كود العينة") for counting unique samples
2. Always quote Arabic column names with double quotes
3. Return only the SQL query, no explanation
4. Limit results to 100 rows
5. Use ILIKE for case-insensitive matching

SQL:"""
            
            # Call LLM
            response = self.llm_client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}
            )
            
            sql = response['message']['content'].strip()
            
            # Clean up SQL
            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            # Validate SQL (basic check)
            if not sql.upper().startswith('SELECT'):
                return None, None
            
            # Execute SQL
            con = self._get_connection()
            df = con.execute(sql).df()
            con.close()
            
            if not df.empty:
                response_text = f"🤖 **نتيجة من الذكاء الاصطناعي:**\n\n"
                response_text += f"📊 **{query}**\n\n"
                response_text += df.to_markdown(index=False)
                return response_text, df
            else:
                return f"🤖 الاستعلام لم يُرجع نتائج.\n\nSQL: `{sql[:200]}...`", None
                
        except Exception as e:
            print(f"⚠️ LLM query failed: {e}")
            return None, None
    
    def _handle_unknown_query(self, query: str) -> str:
        """معالجة الاستعلامات غير المعروفة"""
        response = "⚠️ **عذراً، لم أتمكن من فهم سؤالك بشكل كامل.**\n\n"
        response += "📝 **أمثلة على الأسئلة المدعومة:**\n"
        response += "• ما عدد عينات الطماطم والخيار والكوسة؟\n"
        response += "• ما عدد عينات الطماطم فوق الحد؟\n"
        response += "• ماهي العينات غير المطابقة؟\n"
        response += "• ماهي المبيدات في الخيار؟\n"
        response += "• المبيدات في حي الإسكان والنهضة\n"
        response += "• ترتيب الأحياء حسب المخالفات\n"
        response += "• تحليل شامل للطماطم\n"
        response += "• ماهو اعلي تركيز لمبيد الايميداكلوبريد؟\n"
        response += "• ابحث عن الفيبرونيل في الفاصوليا\n"
        response += "• عينات تحتوي على 6 مبيدات\n"
        return response


# Create singleton instance
_engine = None

def get_query_engine(db_path: str = None, enable_llm: bool = True) -> CoreQueryEngine:
    """
    Get query engine singleton
    
    Args:
        db_path: Database path
        enable_llm: Enable LLM fallback for unknown queries
    """
    global _engine
    if _engine is None:
        _engine = CoreQueryEngine(db_path, enable_llm_fallback=enable_llm)
    return _engine
