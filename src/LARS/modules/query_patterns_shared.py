"""
query_patterns_shared.py
========================
قاعدة بيانات مشتركة للأنماط والكيانات بين النظام الصوتي والنصي
يتم التحديث في مكان واحد وتطبيقه على كلا النظامين
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class ExtractedEntities:
    """الكيانات المستخرجة من الاستعلام"""
    sample_types: List[str] = field(default_factory=list)
    pesticides: List[str] = field(default_factory=list)
    neighborhoods: List[str] = field(default_factory=list)
    receivers: List[str] = field(default_factory=list)
    facilities: List[str] = field(default_factory=list)
    limit_condition: Optional[str] = None  # 'above', 'below', 'compliant', 'non-compliant'
    pesticide_count: Optional[int] = None
    aggregation_type: Optional[str] = None  # 'count', 'search', 'stats', 'classification'
    needs_classification: bool = False
    unique_samples: bool = False
    violation_stats: bool = False


class SharedPatterns:
    """قاعدة البيانات المشتركة للأنماط"""
    
    # أنواع العينات
    SAMPLE_TYPES = {
        'خضروات': ['طماطم', 'خيار', 'كوسة', 'فاصوليا', 'فلفل', 'باذنجان',
                   'بصل', 'ثوم', 'جزر', 'خس', 'سبانخ', 'ملفوف', 'قرنبيط',
                   'بروكلي', 'فجل', 'لفت', 'بقدونس', 'كزبرة', 'نعناع',
                   'شبت', 'ريحان', 'جرجير', 'ملوخية', 'بامية'],
        'فواكه': ['تفاح', 'برتقال', 'موز', 'عنب', 'فراولة', 'مانجو',
                  'بطيخ', 'شمام', 'خوخ', 'كمثرى', 'رمان', 'تين', 'مشمش'],
        'توابل': ['كمون', 'كراوية', 'هيل', 'زعفران', 'كركم', 'فلفل اسود',
                  'قرفة', 'زنجبيل', 'حبة البركة', 'يانسون', 'شمر'],
        'مكسرات': ['فستق', 'لوز', 'كاجو', 'عين جمل', 'بندق', 'فول سوداني'],
        'حبوب': ['قمح', 'شعير', 'أرز', 'ذرة', 'عدس', 'حمص', 'فول']
    }
    
    # خرائط المبيدات (عربي ←→ إنجليزي)
    PESTICIDE_MAP = {
        'عربي_إنجليزي': {
            'بايفنثرين': 'bifenthrin', 'بيفنثرين': 'bifenthrin', 'البايفنثرن': 'bifenthrin',
            'بابروفيزن': 'buprofezin', 'بوبروفيزين': 'buprofezin', 'البوبروفيزين': 'buprofezin',
            'إيميداكلوبريد': 'imidacloprid', 'ايميداكلوبريد': 'imidacloprid', 'الايميداكلوبريد': 'imidacloprid',
            'كلوربيريفوس': 'chlorpyrifos', 'الكلوربيريفوس': 'chlorpyrifos',
            'فيبرونيل': 'fipronil', 'الفبرونيل': 'fipronil', 'فبرونيل': 'fipronil',
            'دلتامثرين': 'deltamethrin', 'الدلتامثرن': 'deltamethrin',
            'سيبرمثرين': 'cypermethrin', 'السايبرمثرن': 'cypermethrin',
            'كاربندازيم': 'carbendazim', 'الكاربندازيم': 'carbendazim',
            'أبامكتين': 'abamectin', 'الابامكتين': 'abamectin', 'ابامكتين': 'abamectin',
            'ثياميثوكسام': 'thiamethoxam', 'الثياميثوكسام': 'thiamethoxam',
            'أسيتاميبريد': 'acetamiprid', 'الاسيتامبريد': 'acetamiprid', 'اسيتامبريد': 'acetamiprid',
            'بروفينوفوس': 'profenofos', 'البروفينوفوس': 'profenofos',
            'ميتالاكسيل': 'metalaxyl', 'الميتالاكسيل': 'metalaxyl',
            'إيثيون': 'ethion', 'الايثيون': 'ethion', 'ايثيون': 'ethion',
            'لامدا': 'lambda-cyhalothrin', 'لامدا سايبرمثرين': 'lambda-cyhalothrin',
            'ديميثويت': 'dimethoate', 'الديميثويت': 'dimethoate',
            'ميثوميل': 'methomyl', 'الميثوميل': 'methomyl',
            'ازوكسيستروبين': 'azoxystrobin', 'الازوكسيستروبين': 'azoxystrobin',
            'سبينوساد': 'spinosad', 'السبينوساد': 'spinosad',
            'ملاثيون': 'malathion', 'الملاثيون': 'malathion',
            'بيرمثرين': 'permethrin', 'البيرمثرين': 'permethrin',
            'كلوروثالونيل': 'chlorothalonil', 'الكلوروثالونيل': 'chlorothalonil',
            'بايفينازيت': 'bifenazate', 'البايفينازيت': 'bifenazate',
        },
        'إنجليزي_عربي': {
            'bifenthrin': 'بايفنثرين',
            'buprofezin': 'بوبروفيزين',
            'imidacloprid': 'ايميداكلوبريد',
            'chlorpyrifos': 'كلوربيريفوس',
            'fipronil': 'فيبرونيل',
            'deltamethrin': 'دلتامثرين',
            'cypermethrin': 'سايبرمثرين',
            'carbendazim': 'كاربندازيم',
            'abamectin': 'ابامكتين',
            'thiamethoxam': 'ثياميثوكسام',
            'acetamiprid': 'اسيتامبريد',
            'profenofos': 'بروفينوفوس',
            'metalaxyl': 'ميتالاكسيل',
            'ethion': 'ايثيون',
            'lambda-cyhalothrin': 'لامدا سايبرمثرين',
            'dimethoate': 'ديميثويت',
            'methomyl': 'ميثوميل',
            'azoxystrobin': 'ازوكسيستروبين',
            'spinosad': 'سبينوساد',
            'malathion': 'ملاثيون',
            'permethrin': 'بيرمثرين',
            'chlorothalonil': 'كلوروثالونيل',
            'bifenazate': 'بايفينازيت',
        }
    }
    
    # الأحياء
    NEIGHBORHOODS = [
        'النهضة', 'النهضه', 'الإسكان', 'الاسكان', 'الريان',
        'الأخضر', 'الاخضر', 'الروضة', 'الروضه', 'النخيل',
        'الفلاح', 'المنتزه', 'الجردة', 'الجرده', 'الأفق',
        'الافق', 'الضاحي', 'السالمية', 'السالميه', 'الفايزية',
        'الفايزيه', 'الشقة', 'الشقه', 'المنار', 'الصفراء',
        'الموطأ', 'المرزوق', 'الخبيب', 'العزيزية', 'الشفا',
        'الربوة', 'الربوه', 'البديعة', 'البديعه', 'الملز',
        'السليمانية', 'السليمانيه', 'العليا', 'المروج'
    ]
    
    # تصحيحات STT المشتركة
    STT_CORRECTIONS = {
        'البير فزن': 'البوبروفيزين',
        'بير فيزين': 'البوبروفيزين',
        'بروفيزين': 'البوبروفيزين',
        'بوبروفزين': 'البوبروفيزين',
        'اميدا كلوبريد': 'الايميداكلوبريد',
        'ايميدا كلوبريد': 'الايميداكلوبريد',
        'اميداكلو بريد': 'الايميداكلوبريد',
        'الفيب رونيل': 'الفبرونيل',
        'في برونيل': 'الفبرونيل',
        'فايبرونيل': 'الفبرونيل',
        'كلور بيريفوس': 'الكلوربيريفوس',
        'كلوربيري فوس': 'الكلوربيريفوس',
        'ابا مكتين': 'الابامكتين',
        'ابامك تين': 'الابامكتين',
        'دلتا مثرين': 'الدلتامثرن',
        'سايبر مثرين': 'السايبرمثرن',
        'ثيا ميثوكسام': 'الثياميثوكسام',
        'كاربن دازيم': 'الكاربندازيم',
        'ملا ثيون': 'الملاثيون',
        'بروفينو فوس': 'البروفينوفوس',
        'ازوكسي ستروبين': 'الازوكسيستروبين',
        'الفصولية': 'الفاصوليا',
        'فصوليا': 'الفاصوليا',
        'فصولية': 'الفاصوليا',
        'طماطه': 'طماطم',
        'الطماطة': 'الطماطم',
        'الخيارات': 'الخيار',
        'الخضروات': 'الخضار',
        'التوابل': 'توابل',
        'مكسرات': 'مكسرات',
        'المكسرات': 'مكسرات',
    }


class EntityExtractor:
    """مستخرج الكيانات من الاستعلامات"""
    
    @staticmethod
    def extract_all(query: str) -> ExtractedEntities:
        """استخراج كل الكيانات من الاستعلام"""
        entities = ExtractedEntities()
        
        # استخراج أنواع العينات
        entities.sample_types = EntityExtractor.extract_sample_types(query)
        
        # استخراج المبيدات
        entities.pesticides = EntityExtractor.extract_pesticides(query)
        
        # استخراج الأحياء
        entities.neighborhoods = EntityExtractor.extract_neighborhoods(query)
        
        # استخراج المستلمين
        entities.receivers = EntityExtractor.extract_receivers(query)
        
        # استخراج المنشآت
        entities.facilities = EntityExtractor.extract_facilities(query)
        
        # تحديد شرط الحد
        entities.limit_condition = EntityExtractor.extract_limit_condition(query)
        
        # استخراج عدد المبيدات
        entities.pesticide_count = EntityExtractor.extract_pesticide_count(query)
        
        # تحديد نوع التجميع
        entities.aggregation_type = EntityExtractor.extract_aggregation_type(query)
        
        # تحديد إذا كان يحتاج تصنيف
        entities.needs_classification = EntityExtractor.check_classification(query)
        
        # تحديد إذا كان طلب عينات فريدة
        entities.unique_samples = EntityExtractor.check_unique_samples(query)
        
        # تحديد إذا كان طلب إحصائيات مخالفات
        entities.violation_stats = EntityExtractor.check_violation_stats(query)
        
        return entities
    
    @staticmethod
    def extract_sample_types(query: str) -> List[str]:
        """استخراج أنواع العينات"""
        found_types = []
        
        for category, types in SharedPatterns.SAMPLE_TYPES.items():
            for sample_type in types:
                if sample_type in query:
                    found_types.append(sample_type)
        
        return list(set(found_types))
    
    @staticmethod
    def extract_pesticides(query: str) -> List[str]:
        """استخراج المبيدات (تعود بالإنجليزية)"""
        found_pesticides = []
        
        for ar_name, en_name in SharedPatterns.PESTICIDE_MAP['عربي_إنجليزي'].items():
            if ar_name in query:
                found_pesticides.append(en_name)
        
        # البحث عن أسماء إنجليزية مباشرة
        for en_name in SharedPatterns.PESTICIDE_MAP['إنجليزي_عربي'].keys():
            if en_name.lower() in query.lower():
                found_pesticides.append(en_name)
        
        return list(set(found_pesticides))
    
    @staticmethod
    def extract_neighborhoods(query: str) -> List[str]:
        """استخراج الأحياء"""
        found_neighborhoods = []
        
        for neighborhood in SharedPatterns.NEIGHBORHOODS:
            if neighborhood in query:
                found_neighborhoods.append(neighborhood)
        
        return list(set(found_neighborhoods))
    
    @staticmethod
    def extract_receivers(query: str) -> List[str]:
        """استخراج أسماء المستلمين"""
        receivers = []
        
        receiver_patterns = [
            r'للمستلم\s+(.+?)(?:\s|$)',
            r'المستلم\s+(.+?)(?:\s|$)',
            r'receiver\s+(.+?)(?:\s|$)',
        ]
        
        for pattern in receiver_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            receivers.extend(matches)
        
        return receivers
    
    @staticmethod
    def extract_facilities(query: str) -> List[str]:
        """استخراج أسماء المنشآت"""
        facilities = []
        
        facility_patterns = [
            r'في منشأة\s+(.+?)(?:\s|$)',
            r'لمنشأة\s+(.+?)(?:\s|$)',
            r'منشأة\s+(.+?)(?:\s|$)',
            r'facility\s+(.+?)(?:\s|$)',
        ]
        
        for pattern in facility_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            facilities.extend(matches)
        
        return facilities
    
    @staticmethod
    def extract_limit_condition(query: str) -> Optional[str]:
        """استخراج شرط الحد"""
        query_lower = query.lower()
        
        if any(phrase in query_lower or phrase in query for phrase in 
               ['فوق الحد', 'above limit', 'exceed', 'تجاوز', 'مخالف', 'راسب', 'غير مطابق']):
            return 'above'
        elif any(phrase in query_lower or phrase in query for phrase in
                 ['تحت الحد', 'below limit', 'within limit', 'ناجح', 'سليم', 'مطابق']):
            return 'below'
        
        return None
    
    @staticmethod
    def extract_pesticide_count(query: str) -> Optional[int]:
        """استخراج عدد المبيدات المطلوب"""
        # الأرقام العربية
        arabic_nums = {'٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
                      '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'}
        
        converted_query = query
        for ar, en in arabic_nums.items():
            converted_query = converted_query.replace(ar, en)
        
        count_patterns = [
            r'(\d+)\s*مبيد',
            r'مبيد\s*(\d+)',
            r'(\d+)\s*pesticide',
            r'pesticide\s*(\d+)',
            r'تحتوي\s+على\s+(\d+)',
            r'تحتوي\s+علي\s+(\d+)',
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, converted_query, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except:
                    continue
        
        return None
    
    @staticmethod
    def extract_aggregation_type(query: str) -> Optional[str]:
        """تحديد نوع التجميع"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['عدد', 'count', 'كم', 'how many', 'إجمالي', 'total']):
            return 'count'
        elif any(word in query_lower for word in ['ابحث', 'find', 'search', 'أعطني', 'أظهر', 'اعرض']):
            return 'search'
        elif any(word in query_lower for word in ['متوسط', 'وسيط', 'أعلى', 'أقل', 'avg', 'median', 'min', 'max']):
            return 'stats'
        elif any(word in query_lower for word in ['صنف', 'تصنيف', 'classify', 'مجموعات']):
            return 'classification'
        
        return 'search'  # الافتراضي
    
    @staticmethod
    def check_classification(query: str) -> bool:
        """التحقق إذا كان الاستعلام يحتاج تصنيف كيميائي"""
        classification_keywords = ['صنف', 'تصنيف', 'classify', 'classification', 'مجموعات', 'groups']
        return any(kw in query.lower() for kw in classification_keywords)
    
    @staticmethod
    def check_unique_samples(query: str) -> bool:
        """التحقق إذا كان طلب عينات فريدة"""
        unique_keywords = ['فريدة', 'unique', 'مختلفة', 'distinct', 'مميزة']
        return any(kw in query.lower() for kw in unique_keywords)
    
    @staticmethod
    def check_violation_stats(query: str) -> bool:
        """التحقق إذا كان طلب إحصائيات مخالفات"""
        violation_keywords = ['مخالفات', 'violations', 'راسب', 'تجاوز', 'مخالف', 'non-compliant']
        return any(kw in query.lower() for kw in violation_keywords)


def correct_stt_text(text: str) -> str:
    """تصحيح أخطاء STT باستخدام القاموس المشترك"""
    corrected = text
    
    for wrong, correct in SharedPatterns.STT_CORRECTIONS.items():
        if wrong in corrected:
            corrected = corrected.replace(wrong, correct)
    
    return corrected


# للاختبار
if __name__ == "__main__":
    test_queries = [
        "ما عدد عينات الطماطم",
        "ابحث عن الفيبرونيل في الخيار",
        "ما هي العينات التي تحتوي على 10 مبيدات",
        "عينات غير مطابقة في حي النهضة",
        "صنف المبيدات في الفاصوليا",
    ]
    
    print("=" * 60)
    print("اختبار استخراج الكيانات")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📝 الاستعلام: {query}")
        entities = EntityExtractor.extract_all(query)
        print(f"   - أنواع العينات: {entities.sample_types}")
        print(f"   - المبيدات: {entities.pesticides}")
        print(f"   - الأحياء: {entities.neighborhoods}")
        print(f"   - شرط الحد: {entities.limit_condition}")
        print(f"   - عدد المبيدات: {entities.pesticide_count}")
        print(f"   - نوع التجميع: {entities.aggregation_type}")
