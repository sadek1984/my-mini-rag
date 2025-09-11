# engines/iso_hierarchical_search.py
class ISO17025HierarchicalSearch:
    """محرك بحث يفهم التسلسل الهرمي"""
    
    def __init__(self, nlp_controller):
        self.nlp_controller = nlp_controller
        self.hierarchy_weights = {
            1: 1.0,    # Quality Manual - أعلى وزن
            2: 0.8,    # Procedures  
            3: 0.6,    # Work Instructions
            4: 0.4     # Forms/Records
        }
    
    async def hierarchical_search(self, query: str, project_id: str) -> Dict:
        """بحث هرمي ذكي"""
        
        # 1. تحليل نوع الاستفسار
        query_analysis = self._analyze_query_intent(query)
        
        # 2. تحديد المستويات ذات الصلة
        relevant_levels = self._determine_relevant_levels(query_analysis)
        
        # 3. البحث في كل مستوى مع الأوزان
        level_results = {}
        for level in relevant_levels:
            results = await self._search_in_level(
                query, project_id, level, query_analysis
            )
            level_results[level] = results
        
        # 4. دمج النتائج هرمياً
        final_answer = self._merge_hierarchical_results(
            level_results, query_analysis
        )
        
        return final_answer
    
    def _analyze_query_intent(self, query: str) -> Dict:
        """تحليل نية الاستفسار"""
        intent_patterns = {
            "policy_inquiry": [
                "سياسة", "policy", "إدارة", "management", "استراتيجية"
            ],
            "procedure_inquiry": [
                "إجراء", "procedure", "خطوات", "steps", "عملية", "process"
            ],
            "instruction_inquiry": [
                "كيف", "how", "تعليمات", "instruction", "طريقة", "method"
            ],
            "compliance_inquiry": [
                "متطلبات", "requirements", "امتثال", "compliance", "معيار"
            ],
            "evidence_inquiry": [
                "نموذج", "form", "سجل", "record", "إثبات", "evidence"
            ]
        }
        
        query_lower = query.lower()
        detected_intents = []
        
        for intent, keywords in intent_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        return {
            "intents": detected_intents,
            "primary_intent": detected_intents[0] if detected_intents else "general",
            "complexity": "complex" if len(detected_intents) > 1 else "simple"
        }
    
    def _determine_relevant_levels(self, query_analysis: Dict) -> List[int]:
        """تحديد المستويات ذات الصلة بناءً على نوع الاستفسار"""
        intent_to_levels = {
            "policy_inquiry": [1, 2],        # Quality Manual + Procedures
            "procedure_inquiry": [2, 3],     # Procedures + Work Instructions  
            "instruction_inquiry": [3, 4],   # Work Instructions + Forms
            "compliance_inquiry": [1, 2, 3], # جميع المستويات التشغيلية
            "evidence_inquiry": [4, 3],      # Forms + Work Instructions
            "general": [1, 2, 3, 4]          # جميع المستويات
        }
        
        primary_intent = query_analysis["primary_intent"]
        return intent_to_levels.get(primary_intent, [1, 2, 3, 4])