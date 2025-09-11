from typing import Dict, Any
# generators/iso_hierarchical_answer.py
class ISO17025AnswerGenerator:
    """مولد إجابات يفهم السياق الهرمي"""
    
    def generate_hierarchical_answer(self, results: Dict, query: str) -> str:
        """توليد إجابة هرمية شاملة"""
        
        answer_structure = {
            "المستوى": self._determine_primary_level(results),
            "النص المسترجع": self._extract_core_information(results),
            "العلاقة الهرمية": self._build_hierarchical_context(results),
            "السبب والسياق": self._explain_iso_context(results),
            "المتطلبات ذات الصلة": self._identify_related_requirements(results),
            "الخطوات التالية": self._suggest_next_steps(results, query)
        }
        
        return self._format_hierarchical_response(answer_structure)
    
    def _build_hierarchical_context(self, results: Dict) -> str:
        """بناء السياق الهرمي"""
        context_lines = []
        
        # ترتيب النتائج حسب المستوى
        sorted_levels = sorted(results.keys())
        
        for level in sorted_levels:
            level_info = self.get_level_description(level)
            if results[level]:
                context_lines.append(
                    f"**{level_info['name']}**: {level_info['role']}"
                )
                
                # إضافة الروابط
                if level > 1:
                    parent_level = level - 1
                    if parent_level in results:
                        context_lines.append(
                            f"  ↳ مرتبط بـ {self.get_level_description(parent_level)['name']}"
                        )
        
        return "\n".join(context_lines)
    
    def get_level_description(self, level: int) -> Dict:
        """وصف المستوى"""
        descriptions = {
            1: {"name": "دليل الجودة", "role": "يحدد السياسات العليا والإطار العام"},
            2: {"name": "الإجراءات", "role": "تطبق السياسات في خطوات عملية"},  
            3: {"name": "تعليمات العمل", "role": "تفصل الخطوات التنفيذية بدقة"},
            4: {"name": "النماذج والسجلات", "role": "توثق التنفيذ الفعلي"}
        }
        return descriptions.get(level, {"name": "غير محدد", "role": ""})