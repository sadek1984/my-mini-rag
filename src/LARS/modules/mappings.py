"""
LARS Centralized Mappings Module.

Single source of truth for all Arabic/English pesticide, sample,
and STT correction dictionaries used across the application.

Why this module exists:
    Before this refactor, identical dictionaries were duplicated in
    voice_lars.py, voice_bot_unified.py, unified_query_processor.py,
    ai_assistant.py, and core_query_engine.py. Any update required
    editing 4-5 files — a major DRY violation and bug magnet.

Usage:
    from modules.mappings import (
        PESTICIDE_AR_TO_EN,
        SAMPLE_CORRECTIONS,
        STT_CORRECTIONS,
        correct_stt_text,
    )
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import Dict, List, Optional


# ============================================================================
# PESTICIDE MAPPINGS: Arabic → English
# ============================================================================
# Canonical mapping. Every Arabic spelling variant maps to ONE English name.
# Merged from voice_lars.py, ai_assistant.py, core_query_engine.py,
# and unified_query_processor.py — deduplicated & alphabetised.

PESTICIDE_AR_TO_EN: Dict[str, str] = {
    # Abamectin
    "ابامكتين": "abamectin",
    "الابامكتين": "abamectin",
    # Acetamiprid
    "اسيتامبريد": "acetamiprid",
    "الاسيتامبريد": "acetamiprid",
    "اسيتاميبريد": "acetamiprid",
    "الاسيتاميبريد": "acetamiprid",
    # Azoxystrobin
    "ازوكسيستروبين": "azoxystrobin",
    "الازوكسيستروبين": "azoxystrobin",
    # Bifenthrin
    "بايفنثرن": "bifenthrin",
    "بايفنثرين": "bifenthrin",
    "البايفنثرن": "bifenthrin",
    # Buprofezin
    "بوبروفيزين": "buprofezin",
    "بابروفيزن": "buprofezin",
    "البابروفيزن": "buprofezin",
    "البوبروفيزين": "buprofezin",
    "بروفيزين": "buprofezin",
    # Carbendazim
    "كاربندازيم": "carbendazim",
    "الكاربندازيم": "carbendazim",
    "الكاربندزيم": "carbendazim",
    # Chlorantraniliprole
    "كلورانترانيليبرول": "chlorantraniliprole",
    # Chlorpyrifos
    "كلوربيريفوس": "chlorpyrifos",
    "الكلوربيريفوس": "chlorpyrifos",
    # Cypermethrin
    "سايبرمثرن": "cypermethrin",
    "سايبرمثرين": "cypermethrin",
    "السايبرمثرن": "cypermethrin",
    "السايبرمثرين": "cypermethrin",
    "السيبرمثرين": "cypermethrin",
    # Deltamethrin
    "دلتامثرن": "deltamethrin",
    "دلتامثرين": "deltamethrin",
    "الدلتامثرن": "deltamethrin",
    "الدلتامثرين": "deltamethrin",
    "الدلتاميثرين": "deltamethrin",
    # Dimethoate
    "الديمثويت": "dimethoate",
    # Emamectin
    "الإيماميكتين": "emamectin",
    # Fipronil
    "فبرونيل": "fipronil",
    "فيبرونيل": "fipronil",
    "الفبرونيل": "fipronil",
    "الفيبرونيل": "fipronil",
    # Imidacloprid
    "ايميداكلوبريد": "imidacloprid",
    "اميداكلوبريد": "imidacloprid",
    "الايميداكلوبريد": "imidacloprid",
    "الأيميداكلوبريد": "imidacloprid",
    "إيميداكلوبريد": "imidacloprid",
    "ايميداكلوبرايد": "imidacloprid",
    # Lambda-cyhalothrin
    "اللامبدا": "lambda-cyhalothrin",
    # Malathion
    "ملاثيون": "malathion",
    "الملاثيون": "malathion",
    # Metalaxyl
    "ميتالاكسيل": "metalaxyl",
    "الميتالاكسيل": "metalaxyl",
    # Methomyl
    "الميثوميل": "methomyl",
    # Profenofos
    "بروفينوفوس": "profenofos",
    "البروفينوفوس": "profenofos",
    # Spinosad
    "سبينوساد": "spinosad",
    "السبينوساد": "spinosad",
    # Thiamethoxam
    "ثياميثوكسام": "thiamethoxam",
    "الثياميثوكسام": "thiamethoxam",
    "الثيامثوكسام": "thiamethoxam",
    # Ethion (from core_query_engine)
    "ايثيون": "ethion",
    "الايثيون": "ethion",
    "الايثون": "ethion",
    # Lambda-cyhalothrin (expanded)
    "لامدا سيهالوثرين": "lambda-cyhalothrin",
    "لامدا": "lambda-cyhalothrin",
    "اللامبدا": "lambda-cyhalothrin",
    # Dimethoate (expanded)
    "ديميثويت": "dimethoate",
    "الديميثويت": "dimethoate",
    "الديمثويت": "dimethoate",
    # Methomyl
    "ميثوميل": "methomyl",
    "الميثوميل": "methomyl",
    # Propiconazole
    "بروكونازول": "propiconazole",
    "البروكونازول": "propiconazole",
    # Tebuconazole
    "تيبوكونازول": "tebuconazole",
    "التيبوكونازول": "tebuconazole",
    # Emamectin
    "الإيماميكتين": "emamectin",
}

# Reverse lookup: English → list of Arabic names (auto-generated)
PESTICIDE_EN_TO_AR: Dict[str, List[str]] = {}
for _ar, _en in PESTICIDE_AR_TO_EN.items():
    PESTICIDE_EN_TO_AR.setdefault(_en, []).append(_ar)


# ============================================================================
# KNOWN ARABIC PESTICIDE NAMES (for fuzzy STT matching)
# ============================================================================

KNOWN_PESTICIDES_AR: List[str] = sorted(set(PESTICIDE_AR_TO_EN.keys()))


# ============================================================================
# SAMPLE / COMMODITY CORRECTIONS: variant → canonical Arabic
# ============================================================================
# Merged from voice_lars.py and unified_query_processor.py.

SAMPLE_CORRECTIONS: Dict[str, str] = {
    # ── Vegetables ──
    "الطماطم": "طماطم", "طماطة": "طماطم", "طماطه": "طماطم", "بندورة": "طماطم",
    "الخيار": "خيار", "قثاء": "خيار",
    "الكوسة": "كوسة", "كوسا": "كوسة", "الكوسا": "كوسة",
    "الفلفل": "فلفل", "فليفلة": "فلفل",
    "الباذنجان": "باذنجان", "بتنجان": "باذنجان",
    "الفصولية": "فاصوليا", "فصوليا": "فاصوليا", "فصولية": "فاصوليا",
    "البامية": "بامية",
    # ── Root & bulb ──
    "البصل": "بصل", "الثوم": "ثوم",
    "الجزر": "جزر", "البطاطس": "بطاطس", "بطاطا": "بطاطس",
    # ── Leafy greens ──
    "الخس": "خس", "البقدونس": "بقدونس", "السبانخ": "سبانخ",
    "الجرجير": "جرجير", "الكزبرة": "كزبرة", "الشبت": "شبت",
    "الملوخية": "ملوخية", "النعناع": "نعناع",
    "الملفوف": "ملفوف", "الكرنب": "ملفوف",
    "الزهرة": "زهرة", "البروكلي": "بروكلي",
    # ── Fruits ──
    "البرتقال": "برتقال", "التفاح": "تفاح", "العنب": "عنب",
    "الموز": "موز", "المانجو": "مانجو", "الفراولة": "فراولة",
    "الرمان": "رمان", "الكمثرى": "كمثرى", "الليمون": "ليمون",
    "التوت": "توت", "التمر": "تمر",
    # ── Grains ──
    "القمح": "قمح", "الرز": "رز", "أرز": "رز",
    "الذرة": "ذرة", "العدس": "عدس",
    "الدقيق": "دقيق", "طحين": "دقيق", "الشوفان": "شوفان",
    # ── Spices ──
    "الهيل": "هيل", "حبهان": "هيل",
    "الكمون": "كمون", "الزعتر": "زعتر",
    "التوابل": "توابل", "بهارات": "توابل", "البهارات": "بهارات",
    "القرنفل": "قرنفل", "اليانسون": "يانسون",
    "الشمر": "شمر", "المحلب": "محلب",
    # ── Nuts ──
    "الفستق": "فستق", "المكسرات": "مكسرات",
    "اللوز": "لوز", "الكاجو": "كاجو", "البندق": "بندق",
    "البيكان": "بيكان", "السمسم": "سمسم",
}

# Arabic → English sample mapping (for queries & display)
SAMPLE_AR_TO_EN: Dict[str, str] = {
    "طماطم": "tomato", "الطماطم": "tomato",
    "خيار": "cucumber", "الخيار": "cucumber",
    "كوسة": "zucchini", "كوسا": "zucchini",
    "فلفل": "pepper", "الفلفل": "pepper",
    "باذنجان": "eggplant", "الباذنجان": "eggplant",
    "فاصوليا": "beans", "الفاصوليا": "beans",
    "هيل": "cardamom", "الهيل": "cardamom",
    "فستق": "pistachio", "الفستق": "pistachio",
    "توابل": "spices", "التوابل": "spices",
    "مكسرات": "nuts", "المكسرات": "nuts",
}

# English → Arabic sample (reverse)
SAMPLE_EN_TO_AR: Dict[str, str] = {
    # ── Vegetables ──
    "tomato": "طماطم", "tomatoes": "طماطم",
    "cucumber": "خيار", "cucumbers": "خيار",
    "zucchini": "كوسة", "squash": "كوسة",
    "pepper": "فلفل", "peppers": "فلفل",
    "eggplant": "باذنجان", "aubergine": "باذنجان",
    "beans": "فاصوليا", "bean": "فاصوليا",
    "okra": "بامية", "potato": "بطاطس",
    "carrot": "جزر", "onion": "بصل",
    # ── Leafy greens ──
    "lettuce": "خس", "parsley": "بقدونس", "spinach": "سبانخ",
    "arugula": "جرجير", "rocket": "جرجير",
    "coriander": "كزبرة", "cilantro": "كزبرة",
    "dill": "شبت", "molokhia": "ملوخية", "mint": "نعناع",
    "cauliflower": "زهرة", "broccoli": "بروكلي", "cabbage": "ملفوف",
    # ── Fruits ──
    "strawberry": "فراولة", "grape": "عنب", "apple": "تفاح",
    "orange": "برتقال", "pomegranate": "رمان", "pear": "كمثرى",
    "lemon": "ليمون", "lime": "ليمون", "berry": "توت",
    "date": "تمر", "dates": "تمر",
    # ── Grains ──
    "wheat": "قمح", "rice": "رز", "corn": "ذرة",
    "lentil": "عدس", "flour": "دقيق", "oat": "شوفان", "oats": "شوفان",
    # ── Spices ──
    "cardamom": "هيل", "cumin": "كمون", "thyme": "زعتر",
    "spices": "توابل", "spice": "توابل",
    "clove": "قرنفل", "anise": "يانسون", "fennel": "شمر",
    # ── Nuts ──
    "pistachio": "فستق", "pistachios": "فستق",
    "nuts": "مكسرات", "almond": "لوز",
    "cashew": "كاجو", "hazelnut": "بندق",
    "pecan": "بيكان", "peanut": "فول سوداني", "peanuts": "فول سوداني",
    "sesame": "سمسم",
}

# Commodity → Arabic (for risk_windows.py reverse lookup)
COMMODITY_TO_ARABIC: Dict[str, str] = {
    "Tomato": "طماطم", "Cucumber": "خيار", "Pepper": "فلفل",
    "Eggplant": "باذنجان", "Zucchini": "كوسة", "Beans": "فاصوليا",
    "Okra": "بامية", "Lettuce": "خس", "Parsley": "بقدونس",
    "Coriander": "كزبرة", "Mint": "نعناع", "Leafy Greens": "ورقيات",
    "Cabbage": "ملفوف", "Root Vegetables": "جزر",
    "Total Vegetables": "", "Total Fruits": "",
}


# ============================================================================
# STT (Speech-to-Text) CORRECTIONS
# ============================================================================
# Whisper frequently splits Arabic pesticide names. This dict maps
# the most common mis-transcriptions to the correct canonical form.

STT_CORRECTIONS: Dict[str, str] = {
    # Buprofezin variants
    "البير فزن": "البوبروفيزين",
    "بير فيزين": "البوبروفيزين",
    "بوبروفزين": "البوبروفيزين",
    # Imidacloprid
    "اميدا كلوبريد": "الايميداكلوبريد",
    "ايميدا كلوبريد": "الايميداكلوبريد",
    "اميداكلو بريد": "الايميداكلوبريد",
    # Fipronil
    "الفيب رونيل": "الفبرونيل",
    "في برونيل": "الفبرونيل",
    "فايبرونيل": "الفبرونيل",
    # Chlorpyrifos
    "كلور بيريفوس": "الكلوربيريفوس",
    "كلوربيري فوس": "الكلوربيريفوس",
    # Abamectin
    "ابا مكتين": "الابامكتين",
    "ابامك تين": "الابامكتين",
    # Deltamethrin
    "دلتا مثرين": "الدلتامثرن",
    # Cypermethrin
    "سايبر مثرين": "السايبرمثرن",
    # Thiamethoxam
    "ثيا ميثوكسام": "الثياميثوكسام",
    # Carbendazim
    "كاربن دازيم": "الكاربندازيم",
    # Malathion
    "ملا ثيون": "الملاثيون",
    # Profenofos
    "بروفينو فوس": "البروفينوفوس",
    # Azoxystrobin
    "ازوكسي ستروبين": "الازوكسيستروبين",
    # Sample name corrections (STT-specific)
    "الفصولية": "الفاصوليا",
    "فصوليا": "الفاصوليا",
    "فصولية": "الفاصوليا",
    "طماطه": "طماطم",
    "الطماطة": "الطماطم",
}

# Neighbourhood normalization (from core_query_engine.py)
NEIGHBORHOOD_CORRECTIONS: Dict[str, str] = {
    "الاسكان": "الإسكان", "الإسكان": "الإسكان", "اسكان": "الإسكان",
    "الصفراء": "الصفراء", "صفراء": "الصفراء",
    "الموطأ": "الموطأ", "موطأ": "الموطأ",
    "الريان": "الريان", "ريان": "الريان",
    "النخيل": "النخيل", "نخيل": "النخيل",
    "الخليج": "الخليج", "خليج": "الخليج",
    "الفيصلية": "الفيصلية", "فيصلية": "الفيصلية",
    "النهضة": "النهضة", "نهضة": "النهضة",
    "السلام": "السلام", "سلام": "السلام",
    "الشفاء": "الشفاء", "شفاء": "الشفاء",
    "المنتزه": "المنتزه", "منتزه": "المنتزه",
    "العليا": "العليا", "عليا": "العليا",
    "الورود": "الورود", "ورود": "الورود",
    "المروج": "المروج", "مروج": "المروج",
    "البساتين": "البساتين", "بساتين": "البساتين",
    "الهلال": "الهلال", "هلال": "الهلال",
    "الأخضر": "الأخضر", "الاخضر": "الأخضر", "اخضر": "الأخضر",
    "الروضة": "الروضة", "روضة": "الروضة",
    "الفلاح": "الفلاح", "فلاح": "الفلاح",
    "المنار": "المنار", "منار": "المنار",
}


# ============================================================================
# ARABIC NUMERAL CONVERSION
# ============================================================================

ARABIC_NUMERALS: Dict[str, str] = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}

ARABIC_NUM_WORDS: Dict[str, str] = {
    "واحد": "1", "اثنين": "2", "ثلاثة": "3", "اربعة": "4",
    "خمسة": "5", "ستة": "6", "سبعة": "7", "ثمانية": "8",
    "تسعة": "9", "عشرة": "10",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def correct_stt_text(text: str, *, use_fuzzy: bool = True) -> str:
    """Apply STT corrections and optional fuzzy matching.

    Args:
        text: Raw transcribed text from Whisper.
        use_fuzzy: If True, attempt fuzzy matching against known
            pesticide names for words not caught by direct replacement.

    Returns:
        Corrected text string.

    Why fuzzy matching:
        Whisper frequently produces novel mis-transcriptions that are
        close (but not identical) to known pesticide names. A 0.6
        cutoff catches most of these while avoiding false positives.
    """
    corrected = text
    for wrong, right in STT_CORRECTIONS.items():
        corrected = corrected.replace(wrong, right)

    if use_fuzzy:
        words = corrected.split()
        for i, word in enumerate(words):
            if len(word) > 3 and word not in KNOWN_PESTICIDES_AR:
                matches = get_close_matches(
                    word, KNOWN_PESTICIDES_AR, n=1, cutoff=0.6
                )
                if matches:
                    words[i] = matches[0]
        corrected = " ".join(words)

    return corrected


def normalize_arabic_query(query: str) -> str:
    """Normalize Arabic numerals and number words in a query.

    Args:
        query: User query that may contain Arabic numerals (٠-٩)
            or number words (واحد, اثنين, …).

    Returns:
        Query with all numerals converted to Western digits.
    """
    for ar, en in ARABIC_NUMERALS.items():
        query = query.replace(ar, en)
    for word, num in ARABIC_NUM_WORDS.items():
        query = query.replace(word, num)
    return query


def translate_pesticide(name: str) -> Optional[str]:
    """Translate an Arabic pesticide name to English.

    Tries exact match first, then with/without the 'ال' prefix.

    Args:
        name: Arabic pesticide name.

    Returns:
        English name or None if no match found.
    """
    key = name.strip()
    if key in PESTICIDE_AR_TO_EN:
        return PESTICIDE_AR_TO_EN[key]
    # Try adding ال
    with_al = "ال" + key
    if with_al in PESTICIDE_AR_TO_EN:
        return PESTICIDE_AR_TO_EN[with_al]
    # Try removing ال
    if key.startswith("ال") and key[2:] in PESTICIDE_AR_TO_EN:
        return PESTICIDE_AR_TO_EN[key[2:]]
    return None


def normalize_sample_name(name: str) -> str:
    """Normalize an Arabic sample/commodity name to its canonical form.

    Args:
        name: Raw sample name (e.g. 'الفصولية', 'طماطه').

    Returns:
        Canonical Arabic name (e.g. 'فاصوليا', 'طماطم').
    """
    return SAMPLE_CORRECTIONS.get(name.strip(), name.strip())


def normalize_neighborhood(name: str) -> str:
    """Normalize neighbourhood name to canonical form.

    Args:
        name: Raw neighbourhood name.

    Returns:
        Canonical neighbourhood name.
    """
    return NEIGHBORHOOD_CORRECTIONS.get(name.strip(), name.strip())