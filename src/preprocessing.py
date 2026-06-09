import re
import unicodedata

# Dictionary mapping Vietnamese slang and abbreviations to standard terms
SLANG_MAP = {
    "sp": "sản phẩm",
    "k": "không",
    "ko": "không",
    "ok": "tốt",
    "oke": "tốt",
    "oki": "tốt",
    "good": "tốt"
}

# Standard Vietnamese stopwords used in e-commerce review cleaning
VIETNAMESE_STOPWORDS = {
    'và', 'là', 'thì', 'mà', 'nhưng', 'của', 'cho', 'để', 'ở', 'tại', 
    'trong', 'ngoài', 'với', 'như', 'các', 'những', 'cái', 'sự', 'cuộc', 
    'việc', 'được', 'bị', 'về', 'ra', 'vào', 'lên', 'xuống'
}

def normalize_unicode(text):
    """
    Standardizes Unicode characters to NFKC layout and resolves fancy styles.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return unicodedata.normalize('NFKC', text)

def remove_emoji(text):
    """
    Removes common emojis and emoticons from text.
    """
    if not isinstance(text, str):
        return ""
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def remove_url(text):
    """
    Strips standard HTTP/HTTPS URLs and www references.
    """
    if not isinstance(text, str):
        return ""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r'', text)

def remove_special_characters(text):
    """
    Retains only standard Vietnamese letters, English letters, digits, and spaces.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^a-zA-Z0-9À-ỹ\s]', ' ', text)

def normalize_repeat(text):
    """
    Compresses repeating characters (e.g. ngonnn -> ngon, hiii -> hi).
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r'(.)\1+', r'\1', text)

def replace_slang(text):
    """
    Translates common slang and abbreviations using SLANG_MAP.
    """
    if not isinstance(text, str):
        return ""
    words = text.split()
    replaced_words = []
    for w in words:
        w_lower = w.lower()
        if w_lower in SLANG_MAP:
            replaced_words.append(SLANG_MAP[w_lower])
        else:
            replaced_words.append(w)
    return " ".join(replaced_words)

def remove_stopwords(text):
    """
    Filters out Vietnamese stopwords. Tries to use underthesea word tokenizer,
    falling back to standard space split if underthesea is not present.
    """
    if not isinstance(text, str):
        return ""
    try:
        from underthesea import word_tokenize
        words = word_tokenize(text)
    except ImportError:
        words = text.split()
    cleaned_words = [w for w in words if w.lower() not in VIETNAMESE_STOPWORDS]
    return " ".join(cleaned_words)

def clean_text(text, remove_sw=True):
    """
    Executes the full preprocessing pipeline on the input text:
    1. Normalize Unicode (NFKC)
    2. Remove URLs
    3. Remove Emojis
    4. Remove Special Characters
    5. Compress Repeating Characters
    6. Replace Slang/Abbreviations
    7. Optionally remove Stopwords
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = normalize_unicode(text)
    text = remove_url(text)
    text = remove_emoji(text)
    text = remove_special_characters(text)
    text = normalize_repeat(text)
    text = replace_slang(text)
    
    if remove_sw:
        text = remove_stopwords(text)
        
    # Standardize whitespace spacing
    text = re.sub(r'\s+', ' ', text).strip()
    return text
