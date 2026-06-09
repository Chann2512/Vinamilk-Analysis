import math
from collections import Counter

# Standard e-commerce and product keywords for relevancy ratio calculation
VALID_KEYWORDS = {
    "vinamilk", "vnm", "shop", "cửa hàng", "sản phẩm", "hàng", "san pham",
    "giao hàng", "ship", "shipper", "giao", "đóng gói", "bao bọc",
    "chất lượng", "vị", "sữa", "hộp", "tiêu hóa", "con", "bé", "giá", "mua",
    "hương vị", "bao bì", "mẫu mã", "dịch vụ", "tư vấn", "date",
    "hạn sử dụng", "hsd", "uống", "sử dụng", "cung cấp", "bổ sung",
    "tốt", "hài lòng", "tuyệt vời", "ngon", "trẻ con", "cha mẹ", "ba mẹ"
}

def shannon_entropy(text):
    """
    Computes the Shannon Entropy of the string to measure character uncertainty.
    Spam reviews with repetitive letters (e.g. 'aaaaaaa', 'dhshdhh') have low entropy.
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
        
    return entropy

def dictionary_ratio(text, keywords=None):
    """
    Calculates the ratio of relevant keywords to the total word count in a text.
    """
    if keywords is None:
        keywords = VALID_KEYWORDS
        
    if not isinstance(text, str) or not text.strip():
        return 0.0
        
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
        
    valid_count = sum(1 for w in words if w in keywords)
    return valid_count / len(words)

def length_filter(text, min_words=2, max_words=200):
    """
    Checks if the word count falls within a specific range.
    """
    if not isinstance(text, str):
        return False
    words = text.split()
    return min_words <= len(words) <= max_words

def apply_filters(text):
    """
    Applies the full filtering sequence on raw text:
    Returns (is_valid, reason)
    """
    if not isinstance(text, str) or not text.strip():
        return False, "empty"
        
    # 1. Word length check (min 2 words to make a meaningful comment)
    if not length_filter(text, min_words=2):
        return False, "length"
        
    # 2. Shannon Entropy check (typical natural comments are within [1.5, 5.2])
    entropy = shannon_entropy(text)
    if not (1.5 < entropy < 5.2):
        return False, "entropy"
        
    # 3. Valid keyword density check (must contain >= 15% relevant vocabulary)
    ratio = dictionary_ratio(text)
    if ratio < 0.15:
        return False, "keyword_ratio"
        
    return True, "valid"
