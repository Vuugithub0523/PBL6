from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np

def load_trained_transformer_model():
    model_path = "peterhung/vietnamese-accent-marker-xlm-roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_trained_transformer_model() 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

def _load_tags_set(fpath):
    labels = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

label_list = _load_tags_set("./selected_tags_names.txt")
assert len(label_list) == 528, f"Expect {len(label_list)} tags"

TOKENIZER_WORD_PREFIX = "▁"

# Từ điển tiếng Việt thông dụng (không dấu)
VIETNAMESE_DICT = {
    'xin', 'chao', 'cac', 'ban', 'toi', 'la', 'nguoi', 'viet', 'nam',
    'nhin', 'nhung', 'mua', 'thu', 'di', 'em', 'nghe', 'sau', 'len', 'trong', 'nang',
    'anh', 'chi', 'co', 'khong', 'rat', 'dep', 'yeu', 'thuong', 'nha', 'truong',
    'hoc', 'sinh', 'giao', 'vien', 'bai', 'tap', 'lam', 'viec', 'cong', 'ty',
    'den', 'tu', 've', 'di', 'an', 'uong', 'ngu', 'thuc', 'day', 'som',
    'muon', 'gio', 'phut', 'ngay', 'thang', 'nam', 'tuan', 'buoi', 'sang', 'trua',
    'chieu', 'toi', 'dem', 'khuya', 'hom', 'qua', 'nay', 'mai', 'kia',
    'mot', 'hai', 'ba', 'bon', 'nam', 'sau', 'bay', 'tam', 'chin', 'muoi',
    'tram', 'ngan', 'trieu', 'ty', 'nghin', 'van', 'dong', 'tien', 'gia',
    'mua', 'ban', 'cho', 'hoi', 'thi', 'cuoc', 'van', 'de', 'su', 'viec',
    'nguoi', 'dan', 'ong', 'nu', 'con', 'bo', 'me', 'cha', 'ma', 'anh',
    'chi', 'em', 'bac', 'chu', 'co', 'di', 'duong', 'chau', 'cau', 'bau',
    'ong', 'ba', 'thay', 'to', 'lon', 'nho', 'be', 'tre', 'gia', 'tre',
    'dep', 'xau', 'tot', 'xau', 'hay', 'do', 'gioi', 'kem', 'nong', 'lanh',
    'am', 'mat', 'kho', 'de', 'cao', 'thap', 'rong', 'hep', 'dai', 'ngan',
    'nhe', 'nang', 'nhanh', 'cham', 'manh', 'yeu', 'khoe', 'om', 'dau', 'benh',
    'vui', 'buon', 'vui', 've', 'gian', 'du', 'hanh', 'phuc', 'kho', 'khan',
    'noi', 'nao', 'do', 'day', 'dau', 'kia', 'khi', 'nao', 'sao', 'ai',
    'gi', 'dau', 'bao', 'nhieu', 'may', 'the', 'nao', 'ra', 'sao', 'vi',
    'cho', 'nen', 'vi', 'the', 'ma', 'nhung', 'neu', 'thi', 'hay', 'hoac', 'chó','cao','bằng', 'bộ', 'pc'
}

def segment_vietnamese_no_accent(text):
    """
    Tách từ tiếng Việt không dấu sử dụng Dynamic Programming
    """
    text = text.lower().strip()
    n = len(text)
    
    # dp[i] = (độ dài từ tốt nhất kết thúc tại i, vị trí bắt đầu từ đó)
    dp = [(-1, -1)] * (n + 1)
    dp[0] = (0, 0)
    
    # Duyệt qua từng vị trí
    for i in range(1, n + 1):
        # Thử tất cả các từ có thể kết thúc tại vị trí i
        for j in range(max(0, i - 15), i):  # Giới hạn độ dài từ <= 15
            word = text[j:i]
            if word in VIETNAMESE_DICT and dp[j][0] != -1:
                if dp[i][0] == -1 or dp[j][0] + 1 > dp[i][0]:
                    dp[i] = (dp[j][0] + 1, j)
    
    # Truy vết để lấy các từ
    if dp[n][0] == -1:
        # Không tách được, trả về từng ký tự
        return list(text)
    
    words = []
    pos = n
    while pos > 0:
        start_pos = dp[pos][1]
        if start_pos == pos:  # Không tách được đoạn này
            break
        words.append(text[start_pos:pos])
        pos = start_pos
    
    words.reverse()
    
    # Nếu không tách được hết, thêm phần còn lại
    if pos > 0:
        words.insert(0, text[:pos])
    
    return words if words else [text]

def chars_to_text(char_array):
    """Chuyển mảng ký tự thành chuỗi văn bản"""
    return ''.join(char_array).lower()

def insert_accents(tokens, model, tokenizer):
    """Thêm dấu cho các token đã được tách từ"""
    inputs = tokenizer(tokens,
                        is_split_into_words=True,
                        truncation=True,
                        padding=True,
                        return_tensors="pt"
                        )
    input_ids = inputs['input_ids']
    subword_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    subword_tokens = subword_tokens[1:-1]

    with torch.no_grad():
        inputs.to(device)
        outputs = model(**inputs)

    predictions = outputs["logits"].cpu().numpy()
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions[0][1:-1]

    assert len(subword_tokens) == len(predictions)

    return subword_tokens, predictions 

def merge_tokens_and_preds(tokens, predictions): 
    """Gộp các subword token thành từ hoàn chỉnh"""
    merged_tokens_preds = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        label_indexes = set([predictions[i]])
        if tok.startswith(TOKENIZER_WORD_PREFIX):
            tok_no_prefix = tok[len(TOKENIZER_WORD_PREFIX):]
            cur_word_toks = [tok_no_prefix]
            j = i + 1
            while j < len(tokens):
                if not tokens[j].startswith(TOKENIZER_WORD_PREFIX):
                    cur_word_toks.append(tokens[j])
                    label_indexes.add(predictions[j])
                    j += 1
                else:
                    break
            cur_word = ''.join(cur_word_toks)
            merged_tokens_preds.append((cur_word, label_indexes))
            i = j
        else:
            merged_tokens_preds.append((tok, label_indexes))
            i += 1

    return merged_tokens_preds

def get_accented_words(merged_tokens_preds, label_list):
    """Thêm dấu vào các từ dựa trên predictions"""
    accented_words = []
    for word_raw, label_indexes in merged_tokens_preds:
        word_accented = word_raw
        for label_index in label_indexes:
            tag_name = label_list[int(label_index)]
            raw, vowel = tag_name.split("-")
            if raw and raw in word_raw:
                word_accented = word_raw.replace(raw, vowel)
                break
        accented_words.append(word_accented)

    return accented_words

def process_char_array(char_array):
    """
    Xử lý mảng ký tự và trả về câu có dấu
    
    Args:
        char_array: list các ký tự, ví dụ ['x','i','n','c','h','a','o','c','a','c','b','a','n']
    
    Returns:
        str: Câu tiếng Việt có dấu
    """
    # Bước 1: Chuyển mảng ký tự thành text liền
    text_input = chars_to_text(char_array)
    print(f"Text liền: {text_input}")
    
    # Bước 2: Tách từ tự động bằng Dynamic Programming
    tokens = segment_vietnamese_no_accent(text_input)
    print(f"Tokens sau khi tách: {tokens}")
    
    # Bước 3: Dùng model XLM-RoBERTa thêm dấu
    subword_tokens, predictions = insert_accents(tokens, model, tokenizer)
    print(f"Subword tokens: {subword_tokens}")
    print(f"Predictions: {[label_list[pred] for pred in predictions]}")
    
    # Bước 4: Merge và thêm dấu
    merged_tokens_preds = merge_tokens_and_preds(subword_tokens, predictions)
    accented_words = get_accented_words(merged_tokens_preds, label_list)
    
    # Bước 5: Ghép thành câu hoàn chỉnh
    result = ' '.join(accented_words)
    return result

# Test với ví dụ
if __name__ == "__main__":
    print("="*60)
    print("Nhập chuỗi tiếng Việt KHÔNG DẤU để thêm dấu tự động")
    print("Ví dụ:  xinchao cacban")
    print("Enter rỗng để thoát.")
    print("="*60)

    while True:
        raw = input("\nNhập chuỗi không dấu: ").strip()
        if raw == "":
            break

        # Bỏ khoảng trắng, giữ lại chuỗi liền không dấu (giống logic test ban đầu)
        raw_clean = raw.replace(" ", "").lower()

        # Chuyển thành mảng ký tự như các ví dụ test cũ
        char_input = list(raw_clean)
        print(f"Input char array: {char_input}")

        output = process_char_array(char_input)
        print(f"\n✅ Câu có dấu: {output}")