from typing import List, Tuple, Optional

Span = Tuple[int, int, str]


def _words_by_spaces(text: str) -> List[Tuple[int,int]]:
    """Разбить всю строку на «слова» как непрерывные группы непробельных символов."""
    words = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n: break
        j = i
        while j < n and not text[j].isspace():
            j += 1
        words.append((i, j))
        i = j
    return words


def _label_for_word(word: Tuple[int,int], spans: List[Span]) -> Optional[str]:
    """Вернуть базовую метку ('TYPE','BRAND','VOLUME','PERCENT') для слова по пересечению со спанами, иначе None."""
    wa, wb = word
    best_lab, best_ol = None, 0
    for a, b, lab in spans:
        # пересечение длинной >0
        ol = max(0, min(wb, b) - max(wa, a))
        if ol > best_ol:
            best_ol = ol
            best_lab = lab
    return best_lab if best_ol > 0 else None


def spans_to_bio_splits(text: str, spans: List[Span]) -> List[Span]:
    """
    Гарантирует: одно слово -> один кортеж (start,end,label).
    Лейблы: 'O' для слов вне сущностей; внутри сущностей — BIO по последовательным словам.
    """
    # нормализуем входные метки до базовых (без 'B-','I-')
    norm_spans: List[Span] = []
    for a, b, lab in sorted(spans, key=lambda x: (x[0], x[1])):
        base = lab.upper().strip()
        if base.startswith('B-') or base.startswith('I-'):
            base = base.split('-', 1)[1]
        norm_spans.append((a, b, base))

    words = _words_by_spaces(text)
    result: List[Span] = []

    prev_base = None
    for (a, b) in words:
        base = _label_for_word((a, b), norm_spans)  # None -> O
        if base is None or base == 'O':
            result.append((a, b, 'O'))
            prev_base = None
        else:
            tag = 'B-' + base if prev_base != base else 'I-' + base
            result.append((a, b, tag))
            prev_base = base
    return result


def _split_into_tokens(text):
    """Разбивает текст на токены по пробелам"""
    if not text:
        return []

    tokens = []
    start = 0
    for i, char in enumerate(text):
        if char == ' ':
            if start < i:
                tokens.append((start, i))
            start = i + 1

    if start < len(text):
        tokens.append((start, len(text)))

    return tokens


def _tokenize_text(text):
    """Токенизирует текст и возвращает список токенов с их текстом и позициями"""
    tokens = _split_into_tokens(text)
    token_texts = []
    for start, end in tokens:
        token_texts.append((text[start:end].lower(), start, end))
    return token_texts


def convert_pred_to_output(text, model2_results):
    """
    Конвертирует результаты NER модели 2 в формат модели 1, используя токенизацию.
    """
    if not isinstance(text, str):
        return []

    text_tokens = _tokenize_text(text)
    if not text_tokens:
        return []

    if not model2_results:
        return [(start, end, 'O') for _, start, end in text_tokens]

    tags = ['O'] * len(text_tokens)
    for entity in model2_results:
        if not isinstance(entity, (list, tuple)) or len(entity) < 2:
            continue

        entity_text, entity_type = entity[0], entity[1]

        if not isinstance(entity_text, str) or not isinstance(entity_type, str):
            continue

        entity_tokens = [token.lower() for token in entity_text.split()]
        if not entity_tokens:
            continue

        i = 0
        while i <= len(text_tokens) - len(entity_tokens):
            match = True
            for j in range(len(entity_tokens)):
                if text_tokens[i + j][0] != entity_tokens[j]:
                    match = False
                    break

            if match:
                tags[i] = 'B-' + entity_type
                for j in range(1, len(entity_tokens)):
                    tags[i + j] = 'I-' + entity_type
                i += len(entity_tokens)
            else:
                i += 1

    result = []
    for (_, start, end), tag in zip(text_tokens, tags):
        result.append((start, end, tag))
    return result
