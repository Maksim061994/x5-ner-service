import pandas as pd
from tqdm import tqdm
import unicodedata
import re
from typing import List, Tuple
import spacy
from ast import literal_eval


def _is_punct(ch: str) -> bool:
    # Любой символ категории Unicode "P" — пунктуация,
    # но % исключаем из удаления
    return unicodedata.category(ch).startswith("P") and ch != "%"


def _strip_punct(s: str) -> str:
    return "".join(ch for ch in s if not _is_punct(ch))


def predict_with_punct(nlp, s: str) -> List[List[str]]:
    """
    Возвращает список [[оригинальный_фрагмент_с_пунктуацией, label], ...]
    на основе предсказаний nlp по строке без пунктуации (кроме %).
    """
    orig_tokens: List[Tuple[int, int, str]] = []
    for m in re.finditer(r"\S+", s):
        start, end = m.span()
        orig_tokens.append((start, end, s[start:end]))

    clean_pieces = []
    clean_spans = []
    clean_cursor = 0
    kept_idx = []

    for i, (st, en, tok) in enumerate(orig_tokens):
        clean_tok = _strip_punct(tok)
        if not clean_tok:
            continue
        if clean_pieces:
            clean_cursor += 1
            clean_pieces.append(" ")
        clean_start = clean_cursor
        clean_pieces.append(clean_tok)
        clean_cursor += len(clean_tok)
        clean_spans.append((clean_start, clean_cursor, i))
        kept_idx.append(i)

    clean_text = "".join(clean_pieces).lower()
    if not clean_text.strip():
        return []

    doc = nlp(clean_text)

    results: List[List[str]] = []
    for ent in doc.ents:
        ent_start, ent_end = ent.start_char, ent.end_char
        covered = []
        for cst, cen, idx in clean_spans:
            if not (cen <= ent_start or cst >= ent_end):
                covered.append(idx)
        if not covered:
            continue
        i0, i1 = min(covered), max(covered)
        start0 = orig_tokens[i0][0]
        end1 = orig_tokens[i1][1]
        orig_fragment = s[start0:end1]
        results.append([orig_fragment, ent.label_])

    return results


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


def convert_model2_to_model1(text, model2_results):
    """
    Конвертирует результаты NER модели 2 в формат модели 1, используя токенизацию.
    """
    if not isinstance(text, str):
        return []

    # Токенизируем текст
    text_tokens = _tokenize_text(text)
    if not text_tokens:
        return []

    # Если результат модели 2 пустой, все токены помечаем как 'O'
    if not model2_results:
        return [(start, end, 'O') for _, start, end in text_tokens]

    # Создаем список для тегов каждого токена, по умолчанию 'O'
    tags = ['O'] * len(text_tokens)

    # Обрабатываем каждую сущность из model2_results
    for entity in model2_results:
        if not isinstance(entity, (list, tuple)) or len(entity) < 2:
            continue

        entity_text, entity_type = entity[0], entity[1]

        if not isinstance(entity_text, str) or not isinstance(entity_type, str):
            continue

        # Токенизируем сущность
        entity_tokens = [token.lower() for token in entity_text.split()]
        if not entity_tokens:
            continue

        # Ищем последовательность токенов сущности в тексте
        i = 0
        while i <= len(text_tokens) - len(entity_tokens):
            # Проверяем, совпадает ли последовательность токенов
            match = True
            for j in range(len(entity_tokens)):
                if text_tokens[i + j][0] != entity_tokens[j]:
                    match = False
                    break

            if match:
                # Нашли совпадение - размечаем токены
                tags[i] = 'B-' + entity_type
                for j in range(1, len(entity_tokens)):
                    tags[i + j] = 'I-' + entity_type

                # Перескакиваем через найденную сущность
                i += len(entity_tokens)
            else:
                i += 1

    # Формируем результат
    result = []
    for (_, start, end), tag in zip(text_tokens, tags):
        result.append((start, end, tag))

    return result


def make_submission(df, spicy_col: str):
    formated_results = [convert_model2_to_model1(text, literal_eval(model2_results)) for (text, model2_results) in
                        zip(df['sample'].tolist(), df[spicy_col].tolist())]
    df['annotation'] = formated_results
    return df[['sample', 'annotation']]



def main():
    nlp = spacy.load("../model/spacy")
    df = pd.read_csv("data_base/russian_supermarket_prices.csv")
    res = []
    for t in tqdm(df["product_name"]):
        pred = convert_model2_to_model1(t, predict_with_punct(nlp, t))
        res.append(pred)
    df["annotation"] = res
    df = df.rename(columns={"product_name": "sample"})
    # Увеличение размера выборки
    samples = []
    for i in range(df.shape[0]):
        var1 = df["sample"][i].lower().replace("«", "").replace("»", "").split(", ")[0][:2]
        var2 = df["sample"][i].lower().replace("«", "").replace("»", "").split(", ")[0][:5]
        var3 = df["sample"][i].lower().replace("«", "").replace("»", "").split(", ")[0][:8]
        var4 = df["sample"][i].lower().replace("«", "").replace("»", "").split(", ")[0][:10]
        var5 = df["sample"][i].lower().replace("«", "").replace("»", "").split(", ")[0][:14]
        var_ = df["sample"][i].lower().replace("«", "").replace("»", "").split(", ")[0].split(" ")
        var7 = " ".join(var_[:2] + var_[-1:])
        var8 = " ".join(var_[-1:] + var_[:2])
        if len(var_) > 4:
            var9 = " ".join(var_[:2] + var_[-2:])
            samples.append(var9)
            var10 = " ".join(var_[-2:] + var_[:2])
            samples.append(var10)
        samples.append(var1)
        samples.append(var2)
        samples.append(var3)
        samples.append(var4)
        samples.append(var5)
        samples.append(var7)
        samples.append(var8)

    preds = []
    samples = list(set(samples))
    for i in tqdm(range(len(samples))):
        t = samples[i]
        pred = convert_model2_to_model1(
            t,
            predict_with_punct(nlp, t)
        )
        preds.append(pred)

    df_res = pd.DataFrame(samples, columns=["sample"])
    df_res["annotation"] = preds
    df_res.to_csv("data/train_dataset_kaggle.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
