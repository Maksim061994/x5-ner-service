from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
import re, ast
from app.service.rules import (
    _MORPH, RE_CYR, RE_LAT, RE_PERCENT, RE_HAS_TM, RE_VOLUME_ANY, RE_NUM, RE_NUM_UNIT_STUCK, RE_UNIT, TAGS
)
import pandas as pd
import unicodedata


def mixed_script(s: str) -> bool:
    return bool(RE_LAT.search(s) and RE_CYR.search(s))


def word_shape(s: str) -> str:
    out = []
    for ch in s:
        if ch.isdigit(): out.append('d')
        elif ch.isalpha(): out.append('X' if ch.isupper() else 'x')
        else: out.append(ch)
    return ''.join(out)


def safe_lemma(s: str) -> str:
    if not _MORPH:
        return s.lower()
    try:
        p = _MORPH.parse(s)[0]
        return p.normal_form
    except Exception:
        return s.lower()


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Простая токенизация: числа, слова (лат/кир), отдельные символы.
    Важно сохранять оффсеты.
    """
    tokens = []
    for m in re.finditer(r'\d+[.,]?\d*%?|'       # числа и проценты
                         r'[A-Za-zА-Яа-яЁё]+'   # слова лат/кир
                         r'|[^\s\w]',           # одиночные знаки
                         text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


def spans_to_bio(tokens: List[Tuple[str,int,int]], spans: List[Tuple[int,int,str]], tagset: List[str]=TAGS) -> List[str]:
    """
    Преобразует char-спаны в токеновые BIO-ярлыки.
    Жёсткое правило: на один токен один тег; если спан пересекает — считаем попадание.
    """
    y = ['O'] * len(tokens)
    # нормализуем входные спаны
    norm = []
    for a,b,t in spans:
        t = t.strip()
        if t.startswith('B-') or t.startswith('I-') or t=='O':
            ent = t.split('-')[-1]
        else:
            ent = t
        norm.append((a,b,ent))

    # маркируем
    mark = [None]*len(tokens)
    for i,(tok, a, b) in enumerate(tokens):
        for sa,sb,ent in norm:
            overlap = max(0, min(b, sb) - max(a, sa))
            if overlap>0:
                mark[i] = ent
                break

    # в BIO
    prev_ent = None
    for i, ent in enumerate(mark):
        if ent is None:
            y[i] = 'O'
            prev_ent = None
        else:
            if prev_ent == ent:
                y[i] = f'I-{ent}'
            else:
                y[i] = f'B-{ent}'
            prev_ent = ent

    # фильтр к допустимому множеству
    y = [t if t in tagset else 'O' for t in y]
    return y


def bio_validate(tags: List[str]) -> List[str]:
    """BIO-валидатор: запрещаем I-* без предшествующего B-*. Исправляем на B-* или O."""
    res = tags[:]
    prev = 'O'
    prev_ent = None
    for i,t in enumerate(res):
        if t=='O':
            prev, prev_ent = 'O', None
            continue
        p = t.split('-',1)
        if len(p)!=2:
            res[i]='O'; prev,prev_ent='O',None; continue
        bi, ent = p
        if bi=='B':
            prev='B'; prev_ent=ent
        elif bi=='I':
            if prev in ('B','I') and prev_ent==ent:
                prev='I'
            else:
                # некорректный I-: превращаем в B-
                res[i]=f'B-{ent}'
                prev='B'; prev_ent=ent
    return res


def build_lexicons(train_df: pd.DataFrame) -> Dict[str,set]:
    """
    Строим простые лексиконы брендов/типов/юнитов по train.
    Ожидает колонки: text, spans (строка со списком кортежей).
    """
    brand, ttype, unit = set(), set(), set()
    for _,row in train_df.iterrows():
        text = str(row['text'])
        spans = parse_spans(row['spans'])
        toks = tokenize_with_offsets(text)
        bio = spans_to_bio(toks, spans)
        for (w,_,_), tag in zip(toks, bio):
            lw = w.lower()
            if tag.endswith('BRAND'):
                brand.add(lw)
            if tag.endswith('TYPE'):
                ttype.add(lw)
            if RE_UNIT.fullmatch(lw):
                unit.add(lw)
            # склейки типа 500мл учитываем как факт юнитов
            if RE_NUM_UNIT_STUCK.fullmatch(lw):
                unit.add(re.sub(r'^\d+([.,]\d+)?', '', lw))
    return {'brand': brand, 'type': ttype, 'unit': unit}


def parse_spans(spans_str: str) -> List[Tuple[int, int, str]]:
    if not isinstance(spans_str, str):
        return []
    s = spans_str.strip()
    if not s.startswith('[') or not s.endswith(']'):
        return []
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    try:
        val = ast.literal_eval(s)
        return [(int(a), int(b), str(c)) for a,b,c in val]
    except Exception:
        return []


@dataclass
class FeatureConfig:
    use_lemma: bool = True
    window: int = 2
    add_context_bigrams: bool = True

@dataclass
class FeatureBuilder:
    lexicons: Dict[str,set] = field(default_factory=lambda: {'brand':set(),'type':set(),'unit':set()})
    cfg: FeatureConfig = field(default_factory=FeatureConfig)

    def token_feats(self, sent: List[Tuple[str,int,int]], i: int) -> Dict[str,Any]:
        w, a, b = sent[i]
        lw = w.lower()
        feats = {
            'bias': 1.0,
            'w': lw,
            'shape': word_shape(w),
            'is_title': w.istitle(),
            'is_upper': w.isupper(),
            'is_digit': w.isdigit(),
            'len': len(w),
            'pre1': lw[:1], 'pre2': lw[:2], 'pre3': lw[:3], 'pre4': lw[:4],
            'suf1': lw[-1:], 'suf2': lw[-2:], 'suf3': lw[-3:], 'suf4': lw[-4:],
            'has_tm': bool(RE_HAS_TM.search(w)),
            'is_num_unit_stuck': bool(RE_NUM_UNIT_STUCK.fullmatch(lw)),
            'is_volume_like': bool(RE_VOLUME_ANY.search(w)),
            'is_percent_like': bool(RE_PERCENT.search(w)),
            'has_mixed_script': mixed_script(w),
            'in_brand_lex': lw in self.lexicons['brand'],
            'in_type_lex': lw in self.lexicons['type'],
            'in_unit_lex': lw in self.lexicons['unit'],
            'BOS': i==0, 'EOS': i==len(sent)-1,
        }
        if self.cfg.use_lemma:
            feats['lemma'] = safe_lemma(w)

        # контекст
        W = self.cfg.window
        for off in range(1, W+1):
            j = i-off
            if j>=0:
                wj = sent[j][0]
                feats.update({
                    f'-{off}:w': wj.lower(),
                    f'-{off}:shape': word_shape(wj)
                })
        for off in range(1, W+1):
            j = i+off
            if j<len(sent):
                wj = sent[j][0]
                feats.update({
                    f'+{off}:w': wj.lower(),
                    f'+{off}:shape': word_shape(wj)
                })

        if self.cfg.add_context_bigrams and len(sent)>1:
            if i>0:
                feats['-1_bigram'] = f"{sent[i-1][0].lower()}__{lw}"
            if i+1<len(sent):
                feats['+1_bigram'] = f"{lw}__{sent[i+1][0].lower()}"
        return feats

    def sent2features(self, sent: List[Tuple[str,int,int]]) -> List[Dict[str,Any]]:
        return [self.token_feats(sent, i) for i in range(len(sent))]


def feature_variant(builder: FeatureBuilder, kind: str) -> FeatureBuilder:
    """
    A: лёгкие ортографические/регексы (use_lemma=False, window=1, no bigrams)
    B: добавляем лексиконы (как есть)
    C: усиливаем контекст (window=2, bigrams=True)
    """
    b = FeatureBuilder(lexicons=builder.lexicons, cfg=FeatureConfig(**vars(builder.cfg)))
    if kind=='A':
        b.cfg.use_lemma = False
        b.cfg.window = 1
        b.cfg.add_context_bigrams = False
    elif kind=='B':
        b.cfg.use_lemma = True
        b.cfg.window = 1
        b.cfg.add_context_bigrams = True
    elif kind=='C':
        b.cfg.use_lemma = True
        b.cfg.window = 2
        b.cfg.add_context_bigrams = True
    elif kind=='D':
        b.cfg.use_lemma = True
        b.cfg.window = 4
        b.cfg.add_context_bigrams = True
    return b


def _is_punct(ch: str) -> bool:
    # Любой символ категории Unicode "P" — пунктуация,
    # но % исключаем из удаления
    return unicodedata.category(ch).startswith("P") and ch != "%"


def _strip_punct(s: str) -> str:
    return "".join(ch for ch in s if not _is_punct(ch))


def predict_spans_spacy(nlp, s: str) -> List[List[str]]:
    """
    Возвращает список [[оригинальный_фрагмент_с_пунктуацией, label], ...]
    на основе предсказаний nlp по строке без пунктуации (кроме %).
    """
    orig_tokens: List[Tuple[int,int,str]] = []
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
