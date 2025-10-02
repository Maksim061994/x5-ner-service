from typing import List, Tuple, Dict, Any
from sklearn_crfsuite import CRF
from collections import defaultdict
import pandas as pd
import numpy as np
from app.service.rules import TAGS
from app.service.features import (
    tokenize_with_offsets, spans_to_bio,build_lexicons, feature_variant,
    FeatureConfig, FeatureBuilder, bio_validate
)
from sklearn.model_selection import KFold


def make_crf(algorithm: str='lbfgs', c1: float=0.1, c2: float=0.1, max_iter: int=200, all_transitions: bool=True) -> CRF:
    return CRF(
        algorithm=algorithm,
        c1=c1, c2=c2,
        max_iterations=max_iter,
        all_possible_transitions=all_transitions,
        verbose=False
    )


class StackedCRF:
    def __init__(self, tags: List[str] = TAGS, n_splits: int = 5, random_state: int = 42):
        self.tags = tags
        self.n_splits = n_splits
        self.random_state = random_state
        # базовые фичебилдеры
        self.base_builders = {}
        self.base_models = {}
        self.meta_builder = None
        self.meta_model = None

    def _probs_as_features(self, probs_seq: List[Dict[str, float]], prefix: str) -> List[Dict[str, float]]:
        """
        Преобразует маргинальные вероятности CRF к фичам: {prefix:TAG -> prob}
        """
        feats_seq = []
        for p in probs_seq:
            d = {}
            for tag in self.tags:
                d[f'{prefix}:{tag}'] = float(p.get(tag, 0.0))
            feats_seq.append(d)
        return feats_seq

    def _merge_feature_dicts(self, base_feats: List[Dict[str, Any]], *others: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        out = []
        for i in range(len(base_feats)):
            m = dict(base_feats[i])
            for block in others:
                m.update(block[i])
            out.append(m)
        return out

    def fit(self, texts: List[str], spans_list: List[List[Tuple[int, int, str]]]):
        # токенизация и BIO
        sents = [tokenize_with_offsets(t) for t in texts]
        y = [spans_to_bio(s, spans) for s, spans in zip(sents, spans_list)]

        # лексиконы
        tmp_df = pd.DataFrame({'text': texts, 'spans': [str(s) for s in spans_list]})
        lex = build_lexicons(tmp_df)

        # билдеры A/B/C
        base_builder = FeatureBuilder(lexicons=lex)
        self.base_builders = {
            'A': feature_variant(base_builder, 'A'),
            'B': feature_variant(base_builder, 'B'),
            'C': feature_variant(base_builder, 'C'),
        }
        self.meta_builder = FeatureBuilder(
            lexicons=lex,
            cfg=FeatureConfig(use_lemma=True, window=2, add_context_bigrams=True)
        )

        # OOF предсказания для A/B/C
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        # храним по предложению: список из dict-ов на каждый токен
        oof_probs = {k: [None] * len(sents) for k in 'ABC'}
        self.base_models = {k: [] for k in 'ABC'}

        idxs = np.arange(len(sents))
        for fold, (tr, va) in enumerate(kf.split(idxs), start=1):
            # подготовка фич
            X_tr = {k: [self.base_builders[k].sent2features(sents[i]) for i in tr] for k in 'ABC'}
            X_va = {k: [self.base_builders[k].sent2features(sents[i]) for i in va] for k in 'ABC'}
            y_tr = [y[i] for i in tr]

            models_fold = {}
            for k in 'ABC':
                crf = make_crf(c1=0.05 if k == 'A' else 0.1,
                               c2=0.1 if k != 'C' else 0.2,
                               max_iter=200)
                crf.fit(X_tr[k], y_tr)
                models_fold[k] = crf

                # маржинали на валидации:
                # probs: List[List[Dict[tag, prob]]] — по предложениям → по токенам
                probs = crf.predict_marginals(X_va[k])
                for pos, i_sent in enumerate(va):
                    oof_probs[k][i_sent] = probs[pos]

            # сохраняем модели фолда (для усреднения на инференсе)
            for k in 'ABC':
                self.base_models[k].append(models_fold[k])

        # sanity-check: все предложения должны иметь заполненные маржинали
        for k in 'ABC':
            for i, seq in enumerate(oof_probs[k]):
                if seq is None:
                    raise RuntimeError(f"OOF probs for model {k} is None at sentence {i}. "
                                       "Проверьте KFold/раскладку.")

        # Формируем meta-фичи из OOF
        X_meta = []
        for i in range(len(sents)):
            base_feats = self.meta_builder.sent2features(sents[i])  # базовые фичи мета-слоя

            # длины должны совпасть по количеству токенов
            n_tokens = len(base_feats)
            if not (len(oof_probs['A'][i]) == len(oof_probs['B'][i]) == len(oof_probs['C'][i]) == n_tokens):
                raise RuntimeError(f"Token-length mismatch at sentence {i}: "
                                   f"{len(oof_probs['A'][i])}/{len(oof_probs['B'][i])}/"
                                   f"{len(oof_probs['C'][i])} vs {n_tokens}")

            pa = self._probs_as_features(oof_probs['A'][i], 'A')
            pb = self._probs_as_features(oof_probs['B'][i], 'B')
            pc = self._probs_as_features(oof_probs['C'][i], 'C')

            mix = self._merge_feature_dicts(base_feats, pa, pb, pc)
            X_meta.append(mix)

        # 6) Обучаем meta-CRF
        self.meta_model = make_crf(c1=0.05, c2=0.2, max_iter=300)
        self.meta_model.fit(X_meta, y)

        self._train_sents = sents
        self._train_y = y
        return self

    def _avg_predict_marginals(self, builder: FeatureBuilder, models: List[CRF], sent: List[Tuple[str, int, int]]) -> \
    List[Dict[str, float]]:
        X = builder.sent2features(sent)
        # усредняем маржинали по k моделям (фолдам)
        probs_list = [m.predict_marginals_single(X) for m in models]
        out = []
        for t in range(len(X)):
            acc = defaultdict(float)
            for probs in probs_list:
                for tag, p in probs[t].items():
                    acc[tag] += p
            # нормализация
            s = sum(acc.values()) or 1.0
            out.append({k: v / s for k, v in acc.items()})
        return out

    def predict(self, texts: List[str]) -> List[List[str]]:
        res = []
        for text in texts:
            sent = tokenize_with_offsets(text)
            # базовые маржинали
            pa = self._avg_predict_marginals(self.base_builders['A'], self.base_models['A'], sent)
            pb = self._avg_predict_marginals(self.base_builders['B'], self.base_models['B'], sent)
            pc = self._avg_predict_marginals(self.base_builders['C'], self.base_models['C'], sent)
            # meta-фичи
            base_feats = self.meta_builder.sent2features(sent)
            fa = self._probs_as_features(pa, 'A')
            fb = self._probs_as_features(pb, 'B')
            fc = self._probs_as_features(pc, 'C')
            X_meta = self._merge_feature_dicts(base_feats, fa, fb, fc)
            # предикт
            tags = self.meta_model.predict_single(X_meta)
            tags = bio_validate(tags)
            res.append(tags)
        return res

    def predict_spans(self, texts: List[str]) -> List[List[Tuple[int, int, str]]]:
        """
        Возвращает списки (start, end, LABEL) по каждому тексту,
        НЕ удаляя 'O' — т.е. 'O' тоже агрегируется в интервалы.
        LABEL здесь «плоский» (без B-/I-): {'O','TYPE','BRAND','VOLUME','PERCENT'}.
        """
        out_all = []
        tag_seqs = self.predict(texts)  # BIO-последовательности: ['B-TYPE','I-TYPE','O',...]
        for text, tags in zip(texts, tag_seqs):
            sent = tokenize_with_offsets(text)  # [(tok, start, end), ...]
            spans = []
            cur_label = None
            cur_start = None
            prev_end = None

            for (w, a, b), t in zip(sent, tags):
                # плоский лейбл для BIO: 'B-TYPE'/'I-TYPE' -> 'TYPE', 'O' -> 'O'
                if t == 'O':
                    flat = 'O'
                else:
                    bi, lab = t.split('-', 1)
                    flat = lab

                if cur_label is None:
                    # старт первого сегмента
                    cur_label = flat
                    cur_start = a
                    prev_end = b
                else:
                    if flat == cur_label:
                        # продолжаем текущий сегмент
                        prev_end = b
                    else:
                        # закрываем предыдущий сегмент и открываем новый
                        spans.append((cur_start, prev_end, cur_label))
                        cur_label = flat
                        cur_start = a
                        prev_end = b

            # финализация
            if cur_label is not None:
                spans.append((cur_start, prev_end, cur_label))

            out_all.append(spans)
        return out_all
