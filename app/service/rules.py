"""Регулярные выражения для извлечения VOLUME и PERCENT сущностей."""

import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Span:
    """Структура для представления спана сущности."""
    start: int
    end: int
    entity: str
    text: str


class RegexRules:
    """Класс для работы с регулярными выражениями."""
    
    def __init__(self):
        # Компилируем регулярные выражения
        self.volume_pattern = re.compile(
            r'\b(\d+[.,]?\d*)\s?(л|ml|мл|г|кг|шт)\b',
            re.IGNORECASE
        )
        self.percent_pattern = re.compile(
            r'\b(\d+[.,]?\d*)\s?%(\b|$)',
            re.IGNORECASE
        )
    
    def extract_volumes(self, text: str) -> List[Span]:
        """Извлечение VOLUME сущностей из текста."""
        spans = []
        
        for match in self.volume_pattern.finditer(text):
            start, end = match.span()
            matched_text = match.group(0)
            
            # Создаем спаны для каждого токена
            tokens = matched_text.split()
            current_pos = start
            
            for i, token in enumerate(tokens):
                token_start = text.find(token, current_pos)
                token_end = token_start + len(token)
                
                if i == 0:
                    entity = "B-VOLUME"
                else:
                    entity = "I-VOLUME"
                
                spans.append(Span(
                    start=token_start,
                    end=token_end,
                    entity=entity,
                    text=token
                ))
                
                current_pos = token_end
        
        return spans
    
    def extract_percents(self, text: str) -> List[Span]:
        """Извлечение PERCENT сущностей из текста."""
        spans = []
        
        for match in self.percent_pattern.finditer(text):
            start, end = match.span()
            matched_text = match.group(0)
            
            # Создаем спаны для каждого токена
            tokens = matched_text.split()
            current_pos = start
            
            for i, token in enumerate(tokens):
                token_start = text.find(token, current_pos)
                token_end = token_start + len(token)
                
                if i == 0:
                    entity = "B-PERCENT"
                else:
                    entity = "I-PERCENT"
                
                spans.append(Span(
                    start=token_start,
                    end=token_end,
                    entity=entity,
                    text=token
                ))
                
                current_pos = token_end
        
        return spans
    
    def extract_all(self, text: str) -> List[Span]:
        """Извлечение всех сущностей из текста."""
        spans = []
        spans.extend(self.extract_volumes(text))
        spans.extend(self.extract_percents(text))
        return spans
