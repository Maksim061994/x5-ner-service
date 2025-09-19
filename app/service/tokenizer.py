"""Токенизация текста с сохранением позиций символов."""

import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Token:
    """Структура для представления токена."""
    text: str
    start: int
    end: int


class SimpleTokenizer:
    """Простой токенизатор с сохранением позиций символов."""
    
    def __init__(self):
        # Паттерн для токенизации: слова, числа, знаки препинания
        self.token_pattern = re.compile(
            r'\S+',  # Любая последовательность непробельных символов
            re.UNICODE
        )
    
    def tokenize(self, text: str) -> List[Token]:
        """Токенизация текста с сохранением позиций."""
        tokens = []
        
        for match in self.token_pattern.finditer(text):
            start, end = match.span()
            token_text = match.group(0)
            
            tokens.append(Token(
                text=token_text,
                start=start,
                end=end
            ))
        
        return tokens
    
    def get_token_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """Получение границ токенов."""
        tokens = self.tokenize(text)
        return [(token.start, token.end) for token in tokens]
    
    def find_token_at_position(self, text: str, position: int) -> Token:
        """Поиск токена по позиции в тексте."""
        tokens = self.tokenize(text)
        
        for token in tokens:
            if token.start <= position < token.end:
                return token
        
        # Если позиция не найдена, возвращаем ближайший токен
        if tokens:
            if position < tokens[0].start:
                return tokens[0]
            elif position >= tokens[-1].end:
                return tokens[-1]
        
        return None
