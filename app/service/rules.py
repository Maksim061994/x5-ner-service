import pymorphy2
from dataclasses import dataclass
import re


@dataclass
class Span:
    """Структура для представления спана сущности."""
    start: int
    end: int
    entity: str
    text: str


_MORPH = pymorphy2.MorphAnalyzer()
TAGS = ['O', 'B-BRAND', 'I-BRAND', 'B-TYPE', 'I-TYPE', 'B-VOLUME', 'I-VOLUME', 'B-PERCENT', 'I-PERCENT']

# предкомпилированные регексы
RE_NUM = re.compile(r'^\d+([.,]\d+)?$')
RE_UNIT = re.compile(r'(?i)^(мл|л|г|кг|шт|уп|пак|бут|бан|таб|мг|мм|см|м)$')
RE_NUM_UNIT_STUCK = re.compile(r'(?i)^\d{1,5}([.,]\d{1,2})?(мл|л|г|кг|шт|уп|пак|бут|бан|таб|мг|мм|см|м)$')
RE_VOLUME_ANY = re.compile(r'(?i)\b\d{1,5}([.,]\d{1,2})?\s?(мл|л|г|кг|шт|уп|пак|бут|бан|таб|мг|мм|см|м)\b')
RE_PERCENT = re.compile(r'(?i)\b\d{1,2}([.,]\d{1,2})?\s?%')
RE_HAS_TM = re.compile(r'[®™]')
RE_LAT = re.compile(r'[A-Za-z]')
RE_CYR = re.compile(r'[А-Яа-яЁё]')

