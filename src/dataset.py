import random
import re
import sys
import typing as tp
import unicodedata

from sacremoses import MosesPunctNormalizer
from torch.utils.data import Dataset
from transformers import NllbTokenizer

# TODO: Refactor this

LANGS = [('ru', 'rus_Cyrl'), ('lez', 'lez_Cyrl')]

class TextPreprocessor:
    def __init__(self, lang: str = "en", replace_by: str = " "):
        # Инициализация MosesPunctNormalizer
        self.mpn = MosesPunctNormalizer(lang=lang)
        self.mpn.substitutions = [
            (re.compile(r), sub) for r, sub in self.mpn.substitutions
        ]
        # Инициализация функции замены непечатаемых символов
        self.replace_nonprint = self.get_non_printing_char_replacer(replace_by)

    def get_non_printing_char_replacer(self, replace_by: str) -> tp.Callable[[str], str]:
        non_printable_map = {
            ord(c): replace_by
            for c in (chr(i) for i in range(sys.maxunicode + 1))
            # same as \p{C} in perl
            # see https://www.unicode.org/reports/tr44/#General_Category_Values
            if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
        }

        def replace_non_printing_char(line: str) -> str:
            return line.translate(non_printable_map)

        return replace_non_printing_char

    def preprocess(self, text: str) -> str:
        # Шаг 1: Нормализация пунктуации с использованием MosesPunctNormalizer
        clean = self.mpn.normalize(text)
        # Шаг 2: Замена непечатаемых символов
        clean = self.replace_nonprint(clean)
        # Шаг 3: Нормализация текста (например, замена "𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞" на "Francesca")
        clean = unicodedata.normalize("NFKC", clean)
        return clean


class ThisDataset(Dataset):
    def __init__(self, df, random: bool):
        self.df = df
        self.random = random

    def __getitem__(self, idx):
        if self.random:
            item = self.df.iloc[random.randint(0, len(self.df)-1)]  # noqa: S311
        else:
            item = self.df.iloc[idx]

        return item
    
    def __len__(self):
        return len(self.df)

class CollateFn():
    def __init__(self, tokenizer: NllbTokenizer, ignore_index = -100, max_length = 128) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.text_preprocessor = TextPreprocessor()

    def __call__(self, batch: list) -> dict:
        langs = random.sample(LANGS, 2) # Random choice between [ru->lez, lez->ru]
        return self.pad_batch(batch, langs)

    def pad_batch(self, batch: list, langs) -> dict:
        (l1, lang1), (l2, lang2) = langs

        x_texts, y_texts = [], []
        for item in batch:
            x_texts.append(self.text_preprocessor.preprocess(item[l1]))
            y_texts.append(self.text_preprocessor.preprocess(item[l2]))

        self.tokenizer.src_lang = lang1
        # x = self.tokenizer(x_texts, return_tensors='pt', padding='longest')
        x = self.tokenizer(x_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        self.tokenizer.src_lang = lang2
        # y = self.tokenizer(y_texts, return_tensors='pt', padding='longest')
        y = self.tokenizer(y_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        y.input_ids[y.input_ids == self.tokenizer.pad_token_id] = self.ignore_index

        return {
            "x": x,
            "y": y,
        }
    
class LangCollateFn(CollateFn):
    def __init__(self, tokenizer: NllbTokenizer, ignore_index = -100, max_length = 128, src_lang = None, tgt_lang = None) -> None:
        super().__init__(tokenizer, ignore_index, max_length)

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __call__(self, batch: list) -> dict:
        if self.src_lang == 'rus_Cyrl' and self.tgt_lang == 'lez_Cyrl':
            langs = LANGS
        elif self.src_lang == 'lez_Cyrl' and self.tgt_lang == 'rus_Cyrl':
            langs = LANGS[::-1]
        else:
            raise ValueError("Not valid src_lang and tgt_lang")
        
        return self.pad_batch(batch, langs)


class TestCollateFn():
    def __init__(self, tokenizer: NllbTokenizer, src_lang, tgt_lang, a=32, b=3, max_input_length=1024, num_beams=4, max_length = 128):
        self.tokenizer = tokenizer

        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        self.max_length = max_length

        self.text_preprocessor = TextPreprocessor()

        # TODO: Change this
        self.covert = {
            "rus_Cyrl": "ru",
            "lez_Cyrl": "lez"
        }

        self.a = a
        self.b = b
        self.max_input_length = max_input_length
        self.num_beams = num_beams

    def __call__(self, batch: list) -> dict:
        return self.pad_batch(batch)

    def pad_batch(self, batch: list) -> dict:        
        x_texts, y_texts = [], []
        for item in batch:
            x_texts.append(self.text_preprocessor.preprocess(item[self.covert[self.tokenizer.src_lang]]))
            y_texts.append(self.text_preprocessor.preprocess(item[self.covert[self.tokenizer.tgt_lang]]))

        inputs = self.tokenizer(x_texts, return_tensors='pt', padding='longest')
        # inputs = self.tokenizer(x_texts,  return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        return {
            "x": inputs,
            "forced_bos_token_id": self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang),
            "max_new_tokens": int(self.a + self.b * inputs.input_ids.shape[1]), # TODO: Think about it
            "num_beams": self.num_beams,
            "tgt_text": y_texts,
            "src_lang": self.tokenizer.src_lang,
            "tgt_lang": self.tokenizer.tgt_lang
        }