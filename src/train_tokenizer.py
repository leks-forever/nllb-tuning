import json
import os
import shutil
from collections import Counter

import pandas as pd
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

from src.dataset import TextPreprocessor

# TODO: REFACTOR THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")
df_val = pd.read_csv("data/val.csv")

df = pd.concat([df_train, df_test, df_val], ignore_index=True)
 
# df = pd.read_csv("data/cleared_v2/cleared_v2.csv")

# this code is adapted from  the Stopes repo of the NLLB team
# https://github.com/facebookresearch/stopes/blob/main/stopes/pipelines/monolingual/monolingual_line_processor.py#L21
 

text_preprocessor = TextPreprocessor()
 
 
def update_nllb_tokenizer(
    old_tokenizer: NllbTokenizer,
    new_spm_path: str,
    new_lang_codes: list[str],
) -> NllbTokenizer:
    """
    Create a new tokenizer for NLLB, with an updated sentencepiece model and some new language codes.
    In order to get rid of the old (and wrong) added token encoders/decoders, we save the tokenizer to disk and remove those files.
    :param old_tokenizer: the original tokenizer
    :param new_spm_path: path to the file with the sentncepiece model
    :param new_lang_codes: list of the new codes to add to the tokenizer
    :return: the new NllbTokenizer
    """
    TKN_DIR = "old_tokenizer"  # todo: make it a temp dir
    old_tokenizer.save_pretrained(TKN_DIR)
 
    with open(f"{TKN_DIR}/tokenizer_config.json", "r") as f:
        cfg = json.load(f)
    cfg["added_tokens_decoder"] = {
        k: v
        for k, v in cfg["added_tokens_decoder"].items()
        if k in ["0", "1", "2", "3"]
    }
    cfg["additional_special_tokens"] = []
    with open(f"{TKN_DIR}/tokenizer_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    # os.remove(f"{TKN_DIR}/tokenizer.json") # this one does not exist
    # this contains added tokens: language codes and mask
    os.remove(f"{TKN_DIR}/added_tokens.json")
    os.remove(f"{TKN_DIR}/special_tokens_map.json")
    os.remove(f"{TKN_DIR}/sentencepiece.bpe.model")
    shutil.copy(new_spm_path, f"{TKN_DIR}/sentencepiece.bpe.model")
 
    new_tokenizer = NllbTokenizer.from_pretrained(
        TKN_DIR,
        additional_special_tokens=sorted(FAIRSEQ_LANGUAGE_CODES + new_lang_codes),
    )
    return new_tokenizer
 
 
print("Creating corpus and counting chars in it")
all_texts = df["lez"].dropna().tolist()
all_text_normalized = [text_preprocessor.preprocess(t) for t in tqdm(all_texts)]
 
chars_cnt = Counter(c for t in all_text_normalized for c in t)
required_chars = ''.join([
    k for k, v in chars_cnt.most_common() 
    if v >= 3 and k not in ' '
])
 
all_texts_file = 'lez_texts_plain.txt'
SPM_PREFIX = 'spm_lez_16k'
with open(all_texts_file, 'w') as f:
    for i, text in enumerate(all_texts):
        print(text, file=f)
 
print("Tokenizer training")
spm.SentencePieceTrainer.train(
    input=all_texts_file,
    model_prefix=SPM_PREFIX,
    vocab_size=13398,  
    character_coverage=1,
    num_threads=16,
    train_extremely_large_corpus=False,
    add_dummy_prefix=False,
    max_sentencepiece_length=128,
    max_sentence_length=4192*4,
    pad_id=0,
    eos_id=1,
    unk_id=2,
    bos_id=-1,
    required_chars=required_chars,
)
 
 
print("Adding missing tokens to NLLB tokenizer and saving result")
tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
sp_trained = spm.SentencePieceProcessor(model_file=f'{SPM_PREFIX}.model')
added_spm = sp_pb2_model.ModelProto()
added_spm.ParseFromString(sp_trained.serialized_model_proto())
old_spm = sp_pb2_model.ModelProto()
old_spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())
 
nllb_tokens_set = {p.piece for p in old_spm.pieces}
prev_min_score = old_spm.pieces[-1].score
for p in added_spm.pieces:
    piece = p.piece
    if p.type != 1:
        continue
    if piece not in nllb_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = p.score + prev_min_score
        old_spm.pieces.append(new_p)
 
NEW_SPM_NAME = 'spm_nllb_lez_268k.model'
with open(NEW_SPM_NAME, 'wb') as f:
    f.write(old_spm.SerializeToString())
 
 
print("Reloading NLLB tokenizer and resizing model")
model_name = 'facebook/nllb-200-distilled-600M'
tokenizer_old = NllbTokenizer.from_pretrained(model_name)
# tokenizer = NllbTokenizer.from_pretrained(model_name, vocab_file=NEW_SPM_NAME)
 
tokenizer = update_nllb_tokenizer(tokenizer_old, NEW_SPM_NAME, new_lang_codes=["lez_Cyrl"])
 
# Checking tokenizer updates
print(f"Tokenizer length after adding 'lez_Cyrl': {len(tokenizer)}")
 
 
# Loading and resizing the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
 
# Re-initializing the new embeddings
added_vocab = set(tokenizer.get_vocab()).difference(set(tokenizer_old.get_vocab()))
for t in tqdm(added_vocab):
    tt = tokenizer_old(t, add_special_tokens=False).input_ids
    if len(tt) == 0:
        tt = [tokenizer_old.unk_token_id]
    idx = tokenizer.convert_tokens_to_ids(t)
    model.model.shared.weight.data[idx] = model.model.shared.weight.data[tt].mean(0)



reinit_model_path = "models/re-init"

model.save_pretrained(reinit_model_path)
tokenizer.save_pretrained(reinit_model_path)
print("DONE")