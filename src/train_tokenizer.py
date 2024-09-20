import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

from src.dataset import TextPreprocessor

"""
python -m src.train_tokenizer
"""

# this code is adapted from  the Stopes repo of the NLLB team
# https://github.com/facebookresearch/stopes/blob/main/stopes/pipelines/monolingual/monolingual_line_processor.py#L21

def load_datasets(train_path: str, test_path: str, val_path: str) -> pd.DataFrame:
    """Loads and concatenates training, testing, and validation datasets into a single DataFrame."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_val = pd.read_csv(val_path)
    return pd.concat([df_train, df_test, df_val], ignore_index=True)

df = load_datasets("data/train.csv", "data/test.csv", "data/val.csv")

# Initialize text preprocessor
text_preprocessor = TextPreprocessor()

def update_nllb_tokenizer(old_tokenizer: NllbTokenizer, new_spm_path: str, new_lang_codes: list[str]) -> NllbTokenizer:
    """
    Update the NLLB tokenizer with a new SentencePiece model and additional language codes.
    
    Args:
        old_tokenizer: The original NLLB tokenizer.
        new_spm_path: Path to the new SentencePiece model file.
        new_lang_codes: List of new language codes to add.
    
    Returns:
        Updated NllbTokenizer.
    """
    TKN_DIR = Path("old_tokenizer")  # Temporary directory for tokenizer storage
    old_tokenizer.save_pretrained(TKN_DIR)
    
    # Modify tokenizer configuration
    config_path = TKN_DIR / "tokenizer_config.json"
    with config_path.open("r") as f:
        cfg = json.load(f)
    cfg["added_tokens_decoder"] = {k: v for k, v in cfg["added_tokens_decoder"].items() if k in ["0", "1", "2", "3"]}
    cfg["additional_special_tokens"] = []
    
    with config_path.open("w") as f:
        json.dump(cfg, f, indent=2)

    # Clean up old added tokens and sentencepiece model
    (TKN_DIR / "added_tokens.json").unlink()
    (TKN_DIR / "special_tokens_map.json").unlink()
    (TKN_DIR / "sentencepiece.bpe.model").unlink()

    # Copy new SentencePiece model
    shutil.copy(new_spm_path, TKN_DIR / "sentencepiece.bpe.model")

    # Load new tokenizer with additional language codes
    new_tokenizer = NllbTokenizer.from_pretrained(
        TKN_DIR,
        additional_special_tokens=sorted(FAIRSEQ_LANGUAGE_CODES + new_lang_codes),
    )
    return new_tokenizer
 

# Main script execution
if __name__ == "__main__":
    MODEL_ID = 'facebook/nllb-200-distilled-600M'

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
    with Path(all_texts_file).open('w') as f:
        for _, text in enumerate(all_texts):
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
    tokenizer = NllbTokenizer.from_pretrained(MODEL_ID)
    sp_trained = spm.SentencePieceProcessor(model_file=f'{SPM_PREFIX}.model')
    added_spm = sp_pb2_model.ModelProto()
    added_spm.ParseFromString(sp_trained.serialized_model_proto())

    # Merge new tokens with old tokenizer
    old_spm = sp_pb2_model.ModelProto()
    old_spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())
    nllb_tokens_set = {p.piece for p in old_spm.pieces}
    prev_min_score = old_spm.pieces[-1].score

    for p in added_spm.pieces:
        if p.type == 1 and p.piece not in nllb_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = p.piece
            new_p.score = p.score + prev_min_score
            old_spm.pieces.append(new_p)

    NEW_SPM_NAME = 'spm_nllb_lez_268k.model'
    with Path(NEW_SPM_NAME).open('wb') as f:
        f.write(old_spm.SerializeToString())

    # Reload tokenizer and update the model
    print("Reloading NLLB tokenizer and resizing model")
    tokenizer_old = NllbTokenizer.from_pretrained(MODEL_ID)
    tokenizer = update_nllb_tokenizer(tokenizer_old, NEW_SPM_NAME, new_lang_codes=["lez_Cyrl"])

    # Check tokenizer updates
    print(f"Tokenizer length after adding 'lez_Cyrl': {len(tokenizer)}")

    # Load and resize the model
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.resize_token_embeddings(len(tokenizer))

    # Reinitialize new embeddings
    print("Re-initializing new embeddings")
    added_vocab = set(tokenizer.get_vocab()).difference(set(tokenizer_old.get_vocab()))
    for t in tqdm(added_vocab):
        tt = tokenizer_old(t, add_special_tokens=False).input_ids
        if len(tt) == 0:
            tt = [tokenizer_old.unk_token_id]
        idx = tokenizer.convert_tokens_to_ids(t)
        model.model.shared.weight.data[idx] = model.model.shared.weight.data[tt].mean(0)

    # Save the updated model and tokenizer
    reinit_model_path = "models/re-init"
    model.save_pretrained(reinit_model_path)
    tokenizer.save_pretrained(reinit_model_path)
    print("DONE")
