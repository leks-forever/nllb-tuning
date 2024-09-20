from typing import Optional, Union

import torch
import typer
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

from src.train_model import LightningModel

"""
python -m src.predict_model
"""

def predict(    
    model_path: str = "models/re-init", 
    tokenizer_path: str = "models/re-init",
    tokenizer_vocab_file_path: str = "models/re-init/sentencepiece.bpe.model", 
    ckpt_path: Optional[Union[str, None]] = "models/it_1/epoch=17-val_loss=1.48184.ckpt",
    sentence: str = "Римдин аскерар ва гьакӀни чӀехи хахамрини  фарисейри ракъурнавай нуькерар Ягьуд галаз багъдиз атана. Абурув виридав яракьар, чирагъар ва шемгьалар гвай."
):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = NllbTokenizer.from_pretrained(tokenizer_path, vocab_file = tokenizer_vocab_file_path)

    print(model)

    # GPU loading
    # model = LightningModel.load_from_checkpoint(ckpt_path)

    # Different device loading
    ckpt = torch.load(ckpt_path, map_location=torch.device("cuda:1"))
    model = LightningModel(model, tokenizer)
    model.load_state_dict(ckpt['state_dict'])

    translation = model.predict(sentence)

    print(translation)

if __name__ == "__main__":
    typer.run(predict)
