
from typing import Optional, Union

import pandas as pd
import typer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

from dataset import TestCollateFn, ThisDataset
from src.train_model import LightningModel

"""
python -m src.test_model
"""

def test(
    df_path: str = "data/test.csv", 
    model_path: str = "models/re-init", 
    tokenizer_path: str = "models/re-init",
    tokenizer_vocab_file_path: str = "models/re-init/sentencepiece.bpe.model", 
    ckpt_path: Optional[Union[str, None]] = "models/it_1/epoch=17-val_loss=1.48184.ckpt",
):
    test_df = pd.read_csv(df_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    tokenizer = NllbTokenizer.from_pretrained(tokenizer_path, vocab_file = tokenizer_vocab_file_path)

    test_dataset = ThisDataset(test_df, random=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=14, collate_fn = TestCollateFn(tokenizer, 'rus_Cyrl', 'lez_Cyrl', num_beams=1))

    logger = TensorBoardLogger("./tb_logs", version="it_1", name = "test")

    lightning_model = LightningModel(model, tokenizer)

    trainer = Trainer(logger=logger, devices = [1], log_every_n_steps=1, precision="32-true")
    trainer.test(model=lightning_model, dataloaders=test_dataloader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    typer.run(test)