from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import sacrebleu
import torch
import typer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import CombinedLoader  #, rank_zero_only

# from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    # get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

# from pytorch_lightning.strategies import FSDPStrategy
from transformers.optimization import Adafactor

from dataset import CollateFn, LangCollateFn, ThisDataset

torch.set_float32_matmul_precision('medium')

# TODO: Loading via config

"""
python -m models.scripts.train_model
"""

class LightningModel(pl.LightningModule):
    def __init__(self, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer = None) -> None:
        super(LightningModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.temp_values = []
        
        self.bleu_calc = sacrebleu.BLEU()
        self.chrf_calc = sacrebleu.CHRF(word_order=2)  # this metric is called ChrF++

    def predict(self, text, src_lang='lez_Cyrl', tgt_lang='rus_Cyrl', a=32, b=3, max_input_length=1024, num_beams=1, **kwargs):
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length)
        result = self.model.generate(
            **inputs.to(self.model.device),
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
            num_beams=num_beams,
            **kwargs
        )
        return self.tokenizer.batch_decode(result, skip_special_tokens=True)

    def forward(self, src):
        out = self.model(**src)

        return out

    def training_step(self, batch):
        src, tgt = batch.values()
        
        loss = self.model(**src, labels=tgt.input_ids).loss
        self.log("train_loss", loss, prog_bar=True)

        return loss
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        src, tgt = batch.values()
        
        loss = self.model(**src, labels=tgt.input_ids).loss
        # for CombinedDataloader metric is [val_loss/dataloader_idx_0', 'val_loss/dataloader_idx_1']
        if "val_loss" not in self.trainer.callback_metrics: # Do not delete this! # TODO: Think about it
            self.trainer.callback_metrics["val_loss"] = 0 # init loading
            # self.log("val_loss", loss, sync_dist=True, add_dataloader_idx=False) # ckpt loading
            # self.logger.experiment.add_scalar("val_loss", loss, global_step=self.global_step)
        else:
            self.log("val_loss", loss, sync_dist=True)

        return loss
    
    def on_validation_epoch_end(self):
        val_losses = [self.trainer.callback_metrics.get(f'val_loss/dataloader_idx_{i}') for i in range(len(self.trainer.val_dataloaders))]

        if all(val_loss is not None for val_loss in val_losses):
            avg_loss = torch.mean(torch.stack(val_losses))
            self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)

            return avg_loss
    
    def test_step(self, batch, batch_idx):
        inputs, forced_bos_token_id, max_new_tokens, num_beams, tgt_text, src_lang, tgt_lang = batch.values()
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        
        result = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )

        result_text = self.tokenizer.batch_decode(result, skip_special_tokens=True)

        self.temp_values.append((result_text[0], tgt_text[0]))
    
    def on_test_epoch_end(self):
        temp_values = self.all_gather(self.temp_values)

        if self.trainer.is_global_zero:
            result_texts, tgt_texts = [], []
            for result_text, tgt_text in temp_values:
                result_texts.append(result_text)
                tgt_texts.append([tgt_text])

            bleu_score = self.bleu_calc.corpus_score(result_texts, tgt_texts).score
            chrf_score = self.chrf_calc.corpus_score(result_texts, tgt_texts).score

            print(self.bleu_calc.corpus_score(result_texts, tgt_texts))
            print(self.chrf_calc.corpus_score(result_texts, tgt_texts))

            self.log("BLEU", bleu_score)
            self.log("chrF", chrf_score)

            self.temp_values.clear()

            return {"BLEU": bleu_score, "chrF": chrf_score}

    def configure_optimizers(self):
        optimizer = Adafactor(
            [p for p in self.model.parameters() if p.requires_grad],
            scale_parameter=False,
            relative_step=False,
            lr=1e-4,
            clip_threshold=1.0,
            weight_decay=1e-3,
        )

        # optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr = 2e-4, betas = (0.9, 0.98))        
        # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=self.trainer.max_steps)


        return {
            'optimizer': optimizer,
            "lr_scheduler":{
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step"
            }
        }
    
    def convert_ckpt_to_tranformers(self, save_directory: str):
        self.model.save_pretrained(save_directory)

# @rank_zero_only
# def prepare_model_and_tokenizer(model_name, vocab_file, reinit_model_path):
#     # TODO: Refactor this

#     # loading the tokenizers
#     tokenizer_old = NllbTokenizer.from_pretrained(model_name)
#     tokenizer = NllbTokenizer.from_pretrained(model_name, vocab_file=vocab_file)

#     added_vocab = set(tokenizer.get_vocab()).difference(set(tokenizer_old.get_vocab()))

#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     model.resize_token_embeddings(len(tokenizer))

#     # re-initializing the new embeddings
#     for t in tqdm(added_vocab, desc = "Re-initializing new embeddings"):
#         tt = tokenizer_old(t, add_special_tokens=False).input_ids
#         if len(tt) == 0:
#             tt = [tokenizer_old.unk_token_id]
#         idx = tokenizer.convert_tokens_to_ids(t)
#         model.model.shared.weight.data[idx] = model.model.shared.weight.data[tt].mean(0)

#     model.save_pretrained(reinit_model_path)
#     tokenizer.save_pretrained(reinit_model_path)

#     del model, tokenizer
    

def train(
    batch_size: int = 16, 
    checkpoints_dir: str = "models/it_1", 
    checkpoint_path: Optional[str] = None,  
    model_name: str = "facebook/nllb-200-distilled-600M", 
    vocab_file: str = "models/re-init/sentencepiece.bpe.model",
    reinit_model_path: str = "models/re-init",
    train_df_path: str = "data/train.csv",
    val_df_path: str = "data/val.csv"
):
    train_df, val_df = pd.read_csv(train_df_path), pd.read_csv(val_df_path)
    
    logger = TensorBoardLogger("./tb_logs", version = "it_1")

    checkpoint_callback = ModelCheckpoint(
        dirpath = checkpoints_dir,
        monitor='val_loss',
        filename='{epoch:02d}-{val_loss:.5f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
        
    model = AutoModelForSeq2SeqLM.from_pretrained(reinit_model_path)
    model.train()
    tokenizer = NllbTokenizer.from_pretrained(reinit_model_path, vocab_file=vocab_file)

    train_dataloader = DataLoader(ThisDataset(train_df, random=True), batch_size=batch_size, shuffle=True, collate_fn=CollateFn(tokenizer), num_workers=14)

    val_ru_lez_dataloader = DataLoader(ThisDataset(val_df, random=False), batch_size=batch_size, collate_fn=LangCollateFn(tokenizer, src_lang='rus_Cyrl', tgt_lang='lez_Cyrl'), num_workers=14)
    val_lez_ru_dataloader = DataLoader(ThisDataset(val_df, random=False), batch_size=batch_size, collate_fn=LangCollateFn(tokenizer, src_lang='lez_Cyrl', tgt_lang='rus_Cyrl'), num_workers=14)

    val_dataloaders = CombinedLoader(iterables=[val_ru_lez_dataloader, val_lez_ru_dataloader], mode="sequential")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    lightning_model = LightningModel(model)

    trainer = Trainer(max_steps=110000, callbacks=[checkpoint_callback, lr_monitor], logger=logger, devices = [1], log_every_n_steps=1, val_check_interval = 388, precision="32-true") # check_val_every_n_epoch=1 val_check_interval=4482,
    trainer.fit(model=lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloaders, ckpt_path=checkpoint_path)



if __name__ == "__main__":
    typer.run(train)

    