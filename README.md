### NLLB-Tuning

### Raw experimental solution
Download [bible.csv](https://huggingface.co/datasets/leks-forever/bible-lezghian-russian) and place it in the [data](data) folder

Install requirements:
```bash
pip install poetry
poetry install
```

Scripts:    
[src/utils.py](src/utils.py) - split prepaired df to train/test/val     
[src/train_tokenizer.py](src/train_model.py) - update tokenizer and model embeddings according to tokenizer     
[src/train_model.py](src/train_model.py) - finetune NLLB model      
[src/test_model.py](src/test_model.py) - test NLLB model  using BLEU and chrF       
[src/predict_model.py](src/predict_model.py) - predict NLLB model   
[src/dataset.py](src/dataset.py) - PyTorch train/test datasets

Notebooks:  
[notebooks/convert_to_transformers.ipynb](notebooks/convert_to_transformers.ipynb) -  convert Lighting ckpt to transformers       
[notebooks/predict_model.ipynb](notebooks/predict_model.ipynb) - predict NLLB model (Recommend)

Logging:
```bash
tensorboard --logdir tb_logs/
```
