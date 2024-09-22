## Translation Examples

| **Russian Sentence**                                                                                     | **Lezgian Translation**                                                                                       |
|-----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Давай поедим пиццу, а потом еще немного погуляем!                                                           | Ша чна са тӀуьнни тӀуьна, ахпа мад са тӀимил къекъуьн патал чара жен!                                           |
| Я люблю гулять по парку ранним утром, когда воздух свежий и тишина вокруг.                                  | Заз пакамахъ, хъсан гар алаз, сагъ-саламатдиз къекъвез кӀанзава.                                               |
| Новый фильм, который мы смотрели вчера, произвёл на меня сильное впечатление                                | Чна йифиз килигай цӀийиди заз гзаф таъсир авуна.                                                                |
| В следующем году я планирую посетить несколько стран, чтобы познакомиться с их культурой.                   | Зун къведай йисуз са шумуд уьлкведиз, абурун адетралди чир хьун патал, къвез кӀанзава.                           |
| После долгого рабочего дня приятно расслабиться с книгой и чашкой чая.                                      | Гзаф кӀвалах авурдалай кьулухъ ктаб ва са гъам гваз ял акьадайвал хъсан я.                                      |

### Install requirements:
```bash
pip install poetry
poetry install
```

### Run Demo 
```bash
python app.py
```
### NLLB-Tuning

You can download the finetuned model from this [link](https://huggingface.co/leks-forever/nllb-200-distilled-600M).

### Raw experimental solution
Download [bible.csv](https://huggingface.co/datasets/leks-forever/bible-lezghian-russian) and place it in the [data](data) folder.

Scripts:    
[src/utils.py](src/utils.py) - split prepaired df to train/test/val     
[src/train_tokenizer.py](src/train_tokenizer.py) - update tokenizer and model embeddings according to tokenizer     
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
