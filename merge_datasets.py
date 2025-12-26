import os
import pandas as pd
from pathlib import Path
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Удаляем лишние пробелы и странные символы в начале
    text = text.strip()
    # Удаляем BOM если есть
    text = text.replace('\ufeff', '')
    return text

def merge_data_refined(input_dir, output_file):
    all_data = []
    input_path = Path(input_dir)
    
    # Конфигурация для каждого файла (на основе анализа head)
    file_configs = {
        'lek_translator_output28072025.csv': {'ru': 'Text', 'lez': 'Translation', 'sep': ','},
        'merged_texts_translations.csv': {'ru': 'text_ru', 'lez': 'text_lez', 'sep': ','},
        'quran_qusar.csv': {'ru': 'text_ru', 'lez': 'text_lez', 'sep': ','},
        'The_Secret_of_Third_Planet_Rus_Lez.csv': {'ru': 'text_ru', 'lez': 'text_lez', 'sep': ','},
        'corpus_of_cnal_lezgian_to_russian.csv': {'ru': 'text_ru', 'lez': 'text_lez', 'sep': ','},
        'num_lez_rus.csv': {'ru': 'text_ru', 'lez': 'text_lez', 'sep': ','}, # В подпапке
        'bible.csv': {'ru': 'text_ru', 'lez': 'text_lez', 'sep': ','},
        'translations_fix_sep.csv': {'ru': 'text_ru', 'lez': 'text_lez', 'sep': ','},
        'lezgigazet_ru_archives_337371.csv': {'ru': 'text_ru', 'lez': 'text_lez', 'sep': ','},
        'bubadin_vesi_and_chalakay_ballada.csv': {'ru': 'ru', 'lez': 'lz', 'sep': ';'},
        '[PRIVATE]parallel_lermontov_ashik_kerib.csv': {'ru': 'rus', 'lez': 'lez', 'sep': ','},
        'rus_lez_razgovornik2.csv': {'ru': 'Title', 'lez': 'Description', 'sep': ','},
        'quran_lz_ru.csv': {'ru': 'rus', 'lez': 'lez', 'sep': ','}
    }

    # Отдельно обработаем файлы с другим сепаратором
    semicolon_files = ['num_lez_rus.csv'] # Тот что в корне

    for file in input_path.rglob('*.csv'):
        fname = file.name
        print(f"Processing: {file}")
        
        config = file_configs.get(fname)
        
        # Специальный случай для файлов с одинаковыми именами в разных папках
        if fname == 'num_lez_rus.csv' and file.parent.name == 'extracted':
            config = {'ru': 'rus', 'lez': 'lez', 'sep': ';'}
        
        if config:
            sep = config.get('sep', ',')
            try:
                df = pd.read_csv(file, sep=sep, on_bad_lines='skip', encoding='utf-8')
                # Удаляем BOM из названий колонок
                df.columns = [c.replace('\ufeff', '') for c in df.columns]
                
                if config['ru'] in df.columns and config['lez'] in df.columns:
                    df = df.rename(columns={config['ru']: 'ru', config['lez']: 'lez'})
                    df = df[['ru', 'lez']]
                    df['ru'] = df['ru'].apply(clean_text)
                    df['lez'] = df['lez'].apply(clean_text)
                    df = df[(df['ru'] != "") & (df['lez'] != "")]
                    all_data.append(df)
                    print(f"  Success: {len(df)} rows")
                else:
                    print(f"  Error: Expected columns {config} not found. Available: {df.columns.tolist()}")
            except Exception as e:
                print(f"  Failed to read {file}: {e}")
        else:
            print(f"  No config for {fname}, skipping.")

    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal rows before cleaning: {len(merged_df)}")
        
        # 1. Удаляем полные дубликаты (ru + lez)
        merged_df = merged_df.drop_duplicates(subset=['ru', 'lez'])
        
        # 2. Удаляем дубликаты по 'ru' (если перевод разный, оставляем первый встреченный)
        # В идеале тут нужен более умный фильтр, но для начала так
        # merged_df = merged_df.drop_duplicates(subset=['ru']) 
        
        print(f"Total rows after cleaning: {len(merged_df)}")
        
        merged_df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
    else:
        print("No data matched.")

if __name__ == "__main__":
    merge_data_refined('/home/said/projects/translator/data/extracted', '/home/said/projects/translator/data/merged_dataset.csv')
