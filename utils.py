from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df_path: str = "data/bible.csv",  save_dir: str = "data"):
    df = pd.read_csv(df_path)
    
    ru = df['text_ru'].values
    lezi = df['text_lez'].values

    # Шаг 1: Разделить данные на train и temp (сначала отделить 90% для train)
    ru_train, ru_temp, lez_train, lez_temp = train_test_split(
        ru, lezi, test_size=0.1, random_state=42
    )

    # Шаг 2: Разделить temp данные на val и test (оставшиеся 10% делим пополам по 5% на каждый)
    ru_val, ru_test, lez_val, lez_test = train_test_split(
        ru_temp, lez_temp, test_size=0.5, random_state=42
    )


    df_train = pd.DataFrame({
        'ru': ru_train,
        'lez': lez_train,
    })

    df_val = pd.DataFrame({
        'ru': ru_val,
        'lez': lez_val,
    })

    df_test = pd.DataFrame({
        'ru': ru_test,
        'lez': lez_test,
    })

    save_dir = Path(save_dir)

    df_train.to_csv(save_dir / 'train.csv', index=False)
    df_val.to_csv(save_dir / 'val.csv', index=False)
    df_test.to_csv(save_dir / 'test.csv', index=False)


if __name__ == "__main__":
    split_data()