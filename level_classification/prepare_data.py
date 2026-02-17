import pandas as pd
import re
import numpy as np
from .config import DATA_PATH, IT_KEYWORDS

def load_raw_data():
    """Загружает данные из CSV."""
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(f"Загружено строк: {len(df)}")
    return df

def filter_it_resumes(df):
    """
    Оставляет только те резюме, где в желаемой должности встречаются ключевые слова IT.
    """
    job_col = 'Ищет работу на должность:'
    # Приводим к нижнему регистру и заполняем пропуски
    df[job_col] = df[job_col].fillna('').astype(str).str.lower()
    # Создаём маску: True, если хотя бы одно ключевое слово есть в строке
    mask = df[job_col].apply(lambda x: any(keyword in x for keyword in IT_KEYWORDS))
    df_it = df[mask].copy().reset_index(drop=True)
    print(f"Найдено IT-резюме: {len(df_it)}")
    return df_it

def extract_experience(exp_text):
    """
    Извлекает общий стаж в годах из текстового поля 'Опыт (двойное нажатие для полной версии)'.
    Пример: 'Опыт работы 6 лет 1 месяц ...' -> 6.08
    """
    if pd.isna(exp_text):
        return np.nan
    text = str(exp_text)
    # Ищем шаблон "Опыт работы X лет Y месяцев"
    pattern = r'Опыт работы\s*(?:(\d+)\s*(?:лет|год|года))?\s*(?:(\d+)\s*месяц)?'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        years = match.group(1)
        months = match.group(2)
        total = 0
        if years:
            total += int(years)
        if months:
            total += int(months) / 12.0
        return round(total, 2)
    return np.nan

def add_target_level(df):
    """
    Добавляет колонку 'level' на основе опыта:
    - junior: опыт < 1 года
    - middle: 1 <= опыт < 3
    - senior: опыт >= 3
    """
    exp_col = 'Опыт (двойное нажатие для полной версии)'
    df['experience_years'] = df[exp_col].apply(extract_experience)
    # Удаляем строки, где не удалось определить опыт
    df = df.dropna(subset=['experience_years']).reset_index(drop=True)
    df['level'] = df['experience_years'].apply(
        lambda x: 'junior' if x < 1 else ('middle' if x < 3 else 'senior')
    )
    print("Распределение по уровням:")
    print(df['level'].value_counts())
    return df

if __name__ == '__main__':
    df = load_raw_data()
    df_it = filter_it_resumes(df)
    df_it = add_target_level(df_it)
    # Сохраняем промежуточный результат (опционально)
    df_it.to_csv('parsing_analysis/it_resumes.csv', index=False)
    print("Промежуточный файл сохранён: parsing_analysis/it_resumes.csv")