import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Добавляем путь к проекту, чтобы импортировать config (если запускаем напрямую)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from level_classification.config import (
    MODEL_SAVE_PATH, SCALER_SAVE_PATH, ENCODER_SAVE_PATH,
    TECH_LIST, MILLION_CITIES
)
from level_classification.prepare_data import load_raw_data, filter_it_resumes, add_target_level

# ==================== Функции инжиниринга признаков ====================

def clean_salary(sal_str):
    if pd.isna(sal_str):
        return np.nan
    sal_str = str(sal_str)
    digits = re.sub(r'[^\d]', ' ', sal_str).strip()
    if digits:
        first_num = digits.split()[0]
        try:
            val = int(first_num)
            # Если число маленькое (вероятно, в тысячах) – умножаем на 1000
            if val < 1000:
                val *= 1000
            return val
        except:
            return np.nan
    return np.nan

def extract_age(gender_age_str):
    """Из строки 'Мужчина , 42 года , родился ...' извлекает возраст."""
    if pd.isna(gender_age_str):
        return np.nan
    match = re.search(r'(\d+)\s*(?:года|лет|год)', str(gender_age_str))
    if match:
        return int(match.group(1))
    return np.nan

def city_group(city_str):
    """Группирует города: Москва, СПб, миллионник, другой."""
    if pd.isna(city_str):
        return 'Другой'
    city = str(city_str).split(',')[0].strip()
    if city == 'Москва':
        return 'Москва'
    elif city == 'Санкт-Петербург':
        return 'СПб'
    elif city in MILLION_CITIES:
        return 'Миллионник'
    else:
        return 'Другой'

def extract_skills_block(exp_text):
    """Из поля с опытом извлекает блок 'Ключевые навыки'."""
    if pd.isna(exp_text):
        return ''
    text = str(exp_text)
    match = re.search(
        r'Ключевые навыки(.*?)(Обо мне|Опыт вождения|Возникли неполадки|$)',
        text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return ''

def count_skills(block):
    """Считает количество навыков (по запятым)."""
    if not block:
        return 0
    block = re.sub(r'\s+', ' ', block)
    parts = [p.strip() for p in block.split(',') if p.strip()]
    return len(parts)

def tech_presence(block, tech):
    """Проверяет, встречается ли технология в блоке навыков."""
    if not block:
        return 0
    return 1 if re.search(r'\b' + re.escape(tech) + r'\b', block, re.IGNORECASE) else 0

def build_features(df):
    df = df.copy()
    
    # 1. Зарплата
    df['salary_clean'] = df['ЗП'].apply(clean_salary)
    # Приводим к рублям (если число < 1000 – умножаем на 1000)
    # Это уже сделано внутри clean_salary, поэтому просто фильтруем
    df = df[(df['salary_clean'] >= 30000) & (df['salary_clean'] <= 10_000_000)]
    df['log_salary'] = np.log1p(df['salary_clean'])

    # 2. Возраст
    df['age'] = df['Пол, возраст'].apply(extract_age)
    df = df.dropna(subset=['age'])

    # 3. Город
    df['city_group'] = df['Город'].apply(city_group)
    city_dummies = pd.get_dummies(df['city_group'], prefix='city', drop_first=True)

    # 4. Возраст начала карьеры
    df['career_start_age'] = df['age'] - df['experience_years']

    # 5. Навыки
    df['skills_block'] = df['Опыт (двойное нажатие для полной версии)'].apply(extract_skills_block)
    df['skills_count'] = df['skills_block'].apply(count_skills)

    tech_features = {}
    for tech in TECH_LIST:
        col_name = 'tech_' + tech.replace('+', 'p').replace('#', 'sharp').replace(' ', '_')
        tech_features[col_name] = df['skills_block'].apply(lambda x: tech_presence(x, tech))
    tech_df = pd.DataFrame(tech_features, index=df.index)

    # 6. Занятость и график
    df['employment_full'] = df['Занятость'].str.contains('полная', na=False).astype(int)
    df['employment_part'] = df['Занятость'].str.contains('частичная', na=False).astype(int)
    df['schedule_full'] = df['График'].str.contains('полный день', na=False).astype(int)
    df['schedule_flex'] = df['График'].str.contains('гибкий', na=False).astype(int)
    df['schedule_remote'] = df['График'].str.contains('удаленная', na=False).astype(int)

    # Сборка X
    base_features = pd.DataFrame({
        'log_salary': df['log_salary'],
        'age': df['age'],
        'career_start_age': df['career_start_age'],
        'skills_count': df['skills_count'],
        'employment_full': df['employment_full'],
        'employment_part': df['employment_part'],
        'schedule_full': df['schedule_full'],
        'schedule_flex': df['schedule_flex'],
        'schedule_remote': df['schedule_remote']
    }, index=df.index)

    X = pd.concat([base_features, city_dummies, tech_df], axis=1)
    X = X.dropna()  # удаляем оставшиеся пропуски (если есть)
    y = df.loc[X.index, 'level']

    return X, y, list(X.columns)

# ==================== Основная функция ====================

def main():
    # 1. Загрузка и подготовка данных
    print("Загрузка исходных данных...")
    df_raw = load_raw_data()
    df_it = filter_it_resumes(df_raw)
    df_it = add_target_level(df_it)

    # 2. Построение признаков
    print("Построение признаков...")
    X, y, feature_names = build_features(df_it)

    # 3. Кодирование целевой переменной
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # 5. Масштабирование числовых признаков
    numeric_cols = ['log_salary', 'age', 'career_start_age', 'skills_count']
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # 6. Обучение модели (Random Forest с балансировкой классов)
    print("Обучение модели...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # 7. Оценка
    y_pred = model.predict(X_test_scaled)
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Матрица ошибок
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join('level_classification', 'confusion_matrix.png'))
    plt.show()

    # 8. Важность признаков
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    print("\nТоп-10 важных признаков:")
    print(feat_imp.head(10).to_string(index=False))

    # Сохраняем важность в CSV
    feat_imp.to_csv(os.path.join('level_classification', 'feature_importance.csv'), index=False)

    # 9. Сохранение модели и трансформеров
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(le, ENCODER_SAVE_PATH)
    print(f"\nМодель сохранена: {MODEL_SAVE_PATH}")
    print(f"Scaler сохранён: {SCALER_SAVE_PATH}")
    print(f"LabelEncoder сохранён: {ENCODER_SAVE_PATH}")

if __name__ == '__main__':
    main()