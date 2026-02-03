import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 1. Загружаем данные из processed_data.csv
print("Загрузка данных...")
data = pd.read_csv('parsing_analysis/processed_data.csv')

# 2. Берем salary как целевую переменную
if 'salary' not in data.columns:
    print("Ошибка: нет колонки 'salary'")
    exit(1)

# 3. Признаки для модели (без утечки данных)
#    Убираем salary, salary_log, high_salary
features = ['is_male', 'age', 'relocation', 'business_trips', 
            'exp_years', 'exp_months', 'total_exp', 'edu_level', 
            'has_car', 'full_time', 'full_day', 'company_length', 
            'position_length', 'is_it', 'recent_update', 
            'age_exp_interaction', 'edu_exp_interaction']

# 4. Проверяем, какие признаки есть
available_features = [f for f in features if f in data.columns]
print(f"Используем {len(available_features)} признаков:")
for feat in available_features:
    print(f"  - {feat}")

# 5. Подготовка данных
X = data[available_features]
y = data['salary']

# Удаляем NaN
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

print(f"\nДанные после очистки: {len(X)} строк")

# 6. Масштабирование и обучение
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# 7. Сохраняем модель
os.makedirs('resources', exist_ok=True)
joblib.dump(model, 'resources/model.pkl')
joblib.dump(scaler, 'resources/scaler.pkl')
joblib.dump(available_features, 'resources/features.pkl')

# 8. Сохраняем также маппинг признаков для app.py
feature_mapping = {
    'is_male': 0,
    'age': 1,
    'relocation': 3,
    'business_trips': 4,
    'exp_years': 5,
    'exp_months': 6,
    'total_exp': 7,
    'edu_level': 8,
    'has_car': 9,
    'full_time': 10,
    'full_day': 11,
    'company_length': 12,
    'position_length': 13,
    'is_it': 14,
    'recent_update': 15,
    'age_exp_interaction': 17,
    'edu_exp_interaction': 18
}
joblib.dump(feature_mapping, 'resources/feature_mapping.pkl')

print(f"\nМодель обучена и сохранена в resources/")
print(f"Использовано признаков: {len(available_features)}")
print(f"Пример зарплаты: {y.iloc[0]:.2f} руб.")
print(f"Коэффициент R² на тренировочных данных: {model.score(X_scaled, y):.4f}")