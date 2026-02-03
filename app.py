import sys
import numpy as np
import joblib
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Использование: python app.py path/to/x_data.npy", file=sys.stderr)
        print("Пример: python app.py parsing_analysis/x_data.npy", file=sys.stderr)
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Файл не найден: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # 1. Загружаем модель и сопоставления
        model = joblib.load('resources/model.pkl')
        scaler = joblib.load('resources/scaler.pkl')
        features = joblib.load('resources/features.pkl')
        feature_mapping = joblib.load('resources/feature_mapping.pkl')
        
        print(f"Загружена модель с {len(features)} признаками", file=sys.stderr)
        
        # 2. Загружаем данные
        x_data = np.load(input_path)
        print(f"Загружено {x_data.shape[0]} записей, {x_data.shape[1]} признаков", file=sys.stderr)
        
        # 3. Собираем индексы нужных признаков
        feature_indices = []
        for feature in features:
            if feature in feature_mapping:
                idx = feature_mapping[feature]
                if idx < x_data.shape[1]:
                    feature_indices.append(idx)
                else:
                    print(f"Предупреждение: признак {feature} (индекс {idx}) за пределами данных", 
                          file=sys.stderr)
                    feature_indices.append(0)  # используем первый признак как заглушку
            else:
                print(f"Предупреждение: признак {feature} не найден в маппинге", file=sys.stderr)
                feature_indices.append(0)
        
        print(f"Используются индексы признаков: {feature_indices}", file=sys.stderr)
        
        # 4. Выбираем только нужные признаки
        x_selected = x_data[:, feature_indices]
        
        # 5. Очистка от NaN
        x_selected = np.nan_to_num(x_selected, nan=0.0)
        
        # 6. Масштабируем
        x_scaled = scaler.transform(x_selected)
        
        # 7. Предсказываем
        predictions = model.predict(x_scaled)
        
        # 8. Корректируем (зарплата не может быть отрицательной)
        predictions = np.maximum(predictions, 10000)
        predictions = np.minimum(predictions, 5000000)
        
        # 9. Выводим результаты (только зарплаты, по одной на строку)
        for salary in predictions:
            print(f"{salary:.2f}")
            
        # 10. Статистика (только в stderr, чтобы не мешать основному выводу)
        print(f"\nСтатистика предсказаний:", file=sys.stderr)
        print(f"  Всего: {len(predictions)}", file=sys.stderr)
        print(f"  Средняя: {np.mean(predictions):.2f} руб.", file=sys.stderr)
        print(f"  Медиана: {np.median(predictions):.2f} руб.", file=sys.stderr)
        print(f"  Минимум: {np.min(predictions):.2f} руб.", file=sys.stderr)
        print(f"  Максимум: {np.max(predictions):.2f} руб.", file=sys.stderr)
        
    except FileNotFoundError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        print("Сначала обучите модель: python simple_train.py", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()