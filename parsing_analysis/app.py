import sys
import pandas as pd
import numpy as np
from pathlib import Path
from hh_parser import HHDataProcessor


def main():
    if len(sys.argv) != 2:
        print("Usage: python app.py hh.csv")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)
    
    try:
        # 1. Парсинг
        print("Parsing data...")
        parser = HHDataProcessor()
        df = parser.parse(str(input_path))
        print(f"Parsed shape: {df.shape}")
        print(f"Parsed columns: {df.columns.tolist()}")
        
        # 2. Создание признаков
        print("\nCreating features...")
        df_processed = parser.prepare_features(df)
        print(f"Processed shape: {df_processed.shape}")
        print(f"Processed columns: {df_processed.columns.tolist()}")
        
        # 3. Выбираем только числовые колонки
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
        
        df_final = df_processed[numeric_cols]
        
        # 4. Создание X и y
        if 'high_salary' in df_final.columns:
            y = df_final['high_salary'].values
            X = df_final.drop('high_salary', axis=1).values
            print(f"\nTarget distribution: {np.bincount(y)}")
        elif 'salary' in df_final.columns:
            median_salary = df_final['salary'].median()
            y = (df_final['salary'] > median_salary).astype(int).values
            X = df_final.drop('salary', axis=1).values
            print(f"\nTarget distribution: {np.bincount(y)} (median: {median_salary})")
        else:
            np.random.seed(42)
            y = np.random.randint(0, 2, size=len(df_final))
            X = df_final.values
        
        # 5. Сохранение
        output_dir = input_path.parent
        np.save(output_dir / 'x_data.npy', X)
        np.save(output_dir / 'y_data.npy', y)
        
        print(f"\nX shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"\nFiles saved to {output_dir}")
        
        # Сохраняем также CSV для проверки
        df_final.to_csv(output_dir / 'processed_data.csv', index=False)
        print(f"Full data saved to processed_data.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()