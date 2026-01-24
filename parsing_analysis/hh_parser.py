import pandas as pd
import numpy as np
import re


class HHDataProcessor:
    def parse(self, file_path):
        # Читаем все колонки
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        print(f"Original columns: {df.columns.tolist()}")
        print(f"Original shape: {df.shape}")
        
        # Обрабатываем только первые N строк для скорости (или все)
        # df = df.head(1000)  # для тестирования
        
        resumes = []
        
        for _, row in df.iterrows():
            resume = {}
            
            # 1. Пол и возраст (первая колонка)
            if 'Пол, возраст' in df.columns:
                text = str(row['Пол, возраст']) if pd.notna(row['Пол, возраст']) else ''
                resume['is_male'] = 1 if 'Мужчина' in text else 0
                age_match = re.search(r'(\d+)\s*год', text)
                resume['age'] = int(age_match.group(1)) if age_match else np.nan
            
            # 2. Зарплата
            if 'ЗП' in df.columns:
                salary = str(row['ЗП']) if pd.notna(row['ЗП']) else ''
                salary_clean = salary.replace(' ', '').replace('\xa0', '')
                numbers = re.findall(r'\d+[.,]?\d*', salary_clean)
                resume['salary'] = float(numbers[0].replace(',', '.')) if numbers else np.nan
            
            # 3. Город и готовность к переезду
            if 'Город' in df.columns:
                loc = str(row['Город']) if pd.notna(row['Город']) else ''
                resume['relocation'] = 1 if 'готов к переезду' in loc.lower() else 0
                resume['business_trips'] = 1 if 'готов к командировкам' in loc.lower() else 0
            
            # 4. Опыт работы
            if 'Опыт (двойное нажатие для полной версии)' in df.columns:
                exp = str(row['Опыт (двойное нажатие для полной версии)']) if pd.notna(row['Опыт (двойное нажатие для полной версии)']) else ''
                years = re.search(r'(\d+)\s*лет', exp)
                months = re.search(r'(\d+)\s*месяц', exp)
                resume['exp_years'] = int(years.group(1)) if years else 0
                resume['exp_months'] = int(months.group(1)) if months else 0
                resume['total_exp'] = resume['exp_years'] * 12 + resume['exp_months']
            
            # 5. Образование
            if 'Образование и ВУЗ' in df.columns:
                edu = str(row['Образование и ВУЗ']) if pd.notna(row['Образование и ВУЗ']) else ''
                if 'Высшее' in edu:
                    resume['edu_level'] = 3
                elif 'Среднее специальное' in edu:
                    resume['edu_level'] = 2
                elif 'Среднее' in edu:
                    resume['edu_level'] = 1
                else:
                    resume['edu_level'] = 0
            
            # 6. Автомобиль
            if 'Авто' in df.columns:
                auto = str(row['Авто']) if pd.notna(row['Авто']) else ''
                resume['has_car'] = 1 if 'автомобиль' in auto.lower() else 0
            
            # 7. Занятость
            if 'Занятость' in df.columns:
                emp = str(row['Занятость']) if pd.notna(row['Занятость']) else ''
                resume['full_time'] = 1 if 'полная занятость' in emp.lower() else 0
            
            # 8. График
            if 'График' in df.columns:
                schedule = str(row['График']) if pd.notna(row['График']) else ''
                resume['full_day'] = 1 if 'полный день' in schedule.lower() else 0
            
            # 9. Дополнительные поля из CSV
            if 'Последенее/нынешнее место работы' in df.columns:
                company = str(row['Последенее/нынешнее место работы']) if pd.notna(row['Последенее/нынешнее место работы']) else ''
                resume['company_length'] = len(company)
            
            if 'Последеняя/нынешняя должность' in df.columns:
                position = str(row['Последеняя/нынешняя должность']) if pd.notna(row['Последеняя/нынешняя должность']) else ''
                resume['position_length'] = len(position)
                # Проверяем IT специальность
                it_keywords = ['админ', 'систем', 'it', 'программ', 'разработ', 'web', 'инженер']
                resume['is_it'] = 1 if any(keyword in position.lower() for keyword in it_keywords) else 0
            
            # 10. Дата обновления резюме
            if 'Обновление резюме' in df.columns:
                update_date = str(row['Обновление резюме']) if pd.notna(row['Обновление резюме']) else ''
                # Простая эвристика: недавнее обновление = 1
                resume['recent_update'] = 1 if '2019' in update_date else 0
            
            resumes.append(resume)
        
        return pd.DataFrame(resumes)
    
    def prepare_features(self, df):
        # Удаляем строки без ключевых данных
        df_clean = df.dropna(subset=['age', 'salary']).copy()
        
        # Логарифмируем зарплату
        df_clean['salary_log'] = np.log1p(df_clean['salary'])
        
        # Возрастные группы (0-4)
        df_clean['age_group'] = pd.cut(df_clean['age'], 
                                      [0, 25, 35, 45, 55, 100], 
                                      labels=[0, 1, 2, 3, 4])
        
        # Группы по опыту (0-5)
        df_clean['exp_group'] = pd.cut(df_clean['exp_years'], 
                                      [0, 1, 3, 5, 10, 20, 100], 
                                      labels=[0, 1, 2, 3, 4, 5])
        
        # Зарплата выше медианы (для целевой переменной)
        df_clean['high_salary'] = (df_clean['salary'] > df_clean['salary'].median()).astype(int)
        
        # Взаимодействие признаков
        df_clean['age_exp_interaction'] = df_clean['age'] * df_clean['exp_years']
        df_clean['edu_exp_interaction'] = df_clean['edu_level'] * df_clean['exp_years']
        
        return df_clean