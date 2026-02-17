import os

# Путь к исходному датасету (можно использовать обработанные данные или сырой hh.csv)
DATA_PATH = os.path.join('parsing_analysis', 'hh.csv')  # замените на нужный файл

# Ключевые слова для отбора IT-специалистов
IT_KEYWORDS = [
    'разработчик', 'программист', 'developer', 'software', 'engineer',
    'backend', 'frontend', 'fullstack', 'full-stack', 'java', 'python',
    'c++', 'c#', 'javascript', 'js', 'php', 'ruby', 'go', 'rust',
    'data scientist', 'аналитик данных', 'data analyst', 'data engineer',
    'системный администратор', 'системный инженер', 'сисадмин',
    'devops', 'qa', 'тестировщик', 'автоматизация', 'web'
]

# Технологии для бинарных признаков (будут проверяться в блоке "Ключевые навыки")
TECH_LIST = [
    'python', 'java', 'c++', 'c#', 'javascript', 'js', 'php', 'sql',
    '1с', '1c', 'linux', 'windows', 'администрирование', 'сеть', 'git',
    'docker', 'kubernetes'
]

# Города-миллионники
MILLION_CITIES = [
    'Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург',
    'Казань', 'Нижний Новгород', 'Челябинск', 'Самара', 'Омск',
    'Ростов-на-Дону', 'Уфа', 'Красноярск', 'Пермь', 'Воронеж', 'Волгоград'
]

# Пути для сохранения артефактов модели (в общей папке resources)
MODEL_SAVE_PATH = os.path.join('resources', 'level_model.pkl')
SCALER_SAVE_PATH = os.path.join('resources', 'level_scaler.pkl')
ENCODER_SAVE_PATH = os.path.join('resources', 'level_encoder.pkl')