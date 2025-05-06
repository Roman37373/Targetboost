# custom_transformers.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from category_encoders import CatBoostEncoder


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Трансформер для создания новых фич из сырых данных визитов.
    Создает временные, сессионные и пользовательские характеристики.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose  # Флаг для отладочного вывода
        self.feature_names_out_ = None  # Для совместимости с sklearn

    def fit(self, df, y=None):
        """Не требует обучения, только для совместимости с sklearn API"""
        if self.verbose:
            print("\n[FeatureEngineer] FIT: Инициализация трансформера")
            print(f"Получено {len(df)} строк для анализа")
        return self

    def transform(self, df):
        """Основной метод преобразования данных"""
        if self.verbose:
            print("\n[FeatureEngineer] TRANSFORM: Начало обработки")
            print(f"Входные данные: {len(df)} строк, {len(df.columns)} колонок")
            print("Первые 5 строк входных данных:")
            print(df.head())

        # 1. Преобразование даты и времени
        if self.verbose:
            print("\n[1/3] Обработка временных меток...")

        df['visit_datetime'] = pd.to_datetime(df['visit_date'] + ' ' + df['visit_time'])
        df['hour'] = df['visit_datetime'].dt.hour
        df['day_of_week'] = df['visit_datetime'].dt.dayofweek

        if self.verbose:
            print("Созданы фичи:")
            print("- visit_datetime (объединенная дата-время)")
            print("- hour (час визита)")
            print("- day_of_week (день недели)")
            print("Пример преобразованных данных:")
            print(df[['visit_date', 'visit_time', 'visit_datetime', 'hour', 'day_of_week']].head())

        # 2. Статистика по сессиям
        if self.verbose:
            print("\n[2/3] Расчет сессионной статистики...")

        session_stats = (
            df.groupby('session_id')
            .agg(
                hit_count=('hit_number', 'count'),
                max_hit=('hit_number', 'max'),
                unique_actions=('event_action', 'nunique')
            )
            .reset_index()
        )

        if self.verbose:
            print("Рассчитаны сессионные метрики:")
            print("- hit_count (количество хитов в сессии)")
            print("- max_hit (максимальный номер хита)")
            print("- unique_actions (уникальные действия)")
            print("Пример сессионной статистики:")
            print(session_stats.head())

        # 3. Статистика по пользователям
        if self.verbose:
            print("\n[3/3] Расчет пользовательской статистики...")

        user_stats = (
            df.groupby('client_id')
            .agg(
                user_sessions=('session_id', 'nunique'),
                avg_hits_per_session=('hit_number', 'mean')
            )
            .reset_index()
        )

        if self.verbose:
            print("Рассчитаны пользовательские метрики:")
            print("- user_sessions (количество сессий пользователя)")
            print("- avg_hits_per_session (среднее число хитов на сессию)")
            print("Пример пользовательской статистики:")
            print(user_stats.head())

        # 4. Объединение всех данных
        if self.verbose:
            print("\nОбъединение всех данных...")

        result_df = (
            df
            .merge(session_stats, on='session_id', how='left')
            .merge(user_stats, on='client_id', how='left')
        )

        if self.verbose:
            print("\n[FeatureEngineer] Преобразование завершено")
            print(f"Итоговый DataFrame: {len(result_df)} строк, {len(result_df.columns)} колонок")
            print("Новые колонки в данных:")
            new_cols = set(result_df.columns) - set(df.columns)
            print(list(new_cols))
            print("\nПример итоговых данных:")
            print(result_df.head())

        return result_df

    def get_feature_names_out(self, input_features=None):
        """Возвращает имена выходных фич для совместимости с sklearn"""
        if self.verbose:
            print("\n[FeatureEngineer] Запрос имен выходных фич")
        return np.array(list(self.feature_names_out_)) if self.feature_names_out_ else None


class HitPagePathTransformer(BaseEstimator, TransformerMixin):
    """Трансформер для преобразования URL пути в числовую фичу - длину пути.
    Обрабатывает пути вида '/path/to/page?param=value', извлекая количество элементов пути.
    """

    def __init__(self, verbose=True):
        self.feature_names_out_ = ['hit_page_path_length']  # Название выходной фичи
        self.verbose = verbose  # Флаг отладочного вывода
        self.feature_names_in_ = None  # Сохраняет имена входных фич при обучении

    def fit(self, X, y=None):
        """Сохраняет имена фич для pandas DataFrame (если есть)"""
        if self.verbose:
            print("\n=== FIT ===")
            print(f"Входные данные типа: {type(X)}")

        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
            if self.verbose:
                print(f"Обнаружены колонки: {self.feature_names_in_}")
                print(f"Первые 5 строк входных данных:\n{X.head()}")

        if self.verbose:
            print(f"Выходные фичи будут: {self.feature_names_out_}")
        return self

    def transform(self, X):
        """Преобразует путь страницы в длину пути (количество элементов)
        Обрабатывает как DataFrame, так и numpy array.
        Удаляет параметры запроса (все что после ?) перед подсчетом элементов пути.
        """
        if self.verbose:
            print("\n=== TRANSFORM ===")
            print(f"Тип входных данных: {type(X)}")

        def path_corrector(x):
            """Внутренняя функция обработки одного URL:
            - Обрабатывает NaN значения
            - Удаляет параметры запроса (после ?)
            - Разбивает путь по '/' и считает количество элементов
            """
            if pd.isna(x):
                return 0
            # Разделяем по ? и берем первую часть (путь)
            path_part = str(x).split('?')[0]
            # Разбиваем путь по '/' и фильтруем пустые элементы
            path_elements = [p for p in path_part.split('/') if p]
            return len(path_elements)

        if isinstance(X, pd.DataFrame):
            if self.verbose:
                print(f"Обрабатываю DataFrame с колонками: {X.columns.tolist()}")
                print("Пример данных до преобразования:")
                print(X['hit_page_path'].head())

            result = X.copy()
            result['hit_page_path_length'] = X['hit_page_path'].apply(path_corrector)
            result = result.drop(columns=['hit_page_path'])

            if self.verbose:
                print("\nРезультат преобразования:")
                print(result.head())
                print(f"Выходные колонки: {result.columns.tolist()}")

            return result

        elif isinstance(X, np.ndarray):
            if self.verbose:
                print(f"Обрабатываю numpy array с формой: {X.shape}")
                print(f"Первые 5 значений:\n{X[:5]}")

            result = np.array([[path_corrector(x)] for x in X])

            if self.verbose:
                print("\nРезультат преобразования:")
                print(f"Первые 5 значений:\n{result[:5]}")

            return result

        else:
            raise TypeError(f"Неподдерживаемый тип данных: {type(X)}")

    def get_feature_names_out(self, input_features=None):
        """Возвращает имена выходных фич (совместимость с sklearn ColumnTransformer)"""
        check_is_fitted(self, 'feature_names_out_')
        if self.verbose:
            print("\n=== GET_FEATURE_NAMES_OUT ===")
            print(f"Возвращаемые фичи: {self.feature_names_out_}")
        return np.array(self.feature_names_out_)

class SafeCatBoostEncoder(BaseEstimator, TransformerMixin):
    """Безопасный CatBoostEncoder с обработкой пропущенных значений и проверкой входных данных.

    Особенности:
    - Автоматически заполняет пропуски указанным значением (по умолчанию 'unknown')
    - Проверяет наличие всех указанных колонок
    - Сохраняет имена фич для совместимости с sklearn Pipeline
    - Подробное логирование процесса при verbose=True
    """

    def __init__(self, cols, fill_value='unknown', verbose=True):
        """
        Инициализация кодировщика.

        Параметры:
        ----------
        cols : str или list
            Колонки для кодирования
        fill_value : str, optional
            Значение для заполнения пропусков (по умолчанию 'unknown')
        verbose : bool, optional
            Флаг подробного вывода (по умолчанию True)
        """

        self.cols = cols if isinstance(cols, list) else [cols]  # Всегда работаем с list
        self.fill_value = fill_value
        self.encoder = None  # Будет инициализирован при обучении
        self.feature_names_in_ = None  # Сохранит имена входных фич
        self.verbose = verbose

    def fit(self, X, y):
        """Обучение кодировщика на данных.

        Этапы:
        1. Проверка и преобразование входных данных
        2. Заполнение пропусков
        3. Обучение CatBoostEncoder
        """
        # Шаг 1: Преобразование в DataFrame и сохранение имен фич
        X = pd.DataFrame(X)
        self.feature_names_in_ = X.columns.tolist()

        if self.verbose:
            print("\n=== FIT ===")
            print(f"Входные колонки: {self.feature_names_in_}")
            print(f"Колонки для кодирования: {self.cols}")
            print(f"Первые 5 строк до обработки:\n{X[self.cols].head()}")

        # Шаг 2: Проверка наличия всех указанных колонок
        missing = set(self.cols) - set(X.columns)
        if missing:
            raise ValueError(f"Отсутствуют колонки: {missing}")

        # Шаг 3: Заполнение пропусков указанным значением
        X_filled = X[self.cols].fillna(self.fill_value)

        if self.verbose:
            print(f"\nПосле заполнения пропусков ('{self.fill_value}'):\n{X_filled.head()}")

        # Шаг 4: Обучение CatBoostEncoder
        X_filled = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.encoder = CatBoostEncoder(cols=self.cols)
        self.encoder.fit(X_filled, y)

        if self.verbose:
            print("\nКодировщик успешно обучен!")
            print(f"Пример закодированных значений (первые 5):")
            encoded_sample = self.encoder.transform(X_filled.head())
            print(encoded_sample.head())

        return self

    def transform(self, X):
        """Применение обученного кодировщика к новым данным.

        Этапы:
        1. Проверка структуры входных данных
        2. Заполнение пропусков
        3. Применение кодирования
        """
        # Шаг 1: Преобразование в DataFrame с проверкой колонок
        X = pd.DataFrame(X, columns=self.feature_names_in_)

        if self.verbose:
            print("\n=== TRANSFORM ===")
            print(f"Входные колонки: {self.feature_names_in_}")
            print(f"Первые 5 строк до обработки:\n{X[self.cols].head()}")

        # Шаг 2: Заполнение пропусков (для новых данных)
        X[self.cols] = X[self.cols].fillna(self.fill_value)

        if self.verbose:
            print(f"\nПосле заполнения пропусков ('{self.fill_value}'):\n{X[self.cols].head()}")

        # Шаг 3: Применение обученного кодировщика
        X[self.cols] = self.encoder.transform(X[self.cols])

        if self.verbose:
            print("\nПосле кодирования CatBoostEncoder:")
            print(X[self.cols].head())

        return X

    def get_feature_names_out(self, input_features=None):
        """Возвращает имена фичей после преобразования.

        Для CatBoostEncoder имена фичей не меняются - исходные категориальные колонки
        заменяются на числовые значения с теми же именами.
        """
        check_is_fitted(self, 'feature_names_in_')
        if input_features is not None:
            # Проверяем соответствие переданных имен обученным
            if len(input_features) != len(self.feature_names_in_):
                raise ValueError("Несоответствие числа фичей.")
            return np.array(input_features)
        return np.array(self.feature_names_in_)