# 🌳 Машинное обучение. Ансамбли и случайные леса

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-green.svg)](https://scikit-learn.org/)

Учебный ноутбук с практическими заданиями по ансамблевым методам машинного обучения.

## 📚 Содержание

- **Ансамблирование**: комитет большинства (hard/soft voting), бэггинг, стэкинг, блендинг
- **Случайные леса**: исследование влияния количества деревьев на качество модели
- **Физическая ML-задача**: предсказание физических величин по потенциалам
- **Визуализация важности признаков**: сравнение LinearSVR и RandomForestRegressor
- **Трансформация данных**: стандартизация потенциалов через пулинг и извлечение статистик

## 🛠️ Используемые технологии

- Scikit-learn (RandomForest, ExtraTrees, LinearSVR, VotingClassifier, StackingClassifier)
- Pandas, NumPy
- Matplotlib, Seaborn
- CatBoost, XGBoost, LightGBM

## 🚀 Основные результаты

- Сравнение эффективности различных методов ансамблирования
- Реализация кастомного `BlendingClassifier`
- Разработка `PotentialTransformer` для обработки физических данных
- Визуализация важности признаков для моделей машинного обучения

## 📊 Задания

| Задание | Описание |
|---------|----------|
| 1 | Комитет большинства (VotingClassifier) |
| 2.1 | Random Forest vs Bagging — теоретический вопрос |
| 2.2 | Сравнение Bagging и RandomForest на разном количестве деревьев |
| 2.3 | График зависимости качества от числа деревьев |
| 3.1 | Техника предотвращения утечки данных в стекинге |
| 3.2 | Стекинг с разными базовыми моделями |
| 4 | Реализация BlendingClassifier |
| 5 | Анализ важности признаков (LinearSVR vs RandomForest) |
| 6 | ML-решение с трансформацией потенциалов |
| 7 | Найти мем про деревья решений |


## 🔧 Установка

```bash
pip install -r requirements.txt
