import os
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import joblib
import shap
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_LGBM = os.path.join(BASE_DIR, "lightgbm_models.pkl")
PATH_CATBOOST = os.path.join(BASE_DIR, "catboost_model.cbm")
PATH_WEIGHTS = os.path.join(BASE_DIR, "ensemble_weights.json")

DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
PATH_MAPPING = os.path.join(DATA_DIR, "data_raw", "categorical_mapping.csv")
PATH_FEATURE_DESC = os.path.join(DATA_DIR, "data_raw", "HomeCredit_columns_description_uk.csv")

BTN_COLOR = "#1b8451"
BTN_TEXT = "white"
RISK_THRESHOLD = 0.17


DERIVED_FEATURES = {
    "AMT_ANNUITY",
    "CREDIT_TO_ANNUITY",
    "ANNUITY_TO_INCOME",
    "CREDIT_TO_INCOME",
    "GOODS_TO_CREDIT",
    "EXT_MEAN",
    "EXT_MIN",
    "EXT_MAX",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "DAYS_EMPLOYED_PERC",
}

# Ознаки, які доцільно дозволити для ручного введення.
# У формі будуть показані тільки ті з них, які реально є у feature_names з навченої моделі.
MANUAL_ALLOWED_FEATURES = {
    # Ідентифікація
    "SK_ID_CURR",

    # Фінансові показники заявки
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_GOODS_PRICE",
    "AMT_ANNUITY",

    # Кількісні соціально-демографічні ознаки
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "HOUR_APPR_PROCESS_START",

    # Бінарні ознаки контакту та документів
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",

    # Категоріальні ознаки анкети
    "CODE_GENDER",
    "NAME_CONTRACT_TYPE",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",

    # Ознаки житла/нерухомості
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG",
    "COMMONAREA_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "FLOORSMIN_AVG",
    "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG",
    "TOTALAREA_MODE",

    # Зовнішні скорингові оцінки, якщо вони є у моделі
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",

    # Запити до кредитного бюро
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
}

FEATURE_LABELS_UA = {
    "AGE_YEARS": "Вік клієнта, років",
    "EMPLOYED_YEARS": "Стаж роботи, років",
    "CREDIT_TERM_MONTHS": "Термін кредиту, місяців",
    "REGISTRATION_YEARS": "Скільки років клієнт зареєстрований за адресою",
    "PHONE_CHANGE_YEARS": "Скільки років тому змінювався телефон",
    "BUILD_YEAR": "Рік побудови житла",

    "SK_ID_CURR": "ID клієнта",
    "AMT_CREDIT": "Сума кредиту",
    "AMT_ANNUITY": "Ануїтетний платіж",
    "AMT_INCOME_TOTAL": "Річний дохід клієнта",
    "AMT_GOODS_PRICE": "Вартість товару / майна",

    "CNT_CHILDREN": "Кількість дітей",
    "CNT_FAM_MEMBERS": "Кількість членів сім’ї",
    "HOUR_APPR_PROCESS_START": "Година подання заявки",

    "FLAG_OWN_CAR": "Наявність автомобіля",
    "FLAG_OWN_REALTY": "Наявність нерухомості",
    "FLAG_MOBIL": "Вказано мобільний телефон",
    "FLAG_EMP_PHONE": "Вказано робочий телефон",
    "FLAG_WORK_PHONE": "Вказано робочий контактний телефон",
    "FLAG_CONT_MOBILE": "Мобільний телефон доступний",
    "FLAG_PHONE": "Вказано телефон",
    "FLAG_EMAIL": "Вказано email",

    "CODE_GENDER": "Стать",
    "NAME_CONTRACT_TYPE": "Тип кредиту",
    "NAME_TYPE_SUITE": "Хто супроводжував клієнта",
    "NAME_INCOME_TYPE": "Тип доходу",
    "NAME_EDUCATION_TYPE": "Освіта",
    "NAME_FAMILY_STATUS": "Сімейний стан",
    "NAME_HOUSING_TYPE": "Тип житла",
    "OCCUPATION_TYPE": "Професія / зайнятість",
    "ORGANIZATION_TYPE": "Сфера або тип організації",
    "WEEKDAY_APPR_PROCESS_START": "День подання заявки",

    "APARTMENTS_AVG": "Площа / характеристика квартири",
    "BASEMENTAREA_AVG": "Площа підвального приміщення",
    "YEARS_BEGINEXPLUATATION_AVG": "Період експлуатації житла",
    "YEARS_BUILD_AVG": "Характеристика року побудови житла",
    "COMMONAREA_AVG": "Загальна площа спільних приміщень",
    "ELEVATORS_AVG": "Наявність / кількість ліфтів",
    "ENTRANCES_AVG": "Кількість під’їздів",
    "FLOORSMAX_AVG": "Максимальна кількість поверхів",
    "FLOORSMIN_AVG": "Мінімальна кількість поверхів",
    "LANDAREA_AVG": "Площа земельної ділянки",
    "LIVINGAPARTMENTS_AVG": "Житлові квартири",
    "LIVINGAREA_AVG": "Житлова площа",
    "NONLIVINGAPARTMENTS_AVG": "Нежитлові квартири",
    "NONLIVINGAREA_AVG": "Нежитлова площа",
    "TOTALAREA_MODE": "Загальна площа житла",

    "EXT_SOURCE_1": "Зовнішній скоринг 1",
    "EXT_SOURCE_2": "Зовнішній скоринг 2",
    "EXT_SOURCE_3": "Зовнішній скоринг 3",

    "AMT_REQ_CREDIT_BUREAU_HOUR": "Запити до кредитного бюро за годину",
    "AMT_REQ_CREDIT_BUREAU_DAY": "Запити до кредитного бюро за день",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "Запити до кредитного бюро за тиждень",
    "AMT_REQ_CREDIT_BUREAU_MON": "Запити до кредитного бюро за місяць",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Запити до кредитного бюро за квартал",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Запити до кредитного бюро за рік",

    "EXT_MEAN": "Середній зовнішній скоринг",
    "EXT_MIN": "Мінімальний зовнішній скоринг",
    "EXT_MAX": "Максимальний зовнішній скоринг",
    "CREDIT_TO_ANNUITY": "Кредит / ануїтет",
    "ANNUITY_TO_INCOME": "Ануїтет / дохід",
    "CREDIT_TO_INCOME": "Кредит / дохід",
    "GOODS_TO_CREDIT": "Вартість товару / кредит",
    "DAYS_EMPLOYED_PERC": "Частка стажу від віку",
}

FEATURE_HINTS = {
    "SK_ID_CURR": "можна залишити пустим",
    "AMT_CREDIT": "сума кредиту, наприклад 500000",
    "AMT_INCOME_TOTAL": "річний дохід, наприклад 180000",
    "AMT_GOODS_PRICE": "вартість товару/майна, наприклад 450000",
    "AMT_ANNUITY": "щомісячний платіж; якщо пусто — розрахується автоматично",

    "CNT_CHILDREN": "ціле число, наприклад 0 або 2",
    "CNT_FAM_MEMBERS": "ціле число, наприклад 3",
    "HOUR_APPR_PROCESS_START": "година 0–23",

    "FLAG_OWN_CAR": "оберіть Y/так або N/ні",
    "FLAG_OWN_REALTY": "оберіть Y/так або N/ні",
    "FLAG_MOBIL": "1 — так, 0 — ні",
    "FLAG_EMP_PHONE": "1 — так, 0 — ні",
    "FLAG_WORK_PHONE": "1 — так, 0 — ні",
    "FLAG_CONT_MOBILE": "1 — так, 0 — ні",
    "FLAG_PHONE": "1 — так, 0 — ні",
    "FLAG_EMAIL": "1 — так, 0 — ні",

    "CODE_GENDER": "оберіть зі списку",
    "NAME_CONTRACT_TYPE": "оберіть тип кредиту зі списку",
    "NAME_TYPE_SUITE": "оберіть зі списку",
    "NAME_INCOME_TYPE": "оберіть зі списку",
    "NAME_EDUCATION_TYPE": "оберіть зі списку",
    "NAME_FAMILY_STATUS": "оберіть зі списку",
    "NAME_HOUSING_TYPE": "оберіть зі списку",
    "OCCUPATION_TYPE": "оберіть зі списку",
    "ORGANIZATION_TYPE": "оберіть узагальнену сферу роботи",
    "WEEKDAY_APPR_PROCESS_START": "оберіть день або код зі списку",

    "APARTMENTS_AVG": "нормоване значення 0–1",
    "BASEMENTAREA_AVG": "нормоване значення 0–1",
    "YEARS_BEGINEXPLUATATION_AVG": "нормоване значення 0–1",
    "YEARS_BUILD_AVG": "нормоване значення 0–1 або заповніть рік побудови вище",
    "COMMONAREA_AVG": "нормоване значення 0–1",
    "ELEVATORS_AVG": "нормоване значення 0–1",
    "ENTRANCES_AVG": "нормоване значення 0–1",
    "FLOORSMAX_AVG": "нормоване значення 0–1",
    "FLOORSMIN_AVG": "нормоване значення 0–1",
    "LANDAREA_AVG": "нормоване значення 0–1",
    "LIVINGAPARTMENTS_AVG": "нормоване значення 0–1",
    "LIVINGAREA_AVG": "нормоване значення 0–1",
    "NONLIVINGAPARTMENTS_AVG": "нормоване значення 0–1",
    "NONLIVINGAREA_AVG": "нормоване значення 0–1",
    "TOTALAREA_MODE": "нормоване значення 0–1",

    "EXT_SOURCE_1": "0–1, якщо відома зовнішня оцінка",
    "EXT_SOURCE_2": "0–1, якщо відома зовнішня оцінка",
    "EXT_SOURCE_3": "0–1, якщо відома зовнішня оцінка",

    "AMT_REQ_CREDIT_BUREAU_HOUR": "кількість запитів",
    "AMT_REQ_CREDIT_BUREAU_DAY": "кількість запитів",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "кількість запитів",
    "AMT_REQ_CREDIT_BUREAU_MON": "кількість запитів",
    "AMT_REQ_CREDIT_BUREAU_QRT": "кількість запитів",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "кількість запитів",
}

CATEGORY_VALUE_LABELS = {
    # Загальні yes/no
    "Y": "Так",
    "N": "Ні",
    "M": "Чоловіча",
    "F": "Жіноча",
    "XNA": "Невідомо / не вказано",

    # Тип договору
    "Cash loans": "Грошовий кредит",
    "Revolving loans": "Відновлювана кредитна лінія",

    # Тип доходу
    "Working": "Працевлаштований клієнт",
    "Commercial associate": "Комерційна діяльність / бізнес",
    "Pensioner": "Пенсіонер",
    "State servant": "Державний службовець",
    "Unemployed": "Безробітний",
    "Student": "Студент",
    "Businessman": "Підприємець / бізнесмен",
    "Maternity leave": "Декретна відпустка",

    # Освіта
    "Secondary / secondary special": "Середня / середня спеціальна освіта",
    "Higher education": "Вища освіта",
    "Incomplete higher": "Незакінчена вища освіта",
    "Lower secondary": "Неповна середня освіта",
    "Academic degree": "Науковий ступінь",

    # Сімейний стан
    "Married": "Одружений / заміжня",
    "Single / not married": "Неодружений / незаміжня",
    "Civil marriage": "Цивільний шлюб",
    "Separated": "Розлучений / проживає окремо",
    "Widow": "Вдівець / вдова",
    "Unknown": "Невідомо",

    # Житло
    "House / apartment": "Власний будинок або квартира",
    "With parents": "Проживає з батьками",
    "Municipal apartment": "Муніципальне житло",
    "Rented apartment": "Орендоване житло",
    "Office apartment": "Службове житло",
    "Co-op apartment": "Кооперативне житло",

    # Супровід клієнта
    "Unaccompanied": "Без супроводу",
    "Family": "Із сім’єю",
    "Spouse, partner": "Із чоловіком/дружиною або партнером",
    "Children": "Із дітьми",
    "Other_A": "Інший супровід A",
    "Other_B": "Інший супровід B",
    "Group of people": "Група людей",

    # Дні тижня
    "MONDAY": "Понеділок",
    "TUESDAY": "Вівторок",
    "WEDNESDAY": "Середа",
    "THURSDAY": "Четвер",
    "FRIDAY": "П’ятниця",
    "SATURDAY": "Субота",
    "SUNDAY": "Неділя",

    # Організації / сфери
    "Business Entity Type 1": "Бізнес / приватна компанія, тип 1",
    "Business Entity Type 2": "Бізнес / приватна компанія, тип 2",
    "Business Entity Type 3": "Бізнес / приватна компанія, тип 3",
    "Self-employed": "Самозайнятість",
    "Government": "Державна установа",
    "School": "Освіта / школа",
    "Kindergarten": "Дитячий садок",
    "Medicine": "Медицина",
    "Military": "Військова сфера",
    "Police": "Поліція",
    "Security": "Охорона / безпека",
    "Security Ministries": "Силові міністерства",
    "Bank": "Банк / фінансова установа",
    "Insurance": "Страхування",
    "Legal Services": "Юридичні послуги",
    "Telecom": "Телекомунікації",
    "Postal": "Поштова служба",
    "Transport: type 1": "Транспорт, тип 1",
    "Transport: type 2": "Транспорт, тип 2",
    "Transport: type 3": "Транспорт, тип 3",
    "Transport: type 4": "Транспорт, тип 4",
    "Trade: type 1": "Торгівля, тип 1",
    "Trade: type 2": "Торгівля, тип 2",
    "Trade: type 3": "Торгівля, тип 3",
    "Trade: type 4": "Торгівля, тип 4",
    "Trade: type 5": "Торгівля, тип 5",
    "Trade: type 6": "Торгівля, тип 6",
    "Trade: type 7": "Торгівля, тип 7",
    "Industry: type 1": "Промисловість, тип 1",
    "Industry: type 2": "Промисловість, тип 2",
    "Industry: type 3": "Промисловість, тип 3",
    "Industry: type 4": "Промисловість, тип 4",
    "Industry: type 5": "Промисловість, тип 5",
    "Industry: type 6": "Промисловість, тип 6",
    "Industry: type 7": "Промисловість, тип 7",
    "Industry: type 8": "Промисловість, тип 8",
    "Industry: type 9": "Промисловість, тип 9",
    "Industry: type 10": "Промисловість, тип 10",
    "Industry: type 11": "Промисловість, тип 11",
    "Industry: type 12": "Промисловість, тип 12",
    "Industry: type 13": "Промисловість, тип 13",
    "Construction": "Будівництво",
    "Housing": "Житлово-комунальна сфера",
    "Restaurant": "Ресторанний бізнес",
    "Hotel": "Готельний бізнес",
    "Electricity": "Енергетика",
    "Emergency": "Надзвичайні служби",
    "Cleaning": "Клінінг / прибирання",
    "Realtor": "Нерухомість / ріелторська діяльність",
    "Services": "Сфера послуг",
    "Advertising": "Реклама",
    "Culture": "Культура",
    "Religion": "Релігійна організація",
    "Agriculture": "Сільське господарство",
    "Mobile": "Мобільний зв’язок",
    "Other": "Інше",
}


df = None
selected_row = None

feature_names = None
ensemble_weights = None
lgbm_model = None
catboost_model = None

categorical_options = {}
categorical_display_to_code = {}
feature_descriptions = {}


def load_feature_descriptions():
    global feature_descriptions

    feature_descriptions = {}

    if not os.path.exists(PATH_FEATURE_DESC):
        print("Файл описів не знайдено:", PATH_FEATURE_DESC)
        return

    try:
        desc_df = pd.read_csv(PATH_FEATURE_DESC)

        if "Row" not in desc_df.columns or "Description" not in desc_df.columns:
            print("У файлі описів немає колонок Row / Description")
            return

        for _, row in desc_df.iterrows():
            feature = str(row["Row"]).strip()
            description = str(row["Description"]).strip()

            if feature and description and description.lower() != "nan":
                feature_descriptions[feature] = description

        print(f"Завантажено описів фіч: {len(feature_descriptions)}")

    except Exception as e:
        print("Помилка завантаження описів:", e)



def make_category_display(feature, raw_value, code_value):
    raw_text = str(raw_value).strip()
    ua_text = CATEGORY_VALUE_LABELS.get(raw_text, raw_text)

    # У ручному режимі користувачу не показуємо технічні коди моделі.
    # Код категорії зберігається у categorical_display_to_code і підставляється автоматично.
    return ua_text if ua_text != raw_text else raw_text


ORGANIZATION_FRIENDLY_GROUPS = [
    (
        "Бізнес / приватна компанія",
        ["Business Entity Type 3", "Business Entity Type 2", "Business Entity Type 1"],
    ),
    (
        "Промисловість / виробництво",
        [
            "Industry: type 9", "Industry: type 3", "Industry: type 11", "Industry: type 1",
            "Industry: type 7", "Industry: type 4", "Industry: type 5", "Industry: type 2",
            "Industry: type 12", "Industry: type 6", "Industry: type 10", "Industry: type 13",
            "Industry: type 8",
        ],
    ),
    (
        "Торгівля",
        ["Trade: type 7", "Trade: type 3", "Trade: type 2", "Trade: type 6", "Trade: type 1", "Trade: type 5", "Trade: type 4"],
    ),
    (
        "Державна установа / бюджетна сфера",
        ["Government", "School", "Kindergarten", "University", "Medicine", "Military", "Police", "Security Ministries", "Emergency"],
    ),
    (
        "Самозайнятість / ФОП",
        ["Self-employed"],
    ),
    (
        "Будівництво / нерухомість / ЖКГ",
        ["Construction", "Realtor", "Housing"],
    ),
    (
        "Транспорт / пошта / логістика",
        ["Transport: type 4", "Transport: type 3", "Transport: type 2", "Transport: type 1", "Postal"],
    ),
    (
        "Фінансова сфера / страхування",
        ["Bank", "Insurance"],
    ),
    (
        "ІТ / телекомунікації / зв’язок",
        ["Telecom", "Mobile"],
    ),
    (
        "Сфера послуг / ресторанно-готельний бізнес",
        ["Services", "Restaurant", "Hotel", "Cleaning", "Advertising", "Legal Services"],
    ),
    (
        "Сільське господарство / енергетика",
        ["Agriculture", "Electricity"],
    ),
    (
        "Культура / релігійна організація",
        ["Culture", "Religion"],
    ),
    (
        "Інше / не вказано",
        ["Other", "XNA"],
    ),
]


def build_friendly_organization_options(sub, value_col, code_col):
    """Повертає зрозумілі користувачу варіанти сфери роботи без технічних Type 1/2/3."""
    raw_to_code = {}
    for _, row in sub.iterrows():
        raw_value = row[value_col]
        code_value = row[code_col]
        if pd.isna(raw_value) or pd.isna(code_value):
            continue
        raw_to_code[str(raw_value).strip()] = code_value

    displays = []
    display_to_code = {}

    for friendly_label, raw_candidates in ORGANIZATION_FRIENDLY_GROUPS:
        selected_raw = next((raw for raw in raw_candidates if raw in raw_to_code), None)
        if selected_raw is None:
            continue

        display = friendly_label
        code_value = raw_to_code[selected_raw]
        displays.append(display)
        try:
            display_to_code[display] = float(code_value)
        except Exception:
            display_to_code[display] = code_value

    return displays, display_to_code


def load_categorical_mapping():
    global categorical_options, categorical_display_to_code

    categorical_options = {}
    categorical_display_to_code = {}

    if not os.path.exists(PATH_MAPPING):
        print("Файл mapping не знайдено:", PATH_MAPPING)
        return

    try:
        mapping = pd.read_csv(PATH_MAPPING)
        mapping.columns = [str(c).strip() for c in mapping.columns]

        lower_cols = {c.lower(): c for c in mapping.columns}

        feature_col = (
            lower_cols.get("column")
            or lower_cols.get("feature")
            or lower_cols.get("feature_name")
            or lower_cols.get("row")
        )

        value_col = (
            lower_cols.get("value")
            or lower_cols.get("category")
            or lower_cols.get("label")
            or lower_cols.get("original")
            or lower_cols.get("original_value")
        )

        code_col = (
            lower_cols.get("encoded")
            or lower_cols.get("code")
            or lower_cols.get("encoded_value")
            or lower_cols.get("new_value")
        )

        if feature_col is None or value_col is None or code_col is None:
            if len(mapping.columns) >= 3:
                feature_col = mapping.columns[0]
                value_col = mapping.columns[1]
                code_col = mapping.columns[2]
            else:
                print("Невідома структура categorical_mapping.csv")
                return

        for feature in mapping[feature_col].dropna().unique():
            feature = str(feature)
            sub = mapping[mapping[feature_col] == feature].copy()

            if feature == "ORGANIZATION_TYPE":
                displays, display_to_code = build_friendly_organization_options(sub, value_col, code_col)
            else:
                displays = []
                display_to_code = {}

                for _, row in sub.iterrows():
                    raw_value = row[value_col]
                    code_value = row[code_col]

                    if pd.isna(raw_value) or pd.isna(code_value):
                        continue

                    display = make_category_display(feature, raw_value, code_value)

                    # Якщо дві технічні категорії мають однаковий український підпис,
                    # залишаємо першу, щоб список не дублювався.
                    if display in display_to_code:
                        continue

                    displays.append(display)

                    try:
                        display_to_code[display] = float(code_value)
                    except Exception:
                        display_to_code[display] = code_value

            if displays:
                categorical_options[feature] = displays
                categorical_display_to_code[feature] = display_to_code

        print(f"Завантажено категоріальних mapping: {len(categorical_options)}")

    except Exception as e:
        print("Помилка завантаження categorical_mapping:", e)


def make_feature_label(feature):
    if feature in FEATURE_LABELS_UA:
        return FEATURE_LABELS_UA[feature]

    if feature in feature_descriptions:
        return feature_descriptions[feature]

    agg_map = {
        "mean": "середнє значення",
        "max": "максимальне значення",
        "min": "мінімальне значення",
        "sum": "сума",
        "std": "стандартне відхилення",
    }

    source_map = {
        "BUREAU": "кредитне бюро",
        "PREV": "попередні заявки",
        "POS": "POS/CASH баланс",
        "INST": "платежі",
        "CC": "кредитна картка",
    }

    parts = feature.split("__")

    if len(parts) >= 3:
        source = source_map.get(parts[0], parts[0])
        base_feature = parts[1]
        agg = agg_map.get(parts[-1], parts[-1])

        base_description = feature_descriptions.get(
            base_feature,
            base_feature.replace("_", " ").lower()
        )

        return f"{agg}: {base_description} ({source})"

    if feature.startswith("LOG_"):
        base_feature = feature.replace("LOG_", "")
        base_description = feature_descriptions.get(
            base_feature,
            base_feature.replace("_", " ").lower()
        )
        return f"логарифм: {base_description}"

    return feature.replace("__", " / ").replace("_", " ").lower()


def get_feature_group(feature):
    if feature in {
        "SK_ID_CURR",
        "CODE_GENDER",
        "NAME_CONTRACT_TYPE",
        "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "HOUR_APPR_PROCESS_START",
    }:
        return "1. Особисті та соціально-демографічні дані"

    if feature in {
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_GOODS_PRICE",
        "AMT_ANNUITY",
                }:
        return "2. Фінансові показники"

    if feature in {
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
                    }:
        return "3. Контакти, майно та документи"

    if feature in {
        "APARTMENTS_AVG",
        "BASEMENTAREA_AVG",
        "YEARS_BEGINEXPLUATATION_AVG",
        "YEARS_BUILD_AVG",
        "COMMONAREA_AVG",
        "ELEVATORS_AVG",
        "ENTRANCES_AVG",
        "FLOORSMAX_AVG",
        "FLOORSMIN_AVG",
        "LANDAREA_AVG",
        "LIVINGAPARTMENTS_AVG",
        "LIVINGAREA_AVG",
        "NONLIVINGAPARTMENTS_AVG",
        "NONLIVINGAREA_AVG",
        "TOTALAREA_MODE",
    }:
        return "4. Дані про житло"

    if feature in {
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    }:
        return "5. Зовнішній скоринг і кредитна історія"

    return "6. Інші ознаки"



def load_resources():
    global lgbm_model, catboost_model, ensemble_weights, feature_names

    try:
        if not os.path.exists(PATH_LGBM):
            raise FileNotFoundError(f"Не знайдено файл LightGBM:\n{PATH_LGBM}")

        if not os.path.exists(PATH_CATBOOST):
            raise FileNotFoundError(f"Не знайдено файл CatBoost:\n{PATH_CATBOOST}")

        if not os.path.exists(PATH_WEIGHTS):
            raise FileNotFoundError(f"Не знайдено файл ваг ансамблю:\n{PATH_WEIGHTS}")

        load_feature_descriptions()
        load_categorical_mapping()

        lgbm_model = joblib.load(PATH_LGBM)

        catboost_model = CatBoostClassifier()
        catboost_model.load_model(PATH_CATBOOST)

        with open(PATH_WEIGHTS, "r", encoding="utf-8") as f:
            ensemble_weights = json.load(f)

        lgbm_for_features = lgbm_model[0] if isinstance(lgbm_model, list) else lgbm_model

        if hasattr(lgbm_for_features, "feature_name_"):
            feature_names = list(lgbm_for_features.feature_name_)
        elif hasattr(lgbm_for_features, "feature_name"):
            feature_names = list(lgbm_for_features.feature_name())
        elif hasattr(lgbm_for_features, "booster_"):
            feature_names = list(lgbm_for_features.booster_.feature_name())
        else:
            raise ValueError("Не вдалося отримати назви ознак з LightGBM-моделі.")

        return True

    except Exception as e:
        return str(e)


def predict_lgbm(row_df):
    if isinstance(lgbm_model, list):
        return float(np.mean([m.predict_proba(row_df)[0][1] for m in lgbm_model]))

    return float(lgbm_model.predict_proba(row_df)[0][1])


def get_lgbm_shap(row_df):
    lgbm_for_shap = lgbm_model[0] if isinstance(lgbm_model, list) else lgbm_model

    explainer = shap.TreeExplainer(lgbm_for_shap)
    shap_values = explainer.shap_values(row_df)

    if isinstance(shap_values, list):
        return shap_values[1][0]

    return shap_values[0]


def add_derived_features(values):
    def safe_div(a, b):
        if a is None or b is None:
            return np.nan
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return a / b

    def fill_if_model_has(feature, value):
        if feature in feature_names:
            values[feature] = value

    amt_credit = values.get("AMT_CREDIT", np.nan)
    amt_income = values.get("AMT_INCOME_TOTAL", np.nan)
    amt_goods = values.get("AMT_GOODS_PRICE", np.nan)

    credit_term_months = values.pop("CREDIT_TERM_MONTHS", np.nan)

    # Якщо користувач не ввів ануїтет, він розраховується наближено:
    # сума кредиту / термін кредиту.
    if "AMT_ANNUITY" not in values or pd.isna(values.get("AMT_ANNUITY", np.nan)):
        if not pd.isna(credit_term_months) and credit_term_months != 0:
            values["AMT_ANNUITY"] = safe_div(amt_credit, credit_term_months)
        else:
            values["AMT_ANNUITY"] = np.nan

    amt_annuity = values.get("AMT_ANNUITY", np.nan)

    values["CREDIT_TO_ANNUITY"] = safe_div(amt_credit, amt_annuity)
    values["ANNUITY_TO_INCOME"] = safe_div(amt_annuity, amt_income)
    values["CREDIT_TO_INCOME"] = safe_div(amt_credit, amt_income)

    if "GOODS_TO_CREDIT" in feature_names:
        values["GOODS_TO_CREDIT"] = safe_div(amt_goods, amt_credit)

    age_years = values.pop("AGE_YEARS", np.nan)
    employed_years = values.pop("EMPLOYED_YEARS", np.nan)
    registration_years = values.pop("REGISTRATION_YEARS", np.nan)
    id_document_age_years = values.pop("ID_DOCUMENT_AGE_YEARS", np.nan)
    phone_change_years = values.pop("PHONE_CHANGE_YEARS", np.nan)
    build_year = values.pop("BUILD_YEAR", np.nan)

    if not pd.isna(age_years):
        values["DAYS_BIRTH"] = -abs(age_years * 365)
    else:
        values["DAYS_BIRTH"] = np.nan

    if not pd.isna(employed_years):
        values["DAYS_EMPLOYED"] = -abs(employed_years * 365)
    else:
        values["DAYS_EMPLOYED"] = np.nan

    if not pd.isna(registration_years):
        fill_if_model_has("DAYS_REGISTRATION", -abs(registration_years * 365))

    if not pd.isna(id_document_age_years):
        fill_if_model_has("DAYS_ID_PUBLISH", -abs(id_document_age_years * 365))

    if not pd.isna(phone_change_years):
        fill_if_model_has("DAYS_LAST_PHONE_CHANGE", -abs(phone_change_years * 365))

    # У Home Credit житлові характеристики подані у нормованому вигляді.
    # Якщо користувач вводить реальний рік побудови, переводимо його в наближений показник 0–1.
    if not pd.isna(build_year):
        try:
            building_age = max(0, 2026 - float(build_year))
            normalized_build_age = min(building_age / 120, 1)
            for col in ["YEARS_BUILD_AVG", "YEARS_BUILD_MODE", "YEARS_BUILD_MEDI"]:
                fill_if_model_has(col, normalized_build_age)
        except Exception:
            pass

    values["DAYS_EMPLOYED_PERC"] = safe_div(
        values.get("DAYS_EMPLOYED", np.nan),
        values.get("DAYS_BIRTH", np.nan)
    )

    ext_values = [
        values.get("EXT_SOURCE_1", np.nan),
        values.get("EXT_SOURCE_2", np.nan),
        values.get("EXT_SOURCE_3", np.nan),
    ]
    ext_values = [v for v in ext_values if not pd.isna(v)]

    values["EXT_MEAN"] = float(np.mean(ext_values)) if ext_values else np.nan
    values["EXT_MIN"] = float(np.min(ext_values)) if ext_values else np.nan
    values["EXT_MAX"] = float(np.max(ext_values)) if ext_values else np.nan

    return values


def apply_manual_risk_adjustment(probability, row_df):
    """
    Бізнес-корекція для ручного режиму.
    Потрібна, бо в ручному режимі частина історичних/агрегованих фіч недоступна
    і залишається NaN, тому модель може недооцінювати очевидно високий ризик.
    """
    p = float(probability)

    credit = row_df["AMT_CREDIT"].iloc[0] if "AMT_CREDIT" in row_df.columns else np.nan
    income = row_df["AMT_INCOME_TOTAL"].iloc[0] if "AMT_INCOME_TOTAL" in row_df.columns else np.nan
    annuity = row_df["AMT_ANNUITY"].iloc[0] if "AMT_ANNUITY" in row_df.columns else np.nan

    credit_to_income = credit / income if pd.notna(credit) and pd.notna(income) and income > 0 else np.nan
    annuity_to_income = annuity / income if pd.notna(annuity) and pd.notna(income) and income > 0 else np.nan

    penalty = 0.0

    # Кредит дуже великий відносно доходу
    if pd.notna(credit_to_income):
        if credit_to_income > 10:
            penalty += 0.30
        elif credit_to_income > 5:
            penalty += 0.20
        elif credit_to_income > 3:
            penalty += 0.10

    # Щомісячний платіж занадто великий відносно доходу
    if pd.notna(annuity_to_income):
        if annuity_to_income > 1:
            penalty += 0.25
        elif annuity_to_income > 0.6:
            penalty += 0.15
        elif annuity_to_income > 0.4:
            penalty += 0.08

    return min(p + penalty, 0.99)


class CreditApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Credit Risk Analysis - Ensemble XAI System")
        self.root.geometry("1250x820")
        self.root.configure(bg="#f4f6f9")

        self.manual_entries = {}
        self.extra_manual_entries = {}
        self.manual_mode_active = False

        self.manual_feature_names = [
            f for f in feature_names
            if f in MANUAL_ALLOWED_FEATURES and f not in DERIVED_FEATURES
        ]

        self.build_tabs()
        self.build_bottom_panel()
        self.on_tab_changed()

    def build_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.file_tab = tk.Frame(self.notebook, bg="#f4f6f9")
        self.manual_tab = tk.Frame(self.notebook, bg="#f4f6f9")

        self.notebook.add(self.file_tab, text="Режим 1: CSV-файл")
        self.notebook.add(self.manual_tab, text="Режим 2: Ручне введення")
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        self.build_file_tab()
        self.build_manual_tab()

    def build_bottom_panel(self):
        self.bottom_panel = tk.Frame(self.root, bg="white", bd=1, relief="ridge", pady=15)
        self.bottom_panel.pack(fill="x", side="bottom", padx=10, pady=10)

        self.bottom_panel.columnconfigure(0, weight=1)
        self.bottom_panel.columnconfigure(1, weight=1)
        self.bottom_panel.columnconfigure(2, weight=3)

        self.lbl_id = tk.Label(
            self.bottom_panel,
            text="Оберіть або введіть клієнта",
            font=("Arial", 10, "bold"),
            bg="white",
        )
        self.lbl_id.grid(row=0, column=0, sticky="n")

        self.calc_btn = tk.Button(
            self.bottom_panel,
            text="Розрахувати",
            state="disabled",
            command=self.start_calc_thread,
            bg=BTN_COLOR,
            fg=BTN_TEXT,
            activebackground=BTN_COLOR,
            activeforeground=BTN_TEXT,
            disabledforeground=BTN_TEXT,
            font=("Arial", 11, "bold"),
            height=2,
            width=22,
            relief="flat",
        )
        self.calc_btn.grid(row=1, column=0, pady=(10, 0))

        res_frame = tk.Frame(self.bottom_panel, bg="#f8f9fa", bd=1, relief="sunken", padx=20)
        res_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10)

        tk.Label(res_frame, text="Ризик:", bg="#f8f9fa", font=("Arial", 10)).pack(pady=(5, 0))

        self.prob_lbl = tk.Label(
            res_frame,
            text="--%",
            font=("Arial", 28, "bold"),
            bg="#f8f9fa",
        )
        self.prob_lbl.pack(expand=True)

        self.factors_area = tk.Frame(self.bottom_panel, bg="white")
        self.factors_area.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=10)

    def build_file_tab(self):
        top_frame = tk.Frame(self.file_tab, bg="#f4f6f9", pady=10)
        top_frame.pack(fill="x")

        tk.Button(
            top_frame,
            text="📁 Завантажити CSV",
            command=self.load_csv,
            bg="#0056b3",
            fg="white",
            activebackground="#0056b3",
            activeforeground="white",
            font=("Arial", 9, "bold"),
        ).pack(side="left", padx=15)

        tk.Label(top_frame, text="Пошук SK_ID_CURR:", bg="#f4f6f9").pack(side="left", padx=5)

        self.id_entry = tk.Entry(top_frame, width=15)
        self.id_entry.pack(side="left", padx=5)

        tk.Button(top_frame, text="Знайти", command=self.find_by_real_id).pack(side="left")

        self.table_frame = tk.LabelFrame(
            self.file_tab,
            text=" База клієнтів ",
            bg="white",
            padx=5,
            pady=5,
        )
        self.table_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.tree = ttk.Treeview(self.table_frame, show="headings", selectmode="browse")

        ysb = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.tree.yview)
        xsb = ttk.Scrollbar(self.table_frame, orient="horizontal", command=self.tree.xview)

        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)

        self.tree.pack(side="top", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        xsb.pack(side="bottom", fill="x")

        self.tree.bind("<<TreeviewSelect>>", self.on_select)

    def is_manual_tab_active(self):
        return self.notebook.select() == str(self.manual_tab)

    def on_tab_changed(self, event=None):
        """Керує станом нижньої кнопки залежно від активного режиму."""
        if self.is_manual_tab_active():
            self.lbl_id.config(text="Клієнт: ручне введення", fg="#333")
            self.calc_btn.config(
                state="normal",
                text="Розрахувати",
                bg=BTN_COLOR,
                fg=BTN_TEXT,
            )
        else:
            if selected_row is not None and not self.manual_mode_active:
                self.calc_btn.config(
                    state="normal",
                    text="Розрахувати",
                    bg=BTN_COLOR,
                    fg=BTN_TEXT,
                )
            else:
                self.lbl_id.config(text="Оберіть клієнта з CSV-файлу", fg="#333")
                self.calc_btn.config(
                    state="disabled",
                    text="Розрахувати",
                    bg=BTN_COLOR,
                    fg=BTN_TEXT,
                )

    def build_manual_tab(self):
        title = tk.Label(
            self.manual_tab,
            text="Заповніть зрозумілі дані клієнта. Технічні поля датасету приховані, невідомі значення можна залишати порожніми.",
            bg="#f4f6f9",
            font=("Arial", 10, "bold"),
        )
        title.pack(anchor="w", padx=15, pady=(10, 3))

        note = tk.Label(
            self.manual_tab,
            text=(
                "Форма згрупована за блоками: анкетні дані, фінансові показники, контакти, житло, "
                "зовнішній скоринг і кредитна історія. Ануїтет може розраховуватися автоматично."
            ),
            bg="#f4f6f9",
            fg="#555",
            font=("Arial", 9),
        )
        note.pack(anchor="w", padx=15, pady=(0, 8))

        controls = tk.Frame(self.manual_tab, bg="#f4f6f9")
        controls.pack(fill="x", padx=15, pady=5)

        tk.Button(
            controls,
            text="Очистити поля",
            command=self.clear_manual_fields,
            bg="#6c757d",
            fg="white",
            font=("Arial", 9, "bold"),
        ).pack(side="left", padx=5)

        outer = tk.Frame(self.manual_tab, bg="#f4f6f9")
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        canvas = tk.Canvas(outer, bg="#f4f6f9", highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)

        self.manual_fields_frame = tk.Frame(canvas, bg="#f4f6f9")
        manual_window = canvas.create_window((0, 0), window=self.manual_fields_frame, anchor="nw")

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            canvas.itemconfig(manual_window, width=event.width)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.manual_fields_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.bind_all("<MouseWheel>", on_mousewheel)

        self.build_extra_manual_fields()
        self.build_model_manual_fields(start_row=3)

    def build_extra_manual_fields(self):
        extra_frame = tk.LabelFrame(
            self.manual_fields_frame,
            text=" 0. Зручні поля для ручного введення ",
            bg="#f4f6f9",
            padx=10,
            pady=10,
        )
        extra_frame.grid(row=0, column=0, columnspan=6, sticky="we", padx=5, pady=10)

        extra_fields = {
            "AGE_YEARS": "наприклад 35",
            "EMPLOYED_YEARS": "наприклад 5",
            "CREDIT_TERM_MONTHS": "наприклад 24",
            "REGISTRATION_YEARS": "наприклад 10",
            "PHONE_CHANGE_YEARS": "наприклад 1",
            "BUILD_YEAR": "наприклад 2005",
        }

        for i, (field, hint) in enumerate(extra_fields.items()):
            row = i // 2
            col = (i % 2) * 3
            display_name = make_feature_label(field)

            tk.Label(
                extra_frame,
                text=display_name,
                bg="#f4f6f9",
                font=("Arial", 8, "bold"),
                width=34,
                anchor="w",
            ).grid(row=row, column=col, sticky="w", padx=5, pady=3)

            entry = tk.Entry(extra_frame, width=18)
            entry.grid(row=row, column=col + 1, padx=5, pady=3)

            tk.Label(
                extra_frame,
                text=hint,
                bg="#f4f6f9",
                fg="#777",
                font=("Arial", 7),
                anchor="w",
                width=28,
            ).grid(row=row, column=col + 2, sticky="w", padx=5, pady=3)

            self.extra_manual_entries[field] = entry

    def build_model_manual_fields(self, start_row=3):
        grouped = {}

        for feature in self.manual_feature_names:
            group = get_feature_group(feature)
            grouped.setdefault(group, []).append(feature)

        current_row = start_row

        for group_name in sorted(grouped.keys()):
            group_frame = tk.LabelFrame(
                self.manual_fields_frame,
                text=f" {group_name} ",
                bg="#f4f6f9",
                padx=10,
                pady=8,
            )
            group_frame.grid(
                row=current_row,
                column=0,
                columnspan=6,
                sticky="we",
                padx=5,
                pady=10,
            )

            features = grouped[group_name]

            for i, feature in enumerate(features):
                row = i // 2
                col = (i % 2) * 3

                display_name = make_feature_label(feature)
                hint = FEATURE_HINTS.get(feature, "числове значення або залиште пустим")

                label = tk.Label(
                    group_frame,
                    text=display_name,
                    bg="#f4f6f9",
                    font=("Arial", 8, "bold"),
                    anchor="w",
                    width=34,
                )
                label.grid(row=row, column=col, sticky="w", padx=5, pady=3)

                if feature in categorical_options:
                    widget = ttk.Combobox(
                        group_frame,
                        values=categorical_options[feature],
                        width=30,
                        state="readonly",
                    )
                    widget.grid(row=row, column=col + 1, padx=5, pady=3)
                    hint_text = "оберіть зі списку"
                else:
                    widget = tk.Entry(group_frame, width=20)
                    widget.grid(row=row, column=col + 1, padx=5, pady=3)
                    hint_text = hint

                hint_label = tk.Label(
                    group_frame,
                    text=hint_text,
                    bg="#f4f6f9",
                    fg="#777",
                    font=("Arial", 7),
                    anchor="w",
                    width=34,
                )
                hint_label.grid(row=row, column=col + 2, sticky="w", padx=3, pady=3)

                self.manual_entries[feature] = widget

            current_row += 1

    def load_csv(self):
        global df

        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

        if not path:
            return

        try:
            df = pd.read_csv(path)
            self.populate_table()
            messagebox.showinfo("Готово", f"Файл завантажено.\nРядків: {len(df)}")
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося завантажити CSV:\n{e}")

    def populate_table(self):
        self.tree.delete(*self.tree.get_children())

        cols = list(df.columns[:25])
        self.tree["columns"] = cols

        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, stretch=False)

        for i, row in df.head(100).iterrows():
            self.tree.insert("", "end", iid=str(i), values=list(row.iloc[:25]))

    def find_by_real_id(self):
        if df is None:
            messagebox.showwarning("Помилка", "Спочатку завантажте CSV-файл.")
            return

        if "SK_ID_CURR" not in df.columns:
            messagebox.showerror("Помилка", "У CSV немає колонки SK_ID_CURR.")
            return

        try:
            target_id = float(self.id_entry.get())
            matches = df[df["SK_ID_CURR"] == target_id].index

            if len(matches) == 0:
                messagebox.showwarning("Помилка", f"ID {target_id} не знайдено.")
                return

            idx = int(matches[0])
            self.activate_client_from_file(idx)

            if self.tree.exists(str(idx)):
                self.tree.selection_set(str(idx))
                self.tree.see(str(idx))

        except Exception:
            messagebox.showerror("Помилка", "Введіть числовий ID.")

    def on_select(self, event):
        selected = self.tree.selection()

        if selected:
            self.activate_client_from_file(int(selected[0]))

    def activate_client_from_file(self, idx):
        global selected_row

        self.manual_mode_active = False
        selected_row = df.iloc[idx]
        curr_id = selected_row.get("SK_ID_CURR", idx)

        self.lbl_id.config(text=f"Клієнт ID: {curr_id}", fg="#333")
        self.prob_lbl.config(text="--%", fg="black")

        self.calc_btn.config(
            state="normal",
            text="Розрахувати",
            bg=BTN_COLOR,
            fg=BTN_TEXT,
        )

        self.clear_factors()

    def clear_manual_fields(self):
        for widget in self.manual_entries.values():
            if isinstance(widget, ttk.Combobox):
                widget.set("")
            else:
                widget.delete(0, tk.END)

        for entry in self.extra_manual_entries.values():
            entry.delete(0, tk.END)

        global selected_row
        self.manual_mode_active = False
        selected_row = None

        self.lbl_id.config(text="Клієнт: ручне введення" if self.is_manual_tab_active() else "Оберіть або введіть клієнта", fg="#333")
        self.prob_lbl.config(text="--%", fg="black")

        self.calc_btn.config(
            state="normal" if self.is_manual_tab_active() else "disabled",
            text="Розрахувати",
            bg=BTN_COLOR,
            fg=BTN_TEXT,
        )

        self.clear_factors()

    def prepare_manual_client(self):
        global selected_row

        values = {}
        invalid_fields = []

        for feature, widget in self.manual_entries.items():
            raw_value = widget.get().strip()

            if raw_value == "":
                values[feature] = np.nan
                continue

            if feature in categorical_display_to_code:
                values[feature] = categorical_display_to_code[feature].get(raw_value, np.nan)
                continue

            try:
                values[feature] = float(raw_value.replace(",", "."))
            except ValueError:
                invalid_fields.append(make_feature_label(feature))

        for feature, entry in self.extra_manual_entries.items():
            raw_value = entry.get().strip()

            if raw_value == "":
                values[feature] = np.nan
                continue

            try:
                values[feature] = float(raw_value.replace(",", "."))
            except ValueError:
                invalid_fields.append(make_feature_label(feature))

        if invalid_fields:
            messagebox.showerror(
                "Помилка",
                "Некоректні числові значення у полях:\n"
                + ", ".join(invalid_fields[:20])
                + ("\n..." if len(invalid_fields) > 20 else ""),
            )
            return False

        values = add_derived_features(values)

        for feature in feature_names:
            if feature not in values:
                values[feature] = np.nan

        self.manual_mode_active = True
        selected_row = pd.Series(values)

        self.lbl_id.config(text="Клієнт: ручне введення", fg="#333")
        self.prob_lbl.config(text="--%", fg="black")

        self.calc_btn.config(
            state="normal",
            text="Розрахувати",
            bg=BTN_COLOR,
            fg=BTN_TEXT,
        )

        self.clear_factors()
        return True

    def start_calc_thread(self):
        if self.is_manual_tab_active():
            if not self.prepare_manual_client():
                return
        elif selected_row is None:
            messagebox.showwarning("Помилка", "Спочатку оберіть клієнта з CSV-файлу або перейдіть у режим ручного введення.")
            return

        self.calc_btn.config(
            state="disabled",
            text="Обробка...",
            bg=BTN_COLOR,
            fg=BTN_TEXT,
            activebackground=BTN_COLOR,
            activeforeground=BTN_TEXT,
            disabledforeground=BTN_TEXT,
        )

        self.prob_lbl.config(text="...", fg="blue")
        self.clear_factors()

        tk.Label(
            self.factors_area,
            text="Обчислення SHAP-факторів...",
            font=("Arial", 9, "italic"),
            bg="white",
        ).pack(anchor="w")

        threading.Thread(target=self.calculate_async, daemon=True).start()

    def calculate_async(self):
        try:
            missing_features = [col for col in feature_names if col not in selected_row.index]

            if missing_features:
                raise ValueError(
                    "Не вистачає ознак для моделі:\n"
                    + ", ".join(missing_features[:20])
                    + ("\n..." if len(missing_features) > 20 else "")
                )

            row_df = selected_row[feature_names].to_frame().T
            row_df = row_df.apply(pd.to_numeric, errors="coerce")

            p_lgbm = predict_lgbm(row_df)
            p_cat = float(catboost_model.predict_proba(row_df)[0][1])

            w_lgbm = ensemble_weights.get("lightgbm_weight", ensemble_weights.get("lightgbm", 0.5))
            w_cat = ensemble_weights.get("catboost_weight", ensemble_weights.get("catboost", 0.5))

            total_weight = w_lgbm + w_cat

            if total_weight == 0:
                w_lgbm, w_cat = 0.5, 0.5
                total_weight = 1.0

            final_probability = (p_lgbm * w_lgbm + p_cat * w_cat) / total_weight

            if self.manual_mode_active:
                final_probability = apply_manual_risk_adjustment(final_probability, row_df)

            shap_l = get_lgbm_shap(row_df)

            cat_explainer = shap.TreeExplainer(catboost_model)
            shap_c = cat_explainer.shap_values(row_df)[0]

            importance = pd.Series(
                (shap_l * w_lgbm + shap_c * w_cat) / total_weight,
                index=feature_names,
            )

            if "SK_ID_CURR" in importance.index:
                importance = importance.drop("SK_ID_CURR")

            self.root.after(0, lambda: self.update_results(final_probability, importance))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Помилка моделі", str(e)))
            self.root.after(0, self.reset_button)

    def reset_button(self):
        can_calculate = self.is_manual_tab_active() or selected_row is not None
        self.calc_btn.config(
            state="normal" if can_calculate else "disabled",
            text="Розрахувати",
            bg=BTN_COLOR,
            fg=BTN_TEXT,
            activebackground=BTN_COLOR,
            activeforeground=BTN_TEXT,
            disabledforeground=BTN_TEXT,
        )

    def update_results(self, final_probability, importance):
        color = "#d01515" if final_probability >= RISK_THRESHOLD else "#1b8451"

        self.prob_lbl.config(text=f"{final_probability:.1%}", fg=color)
        self.reset_button()
        self.clear_factors()

        cols_frame = tk.Frame(self.factors_area, bg="white")
        cols_frame.pack(fill="both", expand=True)

        positive_col = tk.Frame(cols_frame, bg="white")
        positive_col.pack(side="left", fill="both", expand=True, padx=5)

        tk.Label(
            positive_col,
            text="✅ Плюси (знижують ризик):",
            font=("Arial", 8, "bold"),
            fg="green",
            bg="white",
        ).pack(anchor="w")

        for feature, value in importance.nsmallest(10).items():
            if value < 0:
                label_name = make_feature_label(feature)
                tk.Label(
                    positive_col,
                    text=f"• {label_name[:40]}: {value:.3f}",
                    bg="white",
                    font=("Arial", 7),
                ).pack(anchor="w")

        negative_col = tk.Frame(cols_frame, bg="white")
        negative_col.pack(side="left", fill="both", expand=True, padx=5)

        tk.Label(
            negative_col,
            text="⚠️ Ризики (підвищують ризик):",
            font=("Arial", 8, "bold"),
            fg="#d01515",
            bg="white",
        ).pack(anchor="w")

        for feature, value in importance.nlargest(10).items():
            if value > 0:
                label_name = make_feature_label(feature)
                tk.Label(
                    negative_col,
                    text=f"• {label_name[:40]}: +{value:.3f}",
                    bg="white",
                    font=("Arial", 7),
                ).pack(anchor="w")

    def clear_factors(self):
        for widget in self.factors_area.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    print("Starting application...")
    print("BASE_DIR:", BASE_DIR)
    print("MAPPING exists:", os.path.exists(PATH_MAPPING))
    print("FEATURE DESC exists:", os.path.exists(PATH_FEATURE_DESC))
    print("LGBM exists:", os.path.exists(PATH_LGBM))
    print("CATBOOST exists:", os.path.exists(PATH_CATBOOST))
    print("WEIGHTS exists:", os.path.exists(PATH_WEIGHTS))

    root = tk.Tk()
    root.withdraw()

    status = load_resources()

    if status is not True:
        messagebox.showerror(
            "Помилка ресурсів",
            f"Не вдалося завантажити моделі:\n{status}",
        )
        root.destroy()
    else:
        root.deiconify()
        app = CreditApp(root)
        root.mainloop()