import pandas as pd
import numpy as np
import zipfile
from io import BytesIO
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# ------------------- 1. ЗАГРУЗКА ДАННЫХ -------------------
ZIP_PATH = r"C:\Users\umyar\Downloads\data_train.zip"   

dfs = []
with zipfile.ZipFile(ZIP_PATH) as z:
    parquet_files = [x for x in z.namelist() if x.endswith(".parquet") and "invalidResp" not in x]
    for file in parquet_files:
        data = z.read(file)
        dfs.append(pd.read_parquet(BytesIO(data)))
df = pd.concat(dfs, ignore_index=True)
df["row_ots"] = df["Weight"]
print(f"ROWS: {len(df)}, COLUMNS: {len(df.columns)}")

# ------------------- 2. ОПРЕДЕЛЕНИЕ КАТЕГОРИИ -------------------
if "CategoryDelivery" in df.columns:
    cat_col = "CategoryDelivery"
elif "CategoryNameDelivery" in df.columns:
    cat_col = "CategoryNameDelivery"
else:
    raise ValueError("Нет столбца CategoryDelivery или CategoryNameDelivery")

# ------------------- 3. ФИЛЬТРАЦИЯ -------------------
work = df[(df["BrandinDelivery"] == 1) & (df[cat_col].notna())].copy()
print(f"FILTERED ROWS: {len(work)}")

# ------------------- 4. АГРЕГАЦИЯ daily_ots -------------------
agg = (
    work.groupby(["SubjectID", "researchdate", "BrandID", "Brand", cat_col], as_index=False)
    .agg(
        Weight=("Weight", "first"),
        query_cnt=("BrandID", "size")
    )
)
agg["Weight"] = pd.to_numeric(agg["Weight"], errors="coerce")
agg["daily_ots"] = agg["Weight"] * agg["query_cnt"]
print(f"AGG SHAPE: {agg.shape}")

# ------------------- 5. ПРИЗНАКИ ДЛЯ АЛГОРИТМА -------------------
# 5.1 Размер бренда (количество наблюдений в агрегате)
brand_size = agg.groupby("BrandID")["daily_ots"].count().rename("brand_size")
agg = agg.merge(brand_size, on="BrandID", how="left")

# 5.2 Доля респондента в OTS бренда за день (share)
brand_day_total = agg.groupby(["BrandID", "researchdate"])["daily_ots"].transform("sum")
agg["share"] = agg["daily_ots"] / brand_day_total.replace(0, np.nan)

# 5.3 Robust z-score внутри бренда (только для брендов с размером >= 20)
def robust_z(x):
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-9:
        return np.zeros(len(x))
    return 0.6745 * (x - med) / mad

agg["brand_rz"] = 0.0
large_brands = agg["brand_size"] >= 20
if large_brands.any():
    agg.loc[large_brands, "brand_rz"] = (
        agg[large_brands].groupby("BrandID")["daily_ots"]
        .transform(lambda s: robust_z(s.values))
    )

# Для маленьких брендов используем категориальную статистику
small_brands = agg["brand_size"] < 20
if small_brands.any():
    cat_rz = (
        agg[small_brands].groupby([cat_col])["daily_ots"]
        .transform(lambda s: robust_z(s.values))
    )
    agg.loc[small_brands, "brand_rz"] = cat_rz

# 5.4 Активность пользователя за день (суммарный OTS)
user_day = agg.groupby(["SubjectID", "researchdate"])["daily_ots"].sum().reset_index(name="user_day_ots")
user_median = user_day.groupby("SubjectID")["user_day_ots"].median().rename("user_median")
user_day = user_day.merge(user_median, on="SubjectID", how="left")
user_day["user_ratio"] = user_day["user_day_ots"] / user_day["user_median"].replace(0, np.nan)
agg = agg.merge(user_day[["SubjectID", "researchdate", "user_ratio"]], on=["SubjectID", "researchdate"], how="left")

# 5.5 effective_rz: для брендов с большим размером берём brand_rz, иначе (medium) смесь (можно оставить как есть)
# Здесь мы уже для малых брендов использовали категориальную rz, поэтому effective_rz = brand_rz
agg["effective_rz"] = agg["brand_rz"]

# 5.6 daily_ots_global_p95 (для фильтра)
global_p95 = agg["daily_ots"].quantile(0.95)

# ------------------- 6. КОМБИНИРОВАННЫЙ SCORE -------------------
agg["score"] = (
    agg["effective_rz"].clip(lower=0) *
    np.log1p(agg["user_ratio"].fillna(1)) *
    np.sqrt(agg["share"].fillna(0)) *
    np.log1p(agg["daily_ots"])
)

# ------------------- 7. ПОРОГ (АДАПТИВНЫЙ) -------------------
# Порог по правилу Q3 + 3*IQR
q1 = agg.groupby(cat_col)["score"].quantile(0.25)
q3 = agg.groupby(cat_col)["score"].quantile(0.75)
iqr = q3 - q1
threshold_iqr = q3 + 3 * iqr
agg["threshold"] = agg[cat_col].map(threshold_iqr)

# Глобальный порог для категорий, где IQR = 0 или мало данных
global_q1 = agg["score"].quantile(0.25)
global_q3 = agg["score"].quantile(0.75)
global_iqr = global_q3 - global_q1
global_threshold_iqr = global_q3 + 3 * global_iqr
agg["threshold"].fillna(global_threshold_iqr, inplace=True)

# ------------------- 8. ФИЛЬТРЫ АНОМАЛИЙ -------------------
# Категориальный p95 для daily_ots
cat_p95 = agg.groupby(cat_col)["daily_ots"].quantile(0.95).to_dict()
agg["cat_p95"] = agg[cat_col].map(cat_p95).fillna(global_p95)

anomaly_mask = (
    (agg["effective_rz"] >= 8) &
    (agg["user_ratio"] >= 3) &
    (agg["daily_ots"] >= agg["cat_p95"]) &
    (agg["score"] >= agg["threshold"])
)

triggers = agg[anomaly_mask].copy()
print(f"Найдено триггеров (бренд-день-пользователь): {len(triggers)}")

# ------------------- 9. ЕДИНИЦЫ УДАЛЕНИЯ -------------------
anomaly_days = triggers[["SubjectID", "researchdate"]].drop_duplicates()
print(f"Уникальных пар (SubjectID, researchdate) к удалению: {len(anomaly_days)}")

# ------------------- 10. ВЫВОД ДИАГНОСТИЧЕСКИХ МЕТРИК -------------------
total_respondent_days = agg[["SubjectID", "researchdate"]].drop_duplicates().shape[0]
removed_share = len(anomaly_days) / total_respondent_days if total_respondent_days else 0
total_ots_original = agg["daily_ots"].sum()
anomaly_pairs = set(zip(anomaly_days["SubjectID"], anomaly_days["researchdate"]))
removed_mask = agg.apply(lambda row: (row["SubjectID"], row["researchdate"]) in anomaly_pairs, axis=1)
removed_ots = agg.loc[removed_mask, "daily_ots"].sum()
ots_retained = (total_ots_original - removed_ots) / total_ots_original
mean_anomalies_per_day = anomaly_days.groupby("researchdate").size().mean()

print(f"Всего пар респондент-день: {total_respondent_days}")
print(f"Удалено пар респондент-день: {len(anomaly_days)} ({removed_share:.2%})")
print(f"Сохранённая доля OTS: {ots_retained:.2%}")
print(f"Среднее число аномальных респондентов в день: {mean_anomalies_per_day:.2f}")

# ------------------- 11. СОХРАНЕНИЕ ФАЙЛОВ -------------------

output_dir = "output"
print("Создаю папку output...")
os.makedirs(output_dir, exist_ok=True)
print("Папка создана, путь:", os.path.abspath(output_dir))
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# anomalies.csv
print("Текущая рабочая папка:", os.getcwd())
print("Создаю anomalies.csv...")
anomaly_days.to_csv(os.path.join(output_dir, "anomalies.csv"), index=False)

# anomaly_reasons.csv
reasons = triggers[["SubjectID", "researchdate", "BrandID", "Brand", cat_col,
                    "daily_ots", "score", "threshold", "effective_rz", "user_ratio", "share"]].copy()
reasons["reason"] = reasons.apply(
    lambda r: f"effective_rz={r['effective_rz']:.2f}; user_ratio={r['user_ratio']:.2f}; share={r['share']:.3f}",
    axis=1
)
reasons = reasons.rename(columns={cat_col: "CategoryDelivery"})
reasons[["SubjectID", "researchdate", "BrandID", "Brand", "CategoryDelivery",
         "daily_ots", "score", "threshold", "reason"]].to_csv(
    os.path.join(output_dir, "anomaly_reasons.csv"), index=False
)

# ------------------- 12. ФУНКЦИИ ДЛЯ РАСЧЁТА OTS ДО/ПОСЛЕ -------------------
def compute_ots_by_day(df_full, anomaly_set):
    df_full = df_full.copy()
    df_full["researchdate"] = pd.to_datetime(df_full["researchdate"]).dt.date
    if anomaly_set:
        mask = ~df_full.apply(lambda row: (row["SubjectID"], row["researchdate"]) in anomaly_set, axis=1)
        df_filtered = df_full[mask]
    else:
        df_filtered = df_full
    ots = df_filtered.groupby("researchdate")["Weight"].sum()
    return ots

# Преобразуем даты
df["researchdate"] = pd.to_datetime(df["researchdate"]).dt.date
anomaly_set = set(zip(anomaly_days["SubjectID"], anomaly_days["researchdate"]))

ots_before = compute_ots_by_day(df, None)
ots_after = compute_ots_by_day(df, anomaly_set)

# ------------------- 13. ОБЯЗАТЕЛЬНЫЕ ГРАФИКИ -------------------
# 13.1 total_ots_before_after.png
plt.figure(figsize=(12,5))
all_dates = sorted(set(ots_before.index).union(set(ots_after.index)))
ots_before = ots_before.reindex(all_dates, fill_value=0)
ots_after = ots_after.reindex(all_dates, fill_value=0)
plt.plot(all_dates, ots_before, marker='o', label='До очистки')
plt.plot(all_dates, ots_after, marker='s', label='После очистки')
plt.title('Общий OTS по дням')
plt.xlabel('Дата')
plt.ylabel('OTS')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plots", "total_ots_before_after.png"))
plt.close()

# 13.2 category_ots_change.png – исправленная версия
cat_before = df.groupby([cat_col, "researchdate"])["row_ots"].sum().unstack(fill_value=0).sum(axis=1).astype(float)
mask = ~df.apply(lambda r: (r["SubjectID"], r["researchdate"]) in anomaly_set, axis=1)
cat_after = df[mask].groupby([cat_col, "researchdate"])["Weight"].sum().unstack(fill_value=0).sum(axis=1).astype(float)

all_cats = sorted(set(cat_before.index).union(set(cat_after.index)))
cat_before = cat_before.reindex(all_cats, fill_value=0.0)
cat_after = cat_after.reindex(all_cats, fill_value=0.0)

# Вычисляем процент изменения, заменяя inf и -inf на 0
pct_change = (cat_after - cat_before) / cat_before * 100
pct_change = pct_change.replace([np.inf, -np.inf], np.nan).fillna(0)
pct_change = pct_change.sort_values()

if not pct_change.empty:
    plt.figure(figsize=(10,6))
    pct_change.plot(kind='barh', color='steelblue')
    plt.title(f'Изменение OTS по {cat_col} (после удаления, %)')
    plt.xlabel('Изменение (%)')
    plt.ylabel(cat_col)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "category_ots_change.png"))
    plt.close()
else:
    print("Нет данных для графика category_ots_change.png")

# 13.3 daily_anomaly_count.png
daily_counts = anomaly_days.groupby("researchdate").size()
plt.figure(figsize=(12,5))
daily_counts.plot(kind='bar')
plt.title('Количество аномальных респондентов по дням')
plt.xlabel('Дата')
plt.ylabel('Число аномальных пар (SubjectID, researchdate)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plots", "daily_anomaly_count.png"))
plt.close()

print("Обязательные графики сохранены в output/plots/")

# ------------------- 14. АНАЛИТИЧЕСКИЕ ФУНКЦИИ (пункт 8.2) -------------------
def plot_demographics_before_after(demographic_col):
    """ demographic_col: 'Пол', 'Возраст', 'Регион', 'Федеральный округ', 'Количество детей', 'Занятость', 'Доход' """
    before = df.groupby(demographic_col)["Weight"].sum()
    mask = ~df.apply(lambda r: (r["SubjectID"], r["researchdate"]) in anomaly_set, axis=1)
    after = df[mask].groupby(demographic_col)["Weight"].sum()
    all_vals = sorted(set(before.index).union(set(after.index)))
    before = before.reindex(all_vals, fill_value=0)
    after = after.reindex(all_vals, fill_value=0)
    x = np.arange(len(all_vals))
    width = 0.35
    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, before, width, label='До очистки')
    plt.bar(x + width/2, after, width, label='После очистки')
    plt.xticks(x, all_vals, rotation=45, ha='right')
    plt.title(f'OTS по признаку {demographic_col}')
    plt.ylabel('Суммарный OTS')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plots/demo_{demographic_col}.png"))
    plt.close()

def plot_resource_before_after(resource_col):
    """ resource_col: 'ResourceName', 'ResourceType', 'Platform', 'UseType' """
    before = df.groupby(resource_col)["Weight"].sum()
    mask = ~df.apply(lambda r: (r["SubjectID"], r["researchdate"]) in anomaly_set, axis=1)
    after = df[mask].groupby(resource_col)["Weight"].sum()
    all_vals = sorted(set(before.index).union(set(after.index)))
    before = before.reindex(all_vals, fill_value=0)
    after = after.reindex(all_vals, fill_value=0)
    x = np.arange(len(all_vals))
    width = 0.35
    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, before, width, label='До очистки')
    plt.bar(x + width/2, after, width, label='После очистки')
    plt.xticks(x, all_vals, rotation=45, ha='right')
    plt.title(f'OTS по признаку {resource_col}')
    plt.ylabel('Суммарный OTS')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plots/resource_{resource_col}.png"))
    plt.close()

def plot_category_before_after(category_col):
    """ category_col: 'Category1', 'Category2', 'Category3' """
    if category_col not in df.columns:
        print(f"Столбец {category_col} отсутствует")
        return
    before = df.groupby(category_col)["Weight"].sum()
    mask = ~df.apply(lambda r: (r["SubjectID"], r["researchdate"]) in anomaly_set, axis=1)
    after = df[mask].groupby(category_col)["Weight"].sum()
    all_vals = sorted(set(before.index).union(set(after.index)))
    before = before.reindex(all_vals, fill_value=0)
    after = after.reindex(all_vals, fill_value=0)
    x = np.arange(len(all_vals))
    width = 0.35
    plt.figure(figsize=(14,6))
    plt.bar(x - width/2, before, width, label='До очистки')
    plt.bar(x + width/2, after, width, label='После очистки')
    plt.xticks(x, all_vals, rotation=90, ha='right')
    plt.title(f'OTS по {category_col}')
    plt.ylabel('Суммарный OTS')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plots/category_{category_col}.png"))
    plt.close()

def get_queries_for_anomaly(subject_id, date):
    """ Возвращает DataFrame с запросами для аномального респондента в указанный день """
    date_obj = pd.to_datetime(date).date()
    mask = (df["SubjectID"] == subject_id) & (df["researchdate"] == date_obj)
    result = df[mask][["QueryText", "Brand", cat_col, "Weight"]].copy()
    print(f"Запросы респондента {subject_id} на {date_obj}:")
    print(result.to_string(index=False))
    return result

def plot_brand_before_after(brand_id):
    """ График OTS по дням для конкретного бренда """
    brand_mask = (df["BrandID"] == brand_id)
    if brand_mask.sum() == 0:
        print(f"Бренд {brand_id} не найден")
        return
    before = df[brand_mask].groupby("researchdate")["Weight"].sum()
    mask = ~df.apply(lambda r: (r["SubjectID"], r["researchdate"]) in anomaly_set, axis=1)
    after = df[brand_mask & mask].groupby("researchdate")["Weight"].sum()
    all_dates = sorted(set(before.index).union(set(after.index)))
    before = before.reindex(all_dates, fill_value=0)
    after = after.reindex(all_dates, fill_value=0)
    plt.figure(figsize=(12,5))
    plt.plot(all_dates, before, marker='o', label='До очистки')
    plt.plot(all_dates, after, marker='s', label='После очистки')
    plt.title(f'OTS бренда {brand_id} по дням')
    plt.xlabel('Дата')
    plt.ylabel('OTS')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plots/brand_{brand_id}_ots.png"))
    plt.close()

# Пример вызова (раскомментируйте по желанию):
# plot_demographics_before_after("Пол")
# plot_resource_before_after("ResourceType")
# plot_category_before_after("Category1")
# get_queries_for_anomaly(anomaly_days["SubjectID"].iloc[0], anomaly_days["researchdate"].iloc[0])
# plot_brand_before_after(triggers["BrandID"].iloc[0])

print("Готово. Все файлы сохранены в папке output/")