import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_theme(style="whitegrid")
FILE = "weather1.csv"

# 1) Загрузка данных (excell)
cols = ["Местное время в Перми", "T", "P", "U", "Ff", "N", "H", "VV"]
df = pd.read_csv(FILE, sep=";", encoding="utf-8",
                 index_col=False, usecols=cols)
df = df[cols]
df = df.rename(columns={"Местное время в Перми": "time"})
df["time"] = pd.to_datetime(df["time"], format="%d.%m.%Y %H:%M")
df = df.sort_values("time").reset_index(drop=True)

for c in ["T", "P", "U", "Ff", "VV"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["N"] = df["N"].str.strip()
df["cloud100"] = df["N"].str.startswith("100%").fillna(False).astype(bool)


# 2) Диаграмма рассеяния T и U
plt.figure(figsize=(8, 6))
plt.scatter(df["T"], df["U"], s=8, alpha=0.6)
plt.xlabel("Температура, градусы C")
plt.ylabel("Относительная влажность, %")
plt.title("Зависимость влажности от температуры")
plt.tight_layout()
plt.show()

# 3) То же, цвет по облачности (100% — синий, а иначе красный)
colors = np.where(df["cloud100"], "blue", "red")
plt.figure(figsize=(8, 6))
plt.scatter(df["T"], df["U"], c=colors, s=8, alpha=0.6)
plt.xlabel("Температура, градусы C")
plt.ylabel("Относительная влажность, %")
plt.title("Температура и влажность (синий — облачность 100%, красный — остальные)")
plt.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", label="Облачность 100%"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="red", label="Остальные")])
plt.tight_layout()
plt.show()

# 4) График температуры по времени
plt.figure(figsize=(12, 5))
plt.plot(df["time"], df["T"], linewidth=0.8)
plt.xlabel("Местное время")
plt.ylabel("Температура, градусы C ")
plt.title("Изменение температуры во времени")
plt.tight_layout()
plt.show()

# 5) Среднемесячная температура в столбчатой диаграмме
df["month"] = df["time"].dt.month
monthly = df.groupby("month")["T"].mean()
print(monthly)

plt.figure(figsize=(9, 5))
plt.bar(monthly.index, monthly.values, color="tab:orange")
plt.xticks(monthly.index)
plt.xlabel("Номер месяца")
plt.ylabel("Средняя температура, градусы C")
plt.title("Среднемесячная температура")
plt.tight_layout()
plt.show()

# 6) Ленточная диаграмма: наблюдения по облачности
cloud_counts = df["N"].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=cloud_counts.values, y=cloud_counts.index, orient="h")
plt.xlabel("Количество наблюдений")
plt.ylabel("Облачность")
plt.title("Количество наблюдений для каждого варианта облачности")
plt.tight_layout()
plt.show()

# 7) Гистограмма T (10 диапазонов)
plt.figure(figsize=(8, 5))
plt.hist(df["T"].dropna(), bins=10, edgecolor="black")
plt.xlabel("Температура, градусы C")
plt.ylabel("Частота")
plt.title("Гистограмма частот температуры (10 диапазонов)")
plt.tight_layout()
plt.show()

# 8) Boxplot давления по группам дальности видимости
def vis_group(v):
    if pd.isna(v):
        return np.nan
    if v < 5:
        return "менее 5 км"
    if v <= 15:
        return "5–15 км"
    return "более 15 км"

df["vis_group"] = df["VV"].apply(vis_group)
order = ["менее 5 км", "5–15 км", "более 15 км"]

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="vis_group", y="P", order=order)
plt.xlabel("Горизонтальная дальность видимости")
plt.ylabel("Атмосферное давление, мм рт. ст.")
plt.title("Давление в зависимости от дальности видимости")
plt.tight_layout()
plt.show()

# 9) Круговая диаграмма: высота основания облаков
h_counts = df["H"].dropna().value_counts()
plt.figure(figsize=(9, 9))
plt.pie(h_counts.values, labels=h_counts.index, autopct="%1.1f%%", startangle=90,
        pctdistance=0.85, textprops={"fontsize": 8})
plt.title("Высота основания облаков (H)")
plt.tight_layout()
plt.show()