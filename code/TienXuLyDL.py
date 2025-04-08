# Đọc dữ liệu
from ast import literal_eval
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler


data_path = 'original_data_csv.csv'
df = pd.read_csv(data_path)

# Loại bỏ các cột không cần thiết
df.drop(columns=['Num_Ratings', 'Description', 'URL'], inplace=True)

# Loại bỏ dữ liệu trống
df.dropna(inplace=True)

# Chuẩn hóa cột Genres (loại bỏ dấu [])
df['Genres'] = df['Genres'].apply(lambda x: ', '.join(literal_eval(x)) if isinstance(x, str) else x)
# loại bỏ dấu "#"
df = df.apply(lambda col: col.str.replace("#", "", regex=False) if col.dtype == "object" else col)
df.to_csv("book_cleaned_python.csv", index=False)
df.to_excel("book_cleaned_python.xlsx", index=False)
