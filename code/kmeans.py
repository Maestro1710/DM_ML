import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ast import literal_eval
import tkinter as tk
from tkinter import ttk, messagebox

# Đọc dữ liệu
data_path = 'goodreads_data (1).csv'
df = pd.read_csv(data_path)

# Loại bỏ các cột không cần thiết
df.drop(columns=['Num_Ratings', 'Description', 'URL'], inplace=True)

# Loại bỏ dữ liệu trống
df.dropna(inplace=True)
# df.to_csv("processed_books_csv.csv", index=False)

# Chuẩn hóa cột Genres (loại bỏ dấu [])
df['Genres'] = df['Genres'].apply(lambda x: ', '.join(literal_eval(x)) if isinstance(x, str) else x)
df.to_csv("processed_books_csv.csv", index=False)

# One-hot encoding cho Genres
genres_dummies = df['Genres'].str.get_dummies(sep=', ')

df = pd.concat([df, genres_dummies], axis=1)
df.drop(columns=['Genres'], inplace=True)

# Chuẩn hóa Avg-Rating
scaler = StandardScaler()
df['Avg_Rating'] = scaler.fit_transform(df[['Avg_Rating']])

# Chuyển đổi Author thành dạng số (mã hóa catcategorical encoding)
df['Author'] = df['Author'].astype('category').cat.codes


# Áp dụng thuật toán KMeans với số cụm tối ưu \
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[['Author', 'Avg_Rating'] + list(genres_dummies.columns)])

# Đánh giá Silhouette Score
silhouette_avg = silhouette_score(df[['Author', 'Avg_Rating'] + list(genres_dummies.columns)], df['Cluster'])
print(f'Silhouette Score: {silhouette_avg:.4f}')

# Kiểm tra số lượng sách trong mỗi cụm
cluster_counts = df['Cluster'].value_counts()
print("Số lượng sách trong mỗi cụm:")
print(cluster_counts)

# Gợi ý sách theo cụm
def recommend_books():
    book_title = entry.get()
    if book_title not in df['Book'].values:
        messagebox.showerror('lỗi',"Sách không có trong danh sách.")
        return 
    
    book_cluster = df[df['Book'] == book_title]['Cluster'].values[0]
    recommendations = df[df['Cluster'] == book_cluster].sample(5)
    output.delete('1.0', tk.END)
    output.insert(tk.END, "Gợi ý sách:\n")
    for idx, row in recommendations.iterrows():
        output.insert(tk.END, f"- {row['Book']}\n")
    # return recommendations[['Book', 'Author', 'Avg_Rating']]

# Xuất dữ liệu đã xử lý ra file 
# df.to_excel("processed_books_after.xlsx", index=False)
# Chạy thử nghiệm hệ thống
# sample_book = df['Book'].sample(1).values[0]
# print(f"Gợi ý sách cho: {sample_book}")
# print(recommend_books(sample_book, df))

# print("Hoàn thành xử lý dữ liệu, đánh giá và gợi ý sách!")

#UI
root = tk.Tk()
root.title('Gợi ý sách')
root.geometry("500x400")
tk.Label(root, text="tên sách").pack()
entry = tk.Entry(root, width=50)
entry.pack()
btn_Submit = tk.Button(root, text="gợi ý",width=20, pady=5, command=recommend_books)
btn_Submit.pack()
output = tk.Text(height=10, width=60)
output.pack()
root.mainloop()