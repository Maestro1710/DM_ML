from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score

# Đọc dữ liệu
file_path = "goodreads_data (1).csv"
df = pd.read_csv(file_path)

# Loại bỏ các cột không cần thiết
df = df.drop(columns=["Unnamed: 0", "Description", "URL", "Author"])

# Chuyển đổi Num_Ratings thành số nguyên, xử lý lỗi dấu phẩy
df["Num_Ratings"] = df["Num_Ratings"].str.replace(",", "").astype(float)

# Xử lý giá trị trống (Đã sửa lỗi `FutureWarning`)
df["Avg_Rating"] = df["Avg_Rating"].fillna(df["Avg_Rating"].median())
df["Num_Ratings"] = df["Num_Ratings"].fillna(df["Num_Ratings"].median())
df["Genres"] = df["Genres"].fillna("Unknown")

# Lọc dữ liệu bất thường
df = df[(df["Avg_Rating"] >= 0) & (df["Avg_Rating"] <= 5)]  # Rating trong khoảng hợp lệ
df = df[df["Num_Ratings"] >= 10]  # Loại bỏ sách có quá ít đánh giá

# Xử lý cột Genres (chuyển thành nhiều cột)
df["Genres"] = df["Genres"].apply(lambda x: ", ".join(eval(x)) if isinstance(x, str) and x.startswith("[") else str(x))
genres_df = df["Genres"].str.get_dummies(sep=", ")
# Kết hợp lại dữ liệu
df = pd.concat([df, genres_df], axis=1)
df.drop(columns=["Genres"], inplace=True)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[["Avg_Rating", "Num_Ratings"] + list(genres_df.columns)])
print(data_scaled)
# df.to_excel("processed_goodreads.xlsx", index=False, engine="openpyxl")


# Sử dụng phương pháp Elbow để chọn số cụm tối ưu
# inertia_values = []
# K_range = range(1, 11)

# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(data_scaled)
#     inertia_values.append(kmeans.inertia_)

# # Vẽ biểu đồ Elbow
# plt.figure(figsize=(8, 5))
# plt.plot(K_range, inertia_values, marker="o", linestyle="--")
# plt.xlabel("Số cụm (k)")
# plt.ylabel("Tổng khoảng cách nội cụm (Inertia)")
# plt.title("Elbow Method để chọn số cụm tối ưu")
# plt.show()

# from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):  # Không xét k=1 vì silhouette không xác định được
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, labels)
    silhouette_scores.append(score)

# Vẽ đồ thị Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker="o", linestyle="--")
plt.xlabel("Số cụm (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score để chọn số cụm tối ưu")
plt.show()

# Tìm k có silhouette score cao nhất
best_k = range(2, 11)[np.argmax(silhouette_scores)]
print(f"Số cụm tối ưu theo Silhouette Score: {best_k}")
