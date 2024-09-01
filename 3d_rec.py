import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from scipy.spatial import Delaunay

# read left and right images
imgR = cv2.imread("img14.jpg", 0)
imgL = cv2.imread("img15.jpg", 0)
F = 6

# уменьшение качества изображения, бля уменьшения точек изображения
scaler_factor = 0.9
new_size = (int(imgL.shape[1]*scaler_factor), int(imgL.shape[0]*scaler_factor))

imgL = cv2.resize(imgL, new_size)
imgR = cv2.resize(imgR, new_size)

# создание карты глубины
stereo = cv2.StereoBM_create(numDisparities=16,
                            blockSize=15)

disparity = stereo.compute(imgL, imgR)

# нормальзация карты глубины
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)

# размеры карты глубины
height, width = disparity_normalized.shape

# print(height) #ширина
# print(width) #высота

# Вывод карты глубины как массива
# for i in disparity_normalized:
#     print(i)

#Вывод карты глубины
# plt.imshow(disparity_normalized, cmap='plasma')
# plt.colorbar()
# plt.title('DM')
# plt.show()

# Определение списков точек
point3d = np.empty((0, 3), float)
arrCount3d = np.array([])
arrPoint3d = np.empty((0, 3), float)

# фокусное расстояние
F = 3

#Создание трехреных точек
for i in range(len(disparity_normalized)): #x
    for j in range(len(disparity_normalized[1])): #y
        Z = disparity_normalized[i][j]
        X = ((i-width/2) * Z)/F
        Y = ((j-height/2) * Z)/F
        arrCount3d = np.append(arrCount3d, [round(X, 2), round(Y, 2), round(Z, 2)])
        arrPoint3d = np.vstack([arrPoint3d, arrCount3d])
        arrCount3d = np.empty((0, 3), float)
    point3d = np.vstack([point3d, arrPoint3d])
    arrPoint3d = np.empty((0, 3), float)

# Занесение координат в точки
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point3d)

# визуализация точек
# o3d.visualization.draw_geometries([pcd])

# Фильтрация статистических выбросов
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
filtered_pcd = pcd.select_by_index(ind)

# Визуализация до и после фильтрации
# o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")
# o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")

# Создание триангуляции
tri = Delaunay(filtered_pcd[:, :2])

# Визуализация
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filtered_pcd[:, 0], filtered_pcd[:, 1], filtered_pcd[:, 2], color='r')

for simplex in tri.simplices:
    simplex = np.append(simplex, simplex[0])
    ax.plot(filtered_pcd[simplex, 0], filtered_pcd[simplex, 1], filtered_pcd[simplex, 2], 'k-')

plt.show()



# визуализация отфильтрованных точек другим способом
# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111, projection='3d')
# x = [point[0] for point in point3d]
# y = [point[1] for point in point3d]
# z = [point[2] for point in point3d]
#
# # Создаем экземпляр Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# print(X)
#
# # Визуализируем точки
# ax.scatter(x, y, z)
#
# # Устанавливаем заголовок и подписываем оси
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# # Показываем график
# plt.show()