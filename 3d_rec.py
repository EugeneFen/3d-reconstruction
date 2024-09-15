import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

# Параметры камеры
focal_length = 6338.47
baseline = 171.548

# Загрузите стерео изображения
imgL = cv2.imread('im21.png')
imgR = cv2.imread('im22.png')

# Преобразуйте изображения в градации серого
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# Создайте объект StereoSGBM для вычисления карты глубины
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=11)
disparity = stereo.compute(grayL, grayR)
print("Disparity good")
# print("Dispar min", np.min(disparity))
# print("Dispar max", np.max(disparity))

# Нормализуйте карту глубины
disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

#-----------------------------------------------------------------------------
# блок создания трехмерных точек
# Преобразуйте карту глубины в 3D-координаты
h, w = disparity.shape
Q = np.float32([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -focal_length],
                [0, 0, 1/baseline, 0]])

points_3D = cv2.reprojectImageTo3D(disparity, Q)

# Маска для действительных значений
mask = disparity > disparity.min()

# Извлеките действительные точки и цвета
valid_points = points_3D[mask]
valid_colors = cv2.cvtColor(grayL, cv2.COLOR_GRAY2RGB)[mask]
#----------------------------------------------------------------------------

#-------------------------------------------------------------------
# блок сохранения облака точек
# Преобразуйте в формат, подходящий для Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(valid_points)
pcd.colors = o3d.utility.Vector3dVector(valid_colors / 255.0)

# Сохраните облако точек в файл
o3d.io.write_point_cloud('point_cloud.ply', pcd)

print("Облако точек сохранено в 'point_cloud.ply'")
#-------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Визуализация 3D-точек
# valid_points = points_3D
# x = valid_points[:, :, 0].flatten()
# y = valid_points[:, :, 1].flatten()
# z = valid_points[:, :, 2].flatten()
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c='r', marker='o', s=1)
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Point Cloud')
#
# plt.show()

# визуализация по другому
o3d.visualization.draw_geometries([pcd])
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Фильтрация статистических выбросов
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
filtered_pcd = pcd.select_by_index(ind)
print("Filter good")
#------------------------------------------------------------------------------


# Извлечение массива точек
points = np.asarray(filtered_pcd.points)

# print(points)

# Создание трингуляции Делоне
tri = Delaunay(points)

print("Delaunay good")

# Визуализация
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r')

# Добавление тетраэдров
for simplex in tri.simplices:
    simplex = np.append(simplex, simplex[0])
    ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')

plt.show()
