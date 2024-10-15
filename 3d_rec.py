import math
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import re

def find_new_point(data, i):
    # Определение центра вращения
    x_center = np.mean(data[:, 0])
    y_center = np.mean(data[:, 1])
    z_center = np.min(data[:, 2])
    print(z_center)
    # for row in data:
    #     if row[2] == z_center:
    #         x_center = row[0]
    #         y_center = row[1]
    center = np.array([x_center, y_center, z_center])
    print(data)
    print("And ")

    # Угол альфа (в радианах)
    alpha = np.radians(20*i)  # Например, для угла 45 градусов

    # Матрица поворота для вращения вокруг оси Z
    def rotation_matrix_z(theta):
        # return np.array([[np.cos(theta), -np.sin(theta), 0],
        #                  [np.sin(theta), np.cos(theta), 0],
        #                  [0, 0, 1]])
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])

    # Матрица поворота
    rotation_matrix = rotation_matrix_z(alpha)

    # Перенос данных к центру вращения
    data_centered = data - center

    # Поворот данных
    data_rotated = np.dot(data_centered, rotation_matrix.T) + center
    print(data_rotated)

    return data_rotated


# Параметры камеры
focal_length = 2000.47
baseline = 110.548


first = 'imeg/Img'
end = '.jpg'

img_list1 = ['img30.jpg', 'img31.jpg', 'img32.jpg', 'img33.jpg']
img_list2 = ['img34.jpg', 'img35.jpg', 'img36.jpg', 'img37.jpg']
img_list = img_list2
glodal_cloud_point = np.array([])
points_global = np.array([])
corner = 0
dispart_late = np.uint8()

#проверку размера файла

# Загрузите стерео изображения
# imgL = cv2.imread('im21.png')
# imgR = cv2.imread('im22.png')

# for i in range(len(img_list)-1):
for i in range(1,10):
    # if re.fullmatch(r"\w*.jpg|.png", img_list[i]):
    #     print('True')
    # imgL = cv2.imread(img_list[i])
    # imgR = cv2.imread(img_list[i+

    print(first + str(i) + end)

    imgL = cv2.imread(first + str(i) + end)
    imgR = cv2.imread(first + str(i+1) + end)

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

    #-------------------------------------------------------------------------------------
    # # Создание детектора
    # orb = cv2.ORB_create()
    #
    # # Нахождение ключевых точек и дескрипторов
    # kp_left, des_left = orb.detectAndCompute(imgL, None)
    # kp_right, des_right = orb.detectAndCompute(imgR, None)
    # print(kp_left)

    # Сопоставление дескрипторов
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(des_left, des_right)

    # Сортировка совпадений
    # matches = sorted(matches, key=lambda x: x.distance)
    # print(matches)

    # points1 = np.array([kp_left[m.queryIdx].pt for m in matches])
    # points2 = np.array([kp_right[m.trainIdx].pt for m in matches])

    #-----------------------------------------------------------------------------
    # блок создания трехмерных точек
    # Преобразуйте карту глубины в 3D-координаты
    h, w = disparity.shape
    Q = np.float32([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, -focal_length],
                    [0, 0, 1/baseline, 0]])

    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    print(points_3D.ndim)

    # if i > 1:
    #     print(points_3D)
    #     print('AND')
    #     points_3D = shift_image(points_3D, dispart_late,  shift_amount=100)
    #     print(points_3D)
    #
    # dispart_late = disparity

    # Маска для действительных значений
    mask = disparity > disparity.min()

    # Извлеките действительные точки и цвета
    valid_points = points_3D[mask]

    #выводм колчество полученных точек
    print("Count point valid_points: ", len(valid_points))
    print('Min: '+ str(min(valid_points[2])))

    # выводм колчество полученных точек
    print("Count point valid_points: ", len(valid_points))

    valid_colors = cv2.cvtColor(grayL, cv2.COLOR_GRAY2RGB)[mask]
    if i > 1:
        valid_points = find_new_point(valid_points, i)
    # ----------------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Преобразуйте в формат, подходящий для Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(valid_colors / 255.0)

    # отсеиваем половину, поскольку комп может не выдержать большой нагрузки
    # rezult_pcd = pcd.uniform_down_sample(every_k_points=2)
    rezult_pcd = pcd

    print("Count point rezult_pcd: ", len(rezult_pcd.points))

    # ----------------------------------------------------------------------------
    # совемещение облака точек
    if i > 1:
        # Объединение облаков
        combined_pcd = glodal_cloud_point + rezult_pcd

        # Удаление дубликатов
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.01)
        glodal_cloud_point = combined_pcd
    else:
        glodal_cloud_point = rezult_pcd

# Сохраните облако точек в файл
o3d.io.write_point_cloud('point_cloud.ply', glodal_cloud_point)

print("Облако точек сохранено в 'point_cloud.ply'")
# -------------------------------------------------------------------------

# визуализация облака точек
o3d.visualization.draw_geometries([glodal_cloud_point])
# -------------------------------------------------------------------------


# # Визуализация облака точек с помощью matplotlib
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# xyz_point = np.asarray(glodal_cloud_point.points)
#
# # Извлечение координат
# x = xyz_point[:, 0]
# y = xyz_point[:, 1]
# z = xyz_point[:, 2]
#
# # Отображение облака точек
# ax.scatter(x, y, z)
#
# # Настройка меток осей
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# # Настройка пределов осей
# ax.set_xlim([15000, -2000])
# ax.set_ylim([1000, -15000])
# ax.set_zlim([1000, -15000])
#
# # Показать график
# plt.show()



#-------------------------------------------------------------------------------
# Фильтрация статистических выбросов
cl, ind = glodal_cloud_point.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
filtered_pcd = glodal_cloud_point.select_by_index(ind)
# filtered_pcd = glodal_cloud_point
print("Filter good")
#------------------------------------------------------------------------------

'''
# Извлечение массива точек
points = np.asarray(filtered_pcd.points)

# print(points)
'''


# Вычисление нормалей
filtered_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

if filtered_pcd.has_normals():
    print('Normals good')
else:
    print('Normals empty')
    filtered_pcd.estimate_normals(serch_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    print('Normals end')

# Примените метод понижения размерности (например, ball pivoting)
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     glodal_cloud_point,
#     o3d.utility.DoubleVector([0.4, 0.3, 0.4])  # Задайте радиусы для метода
# )
# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#     filtered_pcd, depth=11
# )
# filtered_pcd
# surface reconstruction
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(filtered_pcd, depth=10, n_threads=1)[0]

# rotate the mesh
rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
# mesh.rotate(rotation, center=(0, 0, 0))

if mesh.is_empty():
    print('Mesh empty!')
else:
    # Визуализируйте полученный мэш
    print('Mesh good!')
    o3d.visualization.draw_geometries([mesh])
