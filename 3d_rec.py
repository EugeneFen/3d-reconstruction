import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d
import cv2
import time

feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

def get_map(image):
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    image = image.resize(new_size)

    # prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    # get the prediction from the model
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # remove borders
    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad:-pad, pad:-pad]

    # Инвертируем значения пикселей
    output_inverted = np.max(output) - output

    # Нормализуем значения, если нужно
    output_inverted = (output_inverted - np.min(output_inverted)) / (np.max(output_inverted) - np.min(output_inverted))

    image = image.crop((pad, pad, image.width - pad, image.height - pad))

    # # visualize the prediction
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(image)
    # ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # ax[1].imshow(output, cmap='plasma')
    # ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # plt.tight_layout()
    # plt.pause(5)
    print("Map good")
    return output_inverted, image


def contour(depth_map_inverted, original_image):
    threshold_value = 0.478  # Пороговое значение

    # Преобразуем значения в диапазон [0, 255] для обработки
    depth_map_scaled = (depth_map_inverted * 255).astype(np.uint8)

    # Применяем порог для бинаризации
    _, thresholded = cv2.threshold(depth_map_scaled, threshold_value * 255, 255, cv2.THRESH_BINARY)

    # Находим контуры
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем маску для заполнения контуров
    mask = np.zeros_like(depth_map_scaled)

    # Рисуем контуры на маске
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Получаем точки внутри контуров
    points_inside_contour = np.argwhere(mask > 0)

    # Отсекаем глубину по маске
    depth_inside_contours = np.zeros_like(depth_map_inverted)
    depth_inside_contours[mask > 0] = depth_map_inverted[mask > 0]

    # Изменяем размер оригинального изображения под новую карту глубины
    resized_image = original_image.resize((depth_inside_contours.shape[1], depth_inside_contours.shape[0]))

    # Создаем цветное изображение для отображения
    output_map = cv2.cvtColor(depth_map_scaled, cv2.COLOR_GRAY2BGR)

    # Рисуем контуры на цветном изображении
    cv2.drawContours(output_map, contours, -1, (0, 255, 0), 2)  # Зеленые контуры

    # Отображаем результаты
    cv2.imshow('Detected Objects', output_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Получаем координаты X, Y и Z
    z_values = depth_inside_contours[points_inside_contour[:, 0], points_inside_contour[:, 1]]
    points_3d = np.hstack((points_inside_contour, z_values[:, np.newaxis]))

    pcd4 = o3d.geometry.PointCloud()
    pcd4.points = o3d.utility.Vector3dVector(points_3d)  # Установка точек
    o3d.visualization.draw_geometries([pcd4])  # Визуализация

    # Возвращаем массив точек и глубину внутри контуров, а также измененное изображение
    return points_inside_contour, depth_inside_contours, resized_image

def get_dispance(img1, img2, points):
    imgL = cv2.imread(img1)
    imgR = cv2.imread(img2)

    # Создание детектора
    orb = cv2.ORB_create()

    # Нахождение ключевых точек и дескрипторов
    kp_left, des_left = orb.detectAndCompute(imgL, None)
    kp_right, des_right = orb.detectAndCompute(imgR, None)
    # print(kp_left)

    # Сопоставление дескрипторов
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_left, des_right)

    # Сортировка совпадений
    matches = sorted(matches, key=lambda x: x.distance)
    # print(matches)

    points1 = np.array([kp_left[m.queryIdx].pt for m in matches])
    points2 = np.array([kp_right[m.trainIdx].pt for m in matches])

    # Вычисление разностей
    differences = points1 - points2

    # Вычисление средней разности
    mean_difference = np.mean(differences)

    def calculate_angle(x1, y1, xc, yc, d):
        # Вычисляем угол с помощью арктангенса
        angle = np.arctan2(y1 - yc, x1 - xc)  # Угол между вектором и осью X
        return np.degrees(angle)  # Приводим к градусам

    print(points1[0][0])
    print(points1[0][1])
    # Пример координат и расстояния
    x1, y1 = points1[0][0], points1[0][1]  # Координаты первой точки
    xc, yc = points[0], points[1]  # Координаты центра окружности
    d = mean_difference  # Расстояние до второй точки

    angle = calculate_angle(x1, y1, xc, yc, d)
    print("Угол поворота:", angle)

    return angle

def get_point_cloud(depth_inside_contours, resized_image):
    # Получаем ширину и высоту измененного изображения
    width, height = resized_image.size

    # Преобразуем глубину в формат uint8
    depth_image = (depth_inside_contours * 255 / np.max(depth_inside_contours)).astype('uint8')

    # Преобразуем изображение в массив NumPy
    image = np.array(resized_image)

    # Создаем RGBD изображение
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d,
                                                                    convert_rgb_to_intensity=False)

    # camera settings
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 300, 300, width / 2, height / 2)

    # create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    print("PointCloud good")
    return pcd

def get_filt_point(pcd):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    filtered_pcd = pcd.select_by_index(ind)
    # filtered_pcd = glodal_cloud_point
    print("Filter good")
    return filtered_pcd

def create_mesh(pcd):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
    pcd = pcd.select_by_index(ind)

    # estimate normals
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()

    # surface reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]

    # rotate the mesh
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))
    # o3d.visualization.draw_geometries([mesh])

    # save the mesh
    # o3d.io.write_triangle_mesh(f'./mesh.obj', mesh)
    print("Mesh good")

def get_center_point(pcd):
    points = np.asarray(pcd.points)
    # Определение центра вращения
    x_center = np.mean(points[:, 0])
    y_center = np.mean(points[:, 1])
    z_center = np.mean(points[:, 2])
    farthest_point = np.array([x_center, y_center, z_center])

    print("Center: ", farthest_point)
    return farthest_point


def filter_points_by_distance(pcd, threshold):
    # Получаем координаты точек
    points = np.asarray(pcd.points)

    # Вычисляем расстояние от центра облака
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)

    # Фильтруем точки по порогу расстояния
    mask = distances < threshold
    filtered_points = points[mask]

    # Создаем новое облако точек
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    print("Filter distances good")

    return filtered_pcd


def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

def visualize_point_cloud_math(pcd):
    # Извлечение точек
    points = np.asarray(pcd.points)

    # Извлечение координат
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Создание графика
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o')

    # Установка меток
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Показ графика
    plt.show()

def rotate_point_cloud(pcd, angle, center):
    points = np.asarray(pcd.points)
    print(points)
    print("And ")
    print(points.shape)  # Ожидается (N, 3)
    print(center.shape)  # Ожидается (3,)


    # Угол альфа (в радианах)
    alpha = np.radians(angle)  # Например, для угла 45 градусов

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
    data_centered = points - center

    # Поворот данных
    data_rotated = np.dot(data_centered, rotation_matrix.T) + center
    print(data_rotated)

    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(data_rotated)
    pcd3.colors = pcd.colors

    print("Rotate good")
    return pcd3

def merge_point_clouds(pcd1, pcd2):
    # Объединяем два облака точек
    pcd1 += pcd2  # Или используйте pcd1.points.extend(pcd2.points) для ручного объединения
    print("Marge good")
    return pcd1

def get_point_claster(outlier_cloud):
    # Кластеризация точек
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

    # Визуализация кластеров
    max_label = labels.max()
    colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))

    # Установка цветов
    outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Визуализация
    o3d.visualization.draw_geometries([outlier_cloud])


def segment_planes(pcd):
    # Вычисление нормалей
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Сегментация плоскостей с помощью RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

    # Извлечение плоскости и остатка
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    return inlier_cloud, outlier_cloud, plane_model

def visualize_segmented_objects(inlier_cloud, outlier_cloud):
    inlier_cloud.paint_uniform_color([1, 0, 0])  # Красный для плоскости
    outlier_cloud.paint_uniform_color([0, 1, 0])  # Зеленый для остального облака

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# Начало отсчета времени
start_time = time.time()

name_1 = "imeg/Img1.jpg"
name_2 = "imeg/Img2.jpg"

image = Image.open(name_1)
imageR = Image.open(name_2)

#получаем крту глубины
output_inverted, image = get_map(image)
output_invertedR, imageR = get_map(imageR)

#Выделение обьекта на изображении
# points_inside_contourR, depth_inside_contoursR, resized_imageR = contour(output_invertedR, imageR)
# points_inside_contour, depth_inside_contours, resized_image = contour(output_inverted, image)
#получение облака точек из выделенной области
# pcd = get_point_cloud(depth_inside_contours, resized_image)
# pcdR = get_point_cloud(depth_inside_contoursR, resized_imageR)

#получаем облако точек
pcd = get_point_cloud(output_inverted, image)
pcdR = get_point_cloud(output_invertedR, imageR)

# # Сегментация
# inlier_cloud, outlier_cloud, plane_model = segment_planes(pcd)
# visualize_segmented_objects(inlier_cloud, outlier_cloud)  # Визуализация

# визуплизация с осями
# visualize_point_cloud_math(pcd)
# визуальзация в пространстве
visualize_point_cloud(pcd)
visualize_point_cloud(pcdR)

# фильтрация облака точек
print("Points before filtering: ", len(pcd.points))
pcd = get_filt_point(pcd)
pcdR = get_filt_point(pcdR)
print("Points after filtering: ", len(pcd.points))

#смещение облака точек и совмещение
center = get_center_point(pcd)
centerR = get_center_point(pcdR)
angle = get_dispance(name_1, name_2, center)
pcd = merge_point_clouds(pcd, rotate_point_cloud(pcdR, angle, centerR))

#создание обькта
create_mesh(pcd)

# Конец отсчета времени
end_time = time.time()

# Вычисление времени выполнения
execution_time = end_time - start_time
print(f"Время выполнения: {execution_time:.6f} секунд")

