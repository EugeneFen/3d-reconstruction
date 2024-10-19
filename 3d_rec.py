import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d
import cv2

feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

def get_map():
    image = Image.open("imeg/Img1.jpg")

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

    # visualize the prediction
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax[1].imshow(output, cmap='plasma')
    ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.pause(5)
    return output_inverted, image

def get_disparity():
    # Загружаем изображения
    imgL = cv2.imread('imeg/Img1.jpg')
    imgR = cv2.imread('imeg/Img2.jpg')

    # Преобразуйте изображения в градации серого
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Создайте объект StereoSGBM для вычисления карты глубины
    stereo = cv2.StereoBM_create(numDisparities=48, blockSize=15)
    disparity = stereo.compute(grayL, grayR)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # return disparity_normalized
    # Отображение или сохранение
    # cv2.imshow('Disparity Map', disparity_normalized)
    cv2.imwrite('disparity_map.jpg', disparity_normalized)

def get_point_cloud(output_inverted, image):
    width, height = image.size

    depth_image = (output_inverted * 255 / np.max(output_inverted)).astype('uint8')
    image = np.array(image)

    # create rgbd image
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d,
                                                                    convert_rgb_to_intensity=False)

    # camera settings
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 300, 300, width / 2, height / 2)

    # create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    return pcd

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
    o3d.visualization.draw_geometries([mesh])

    # save the mesh
    # o3d.io.write_triangle_mesh(f'./mesh.obj', mesh)

# # load and resize the input image
# disparity = cv2.imread("disparity_map.jpg", cv2.IMREAD_UNCHANGED)
#
#
# new_height = 480 if image.height > 480 else image.height
# new_height -= (new_height % 32)
# new_width = int(new_height * image.width / image.height)
# diff = new_width % 32
# new_width = new_width - diff if diff < 16 else new_width + 32 - diff
# new_size = (new_width, new_height)
# image = image.resize(new_size)
#
# # prepare image for the model
# inputs = feature_extractor(images=image, return_tensors="pt")

# Load the disparity map
# load and resize the input image

output_inverted, image = get_map()

pcd = get_point_cloud(output_inverted, image)

# plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
# inlier_cloud = pcd.select_by_index(inliers)
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# inlier_cloud.paint_uniform_color([1, 0, 0])
# outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
#
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
#--------------------------------------------------------------------------

# from sklearn.cluster import DBSCAN
#
# # Загрузка облака точек
# # pcd = o3d.io.read_point_cloud("path_to_your_point_cloud.ply")
# points = np.asarray(pcd.points)
#
# # Кластеризация
# dbscan = DBSCAN(eps=0.05, min_samples=10)
# labels = dbscan.fit_predict(points)
#
# # Визуализация результатов
# max_label = labels.max()
# colors = plt.get_cmap("jet")(labels / max_label)  # Используем цветовую карту
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Убираем альфа-канал
#
# o3d.visualization.draw_geometries([pcd])
#
# # Сегментация плоскости с использованием RANSAC
# plane_model, inliers = pcd.segment_plane(distance_threshold=2,
#                                          ransac_n=5,
#                                          num_iterations=1000)
#
# # Извлечение точек плоскости и остальных точек
# inlier_cloud = pcd.select_by_index(inliers)
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
#
# # Визуализация результатов
# inlier_cloud.paint_uniform_color([1, 0, 0])  # Красный для плоскости
# outlier_cloud.paint_uniform_color([0, 1, 0])  # Зеленый для остальных точек
#
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

create_mesh(pcd)

