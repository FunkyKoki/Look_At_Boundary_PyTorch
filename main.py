import os
import cv2
import dlib
import time
import copy
import numpy as np
from dataset import GeneralDataset
from models import *
from utils import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.interpolate import spline

use_dataset = 'WFLW'
use_epoch = '750'

# load network
devices = torch.device('cuda:0')
print('*****  WFLW trained Model Evaluating  *****')
print('Loading network ...')
estimator = Estimator()
regressor = Regressor(output=2*kp_num[use_dataset])
estimator = load_weights(estimator, 'estimator_'+use_epoch+'.pth', devices)
regressor = load_weights(regressor, use_dataset+'_regressor_'+use_epoch+'.pth', devices)
estimator = estimator.cuda(device=devices)
regressor = regressor.cuda(device=devices)
estimator.eval()
regressor.eval()
print('Loading network done!\nStart testing ...')

# detect face and facial landmark
rescale_ratio = 0.1/2
cv2.namedWindow("Face Detector")
cap = cv2.VideoCapture(0)
face_keypoint_coords = []

while cap.isOpened():      # isOpened()  检测摄像头是否处于打开状态
    ret, img = cap.read()  # 把摄像头获取的图像信息保存至img变量
    if ret is True:        # 如果摄像头读取图像成功
        cv2.imshow('Image', img)
        k = cv2.waitKey(100)
        if k == ord('c') or k == ord('C'):
            t_start = str(int(time.time()))

            face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
            rec = face_detector(img, 1)

            if len(rec) == 0:
                print('No Face Detected!')
            else:
                print('Detect ' + str(len(rec)) + ' face(s).')

            with torch.no_grad():
                for face_i in range(len(rec)):
                    t = str(int(time.time()))

                    rec_list = rec.pop().rect
                    height = rec_list.bottom() - rec_list.top()
                    width = rec_list.right() - rec_list.left()
                    bbox = [
                        int(rec_list.left() - rescale_ratio * width),
                        int(rec_list.top() - rescale_ratio * height),
                        int(rec_list.right() + rescale_ratio * width),
                        int(rec_list.bottom() + rescale_ratio * height)
                    ]
                    position_before = np.float32([
                        [int(bbox[0]), int(bbox[1])],
                        [int(bbox[0]), int(bbox[3])],
                        [int(bbox[2]), int(bbox[3])]
                    ])
                    position_after = np.float32([
                        [0, 0],
                        [0, 255],
                        [255, 255]
                    ])
                    crop_matrix = cv2.getAffineTransform(position_before, position_after)
                    face_img = cv2.warpAffine(img, crop_matrix, (256, 256))
                    face_gray = convert_img_to_gray(face_img)
                    face_norm = pic_normalize(face_gray)

                    input_face = torch.Tensor(face_norm)
                    input_face = input_face.unsqueeze(0)
                    input_face = input_face.unsqueeze(0).cuda()

                    pred_heatmaps = estimator(input_face)
                    pred_coords = regressor(input_face, pred_heatmaps[-1].detach()).detach().cpu().squeeze().numpy()

                    for kp_index in range(kp_num[use_dataset]):
                        cv2.circle(
                            face_img,
                            (int(pred_coords[2 * kp_index]), int(pred_coords[2 * kp_index + 1])),
                            2,
                            (0, 0, 255),
                            -1
                        )
                    show_img(face_img, 'face_small_keypoint'+str(face_i), 500, 650, keep=True)
                    cv2.imwrite('./pics/face_' + t + '_0.png', face_img)

                    heatmaps = F.interpolate(
                        pred_heatmaps[-1],
                        scale_factor=4,
                        mode='bilinear',
                        align_corners=True
                    )
                    heatmaps = heatmaps.squeeze(0).detach().cpu().numpy()
                    heatmaps_sum = heatmaps[0]
                    for heatmaps_index in range(boundary_num - 1):
                        heatmaps_sum += heatmaps[heatmaps_index + 1]
                    plt.axis('off')
                    plt.imshow(heatmaps_sum, interpolation='nearest', vmax=1., vmin=0.)
                    fig = plt.gcf()
                    fig.set_size_inches(2.56 / 3, 2.56 / 3)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    fig.savefig('hm.png', format='png', transparent=True, dpi=300, pad_inches=0)
                    hm = cv2.imread('hm.png')
                    syn = cv2.addWeighted(face_img, 0.4, hm, 0.6, 0)
                    show_img(syn, 'face_small_boundary'+str(face_i), 900, 650)
                    cv2.imwrite('./pics/face_' + t + '_1.png', syn)

                    pred_coords_copy = copy.deepcopy(pred_coords)
                    for i in range(kp_num[use_dataset]):
                        pred_coords_copy[2 * i] = \
                            bbox[0] + pred_coords_copy[2 * i] / 255 * (bbox[2] - bbox[0])
                        pred_coords_copy[2 * i + 1] = bbox[1] + pred_coords_copy[2 * i + 1] / 255 * (
                                    bbox[3] - bbox[1])
                    face_keypoint_coords.append(pred_coords_copy)

            if len(face_keypoint_coords) != 0:
                for face_id, coords in enumerate(face_keypoint_coords):
                    for kp_index in range(kp_num[use_dataset]):
                        cv2.circle(
                            img,
                            (int(coords[2 * kp_index]), int(coords[2 * kp_index + 1])),
                            2,
                            (0, 0, 255),
                            -1
                        )
                show_img(img, 'face_whole', 1400, 650)
                cv2.imwrite('./pics/face_' + t_start + '.png', img)
                face_keypoint_coords = []

        if k == ord('q') or k == ord('Q'):
            break

print('QUIT.')
if os.path.exists('hm.png'):
    os.remove('hm.png')
cap.release()              # 关闭摄像头
cv2.destroyAllWindows()

