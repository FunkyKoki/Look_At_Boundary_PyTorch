import cv2
import dlib
import time
import numpy as np
from dataset import GeneralDataset
from models import *
from utils import *


# load network
devices = torch.device('cuda:0')
print('*****  WFLW trained Model Evaluating  *****')
print('Loading network ...')
estimator = Estimator()
regressor = Regressor(output=2*kp_num[args.dataset])
estimator = load_weights(estimator, 'estimator_750.pth', devices)
regressor = load_weights(regressor, 'WFLW_regressor_750.pth', devices)
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
    ret, img = cap.read()  # 把摄像头获取的图像信息保存之img变量
    if ret is True:        # 如果摄像头读取图像成功
        cv2.imshow('Image', img)
        k = cv2.waitKey(100)
        if k == ord('c') or k == ord('C'):
            face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
            rec = face_detector(img, 1)

            if len(rec) == 0:
                print('No Face Detected!')

            with torch.no_grad():
                for face_i in range(len(rec)):
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

                    watch_pic_kp(args.dataset, face_img, pred_coords)

                    face_keypoint_coords.append(inverse_affine(args, pred_coords, bbox))

        if k == ord('q') or k == ord('Q'):
            break

cap.release()              # 关闭摄像头
cv2.waitKey(0)
cv2.destroyAllWindows()

