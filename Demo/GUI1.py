from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
import cv2
import os
import os.path as osp
import imutils
import numpy as np
import torch
import time
import pandas as pd
from PoseEstimation.Hrnet import Hrnet
from PoseEstimation.Yolov7 import Yolov7
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from SCL import TransformerModel
import pickle
from sklearn.svm import SVC
from scipy.spatial.distance import cdist



def align(query_feats, candidate_feats):
    """Align videos based on nearest neighbor."""

    dists = cdist(query_feats, candidate_feats, 'sqeuclidean')
    nns = np.argmin(dists, axis=1)
    return nns, dists


def resize_img(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size
    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC
    # aspect ratio of image
    aspect = w/h
    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3
    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    return scaled_img

class Pose_detect_thread(QThread):
    pose_results = pyqtSignal(list)
    progressing = pyqtSignal(int)


    def __init__(self, Yolov7, Hrnet, video):
        super(Pose_detect_thread, self).__init__()
        self.Yolov7 = Yolov7
        self.Hrnet = Hrnet
        self.video = video
        self.time_process = 0


    def run(self):
        cnt = 0
        total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_results = []
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        flag, image = self.video.read()
        while flag:
            detections = self.Yolov7.inference(image)
            pose_res = self.Hrnet.inference_from_bbox(image, detections)
            cnt += 1
            self.progressing.emit(cnt)
            pose_results.append(pose_res)
            flag, image = self.video.read()
        self.pose_results.emit(pose_results)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)



class My_GUI(QMainWindow):

    def __init__(self):
        super(My_GUI, self).__init__()
        uic.loadUi('form1.ui', self)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)

        self.gridLayout.addWidget(self.canvas)

        self.pose_results = None
        self.nns = None
        self.model_yolov7 = Yolov7(engine_path='PoseEstimation/yolov7-tiny-nms.trt')
        self.model_hrnet = Hrnet(engine_path='PoseEstimation/HR_net48.trt')

        self.quit = QAction("Quit", self)
        self.quit.triggered.connect(self.closeEvent)

        self.thread = Pose_detect_thread(Yolov7=self.model_yolov7, Hrnet=self.model_hrnet, video=None)
        # self.thread.pose_results.connect(self.receive_pose_result)

        self.model = TransformerModel()
        self.model.load_state_dict(torch.load('model_180.pth'))
        self.model.eval()
        self.model.to('cuda')

        self.video_path1 = ''
        self.video_path2 = ''
        self.load_video_btn1.clicked.connect(self.load_video1)
        self.load_video_btn2.clicked.connect(self.load_video2)
        self.detect_pose_btn1.clicked.connect(self.pose_detect1)
        self.detect_pose_btn2.clicked.connect(self.pose_detect2)
        self.compare_btn.clicked.connect(self.extract_embeddings)
        self.frame_no_slider.valueChanged.connect(self.set_frame_no_slider)

    def pose_detect1(self):
        self.thread.video = self.video1
        self.thread.progressing.connect(self.progressing_bar1)
        self.thread.pose_results.connect(self.receive_pose_result1)
        self.thread.start()

    def pose_detect2(self):
        self.thread.video = self.video2

        self.thread.progressing.disconnect(self.progressing_bar1)
        self.thread.progressing.connect(self.progressing_bar2)

        self.thread.pose_results.disconnect(self.receive_pose_result1)
        self.thread.pose_results.connect(self.receive_pose_result2)
        self.thread.start()

    def progressing_bar1(self, value):
        value = int(value * 100 / self.total_frame1)
        self.progressBar1.setValue(value)

    def progressing_bar2(self, value):
        value = int(value * 100 / self.total_frame2)
        self.progressBar2.setValue(value)

    def extract_embeddings(self):
        kp1 = []
        for pose in self.pose_results1:
            kp1.append(pose['keypoints'])
        kp1 = np.array(kp1)

        h1, w1 = self.image_size1
        x1 = kp1
        x1[:, :, 0] = x1[:, :, 0]/w1
        x1[:, :, 1] = x1[:, :, 1]/h1
        x1 = torch.from_numpy(x1).float()

        kp2 = []
        for pose in self.pose_results2:
            kp2.append(pose['keypoints'])
        kp2 = np.array(kp2)

        h2, w2 = self.image_size2
        x2 = kp2
        x2[:, :, 0] = x2[:, :, 0]/w2
        x2[:, :, 1] = x2[:, :, 1]/h2
        x2 = torch.from_numpy(x2).float()

        self.embs1 = self.model(x1[None].to('cuda'))
        self.embs2 = self.model(x2[None].to('cuda'))
        self.embs1 = self.embs1[0].detach().cpu().numpy()
        self.embs2 = self.embs2[0].detach().cpu().numpy()
        nns, dist = align(self.embs1, self.embs2)
        self.nns = nns
        distance = []
        for i in range(len(nns)):
            distance.append(dist[i][nns[i]])
        self.distance = distance
        self.canvas.axes.cla()
        self.canvas.axes.plot(range(len(distance)), distance)
        self.canvas.axes.set_xlabel('Frame #')
        self.canvas.axes.plot(self.frame_no_slider.value(), distance[self.frame_no_slider.value()], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")

        self.canvas.axes.set_ylabel('Distance')
        self.canvas.draw()


    def set_frame_no_slider(self, value):
        self.video1.set(cv2.CAP_PROP_POS_FRAMES, value)
        _, frame = self.video1.read()
        self.curr_frame1 = frame
        if self.checkBox1.isChecked():
            self.curr_frame1 = self.vis_pose(frame, self.pose_results1[value])
        self.video2.set(cv2.CAP_PROP_POS_FRAMES, self.nns[value])
        _, frame = self.video2.read()
        self.curr_frame2 = frame
        if self.checkBox2.isChecked():
            self.curr_frame2 = self.vis_pose(frame, self.pose_results2[self.nns[value]])
        if self.nns is not None:
            self.canvas.axes.cla()
            self.canvas.axes.plot(range(len(self.distance)), self.distance)
            self.canvas.axes.set_xlabel('Frame #')
            self.canvas.axes.plot(value, self.distance[value], marker="o", markersize=5,
                                  markeredgecolor="red", markerfacecolor="green")

            self.canvas.axes.set_ylabel('Distance')
            self.canvas.draw()

        self.image1_set(self.curr_frame1)
        self.image2_set(self.curr_frame2)


    def receive_pose_result1(self, pose_results):
        self.pose_results1 = pose_results

    def receive_pose_result2(self, pose_results):
        self.pose_results2 = pose_results
    

    def load_video1(self):
        self.video_path1 = QtWidgets.QFileDialog.getOpenFileName(self,'Open video file', filter='Video files (*.mp4 *.mkv, *.avi)')[0]
        if len(self.video_path1) == 0:
            return
        self.video1 = cv2.VideoCapture(self.video_path1)
        self.total_frame1 = int(self.video1.get(cv2.CAP_PROP_FRAME_COUNT))
        _, self.curr_frame1 = self.video1.read()
        self.image_size1 = self.curr_frame1.shape[:2]
        self.frame_no_slider.setRange(0, int(self.total_frame1) - 1)
        self.frame_no_slider.setValue(0)
        self.image1_set(self.curr_frame1)
        self.pose_results1 = None


    def load_video2(self):
        self.video_path2 = QtWidgets.QFileDialog.getOpenFileName(self,'Open video file', filter='Video files (*.mp4 *.mkv, *.avi)')[0]
        if len(self.video_path2) == 0:
            return
        self.video2 = cv2.VideoCapture(self.video_path2)
        self.total_frame2 = int(self.video2.get(cv2.CAP_PROP_FRAME_COUNT))
        _, self.curr_frame2 = self.video2.read()
        self.image_size2 = self.curr_frame2.shape[:2]

        self.image2_set(self.curr_frame2)
        self.pose_results = None


    def image1_set(self, image):
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_img(image, (650, 570))
        image_Qt = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        self.image_label_1.setPixmap(QPixmap.fromImage(image_Qt))

    def image2_set(self, image):
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_img(image, (650, 570))
        image_Qt = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        self.image_label_2.setPixmap(QPixmap.fromImage(image_Qt))


    def vis_pose(self, image, pose_result, threshold=0.3):
        bbox = pose_result['bbox']
        keypoints = pose_result['keypoints'][:,:2]
        keypoints_score = pose_result['keypoints'][:,2]

        skeleton_edge = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                         [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                         [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                         [3, 5], [4, 6]]
        for edge in skeleton_edge:
            start = keypoints[edge[0]]
            end = keypoints[edge[1]]
            if keypoints_score[edge[0]] < threshold or keypoints_score[edge[1]] < threshold:
                continue
            image = cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 255, 0), 2)

        for i in range(17):
            if keypoints_score[i] < threshold:
                continue
            (x, y) = keypoints[i]
            color = (255, 255, 255)

            image = cv2.circle(image, (int(x), int(y)), 5, color, -1)

        image_vis = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
        return image_vis

    def closeEvent(self, event):
        self.model_yolov7.destory()
        self.model_hrnet.destory()
        event.accept()

def main():
    app = QApplication([])
    window = My_GUI()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()