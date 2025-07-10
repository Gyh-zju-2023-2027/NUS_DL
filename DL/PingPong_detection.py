import cv2
import numpy as np
from prediction import predict_trajectory_lstm

def preprocess(image,lower_yellow,upper_yellow):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow ,upper_yellow)
    return mask

def nothing(x): pass

def creat_Trackbar():
    cv2.createTrackbar('SeXiang_L', 'Trackbar', 20, 179, nothing)
    cv2.createTrackbar('SeXiang_H', 'Trackbar', 35, 179, nothing)
    cv2.createTrackbar('BaoHeDu_L', 'Trackbar', 100, 255, nothing)
    cv2.createTrackbar('BaoHeDu_H', 'Trackbar', 255, 255, nothing)
    cv2.createTrackbar('Zhi_L', 'Trackbar', 100, 255, nothing)
    cv2.createTrackbar('Zhi_H', 'Trackbar', 255, 255, nothing)

def read_Trackbar():
    SeXiang_L = cv2.getTrackbarPos('SeXiang_L', 'Trackbar')
    SeXiang_H = cv2.getTrackbarPos('SeXiang_H', 'Trackbar')
    BaoHeDu_L = cv2.getTrackbarPos('BaoHeDu_L', 'Trackbar')
    BaoHeDu_H = cv2.getTrackbarPos('BaoHeDu_H', 'Trackbar')
    Zhi_L = cv2.getTrackbarPos('Zhi_L', 'Trackbar')
    Zhi_H = cv2.getTrackbarPos('Zhi_H', 'Trackbar')
    return np.array([SeXiang_L, BaoHeDu_L, Zhi_L]), np.array([SeXiang_H, BaoHeDu_H, Zhi_H])

def show_image():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Trackbar', 2)
    creat_Trackbar()

    trajectory = []  # 存储圆心坐标

    while True:
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.GaussianBlur(image, (5, 5), 0)
        cv2.imshow('Trackbar', image)

        lower_yellow, upper_yellow = read_Trackbar()
        mask = preprocess(image, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(image, image, mask=mask)
        res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(res_gray, cv2.HOUGH_GRADIENT, 1, 1000000, param1=100, param2=11, minRadius=10, maxRadius=100)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                x, y, r = int(i[0]), int(i[1]), int(i[2])
                trajectory.append((x, y))
                cv2.circle(image, (x, y), r, (255, 255, 255), 2)
                cv2.circle(image, (x, y), 2, (255, 255, 255), 2)
                print("圆心坐标为：", (x, y), "半径：", r)

        # 显示图像
        cv2.imshow('res', res_gray)
        cv2.imshow('image', image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC退出
            break
        elif key == ord('p'):  # 按 P 键预测
            print(f"\n[LSTM] 当前轨迹点数：{len(trajectory)}，开始拟合预测...")
            predict_trajectory_lstm(trajectory, future_steps=5)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    show_image()