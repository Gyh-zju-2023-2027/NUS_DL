import cv2
import json
import signal
import sys
from extract_pose import PoseDetector           # 确保 extract_pose.py 在同级目录
from stroke_detection import get_pose_10hz_6lm    # 确保 stroke_detection.py 在同级目录

# 全局标志，用于 signal handler
running = True

def signal_handler(sig, frame):
    global running
    print("\n捕获到中断信号，准备停止采集…")
    running = False

def main():
    global running
    # 注册 Ctrl-C 信号
    signal.signal(signal.SIGINT, signal_handler)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: 无法打开摄像头")
        sys.exit(1)

    detector = PoseDetector()
    frames_data = []

    print("开始采集姿态数据，按 'q' 键或 Ctrl-C 停止…")
    try:
        while running:
            ok, img = cap.read()
            if not ok:
                print("Error: 读取摄像头帧失败，退出采集")
                break

            # 可选：对图像应用掩膜，只保留左半区
            # mask = create_mask(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            #                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            # mask_3d = cv2.merge([mask, mask, mask])
            # img = cv2.bitwise_and(img, mask_3d)

            # 提取姿态并在图像上绘制关键点
            img, _ = detector.findPose(img, draw=True)
            lmList = detector.findPosition(img)
            frames_data.append({'landmarks': lmList})

            cv2.imshow("Webcam Pose Extraction", img)
            # 按 'q' 停止
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("检测到 'q' 键，停止采集")
                break

    finally:
        # 退出循环后，无论因为什么原因，都执行清理和后处理
        cap.release()
        cv2.destroyAllWindows()

        # 1. 保存原始姿态数据
        pose_json = "../../../../sampledata/test/player_01/pose_data.json"
        with open(pose_json, "w", encoding="utf-8") as f:
            json.dump(frames_data, f, ensure_ascii=False, indent=2)
        print(f"[保存] 姿态数据 → {pose_json}")

        # 2. 调用击打峰值检测并可视化
        output_path = "../../../../sampledata/test/player_01/aligned_downsampled.json"
        get_pose_10hz_6lm(frames_data, output_path)
        print(f"[完成] 峰值结果 → {output_path} 及同名可视化图像")

if __name__ == "__main__":
    main()
