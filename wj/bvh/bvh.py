from pose_estimator_2d import openpose_estimator
from pose_estimator_3d import estimator_3d
from utils import smooth, vis, camera
from bvh_skeleton import openpose_skeleton, h36m_skeleton, cmu_skeleton

import cv2
import importlib
import numpy as np
import os
from pathlib import Path
from IPython.display import HTML





# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# BVH 파일 생성을 위한 초기화
#bvh_writer = o3d.io.BVHWriter()

# BVH 파일 설정
output_bvh = "output_pose.bvh"
frame_rate = 30  # 프레임 속도 설정

# 웹캠 캡처 설정
cap = cv2.VideoCapture(0)

# 포즈를 저장할 빈 리스트
pose_data = []
total_frames = 200

# BVH 파일을 저장하기 위한 함수
def write_bvh(filename, frames, frame_rate):
    with open(filename, 'w') as f:
        f.write(f'HIERARCHY\n')
        f.write(f'ROOT Body\n')
        f.write(f'{len(frames)}\n')  # 총 프레임 수
        f.write(f'Frame Time: {1.0 / frame_rate}\n')

        # 각 프레임의 데이터 작성
        for frame in frames:
            f.write(' '.join([str(val) for val in frame]) + '\n')


# 미디어 파이프를 사용하여 비디오 스트림 또는 이미지를 읽고 포즈 인식
with mp_pose.Pose() as pose:
    # 여기에서 비디오 스트림 또는 이미지를 읽어올 수 있습니다.
    # 루프를 통해 각 프레임에서 포즈를 감지하고 저장
    for frame_number in range(total_frames):  # 총 프레임 수를 적절히 설정해야 합니다.
        # 비디오 프레임 또는 이미지를 frame 변수에 할당
        success, frame = cap.read()

        # MediaPipe를 사용하여 포즈 인식
        results = pose.process(frame)

        # 포즈 데이터를 BVH 파일에 추가
        if results.pose_landmarks:
            # 미디어 파이프의 랜드마크 정보를 포즈 데이터로 변환
            pose_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
            pose_data.append(pose_landmarks.flatten())  # 데이터를 1차원으로 평면화



# BVH 파일에 저장
write_bvh(output_bvh, pose_data, frame_rate)

# 자원 정리
pose.close()
