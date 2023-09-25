from flask import Flask, Blueprint, render_template, Response, redirect, url_for

from examples.image_to_annotations import image_to_annotations
from examples.annotations_to_animation import annotations_to_animation
from pathlib import Path
import logging
import sys
from pkg_resources import resource_filename
from animated_drawings import render

import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# 웹캠 캡처 설정
cap = cv2.VideoCapture(0)


# BVH 파일 설정
bvh_file = "output_pose.bvh"
frame_rate = 30  # 프레임 속도 설정

# 포즈를 저장할 빈 리스트
bvh_frames = []
#recording = False

bp = Blueprint("main", __name__, url_prefix="/")

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

def image_to_animation(img_fn: str, char_anno_dir: str, motion_cfg_fn: str, retarget_cfg_fn: str):
    """
    Given the image located at img_fn, create annotation files needed for animation.
    Then create animation from those animations and motion cfg and retarget cfg.
    """
    # create the annotations
    image_to_annotations(img_fn, char_anno_dir)

    # create the animation
    annotations_to_animation(char_anno_dir, motion_cfg_fn, retarget_cfg_fn)

    #render.start('./examples/config/mvc/rokoko_motion_example.yaml')

@bp.route('/start_recording', methods=['POST'])
def start_recording():
    global bvh_frames, recording
    bvh_frames = []  # 녹화를 시작할 때 데이터 초기화
    recording = True
    print("start")
    return jsonify({'recording': recording})
    #return redirect(url_for('main.main_get'))

@bp.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    recording = False
    print("stop")
    write_bvh(bvh_file, bvh_frames, frame_rate)  # BVH 파일 저장
    return jsonify({'recording': recording})
    #return redirect(url_for('main.main_get'))

@bp.route('/')
def main_get():

    # 미디어 파이프 _ BVH 파일 얻기

    # BVH to yaml

    # 애니메이션 파일로 변환
    '''
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.DEBUG)

    img_fn = sys.argv[1]
    char_anno_dir = sys.argv[2]
    if len(sys.argv) > 3:
        motion_cfg_fn = sys.argv[3]
    else:
        motion_cfg_fn = resource_filename(__name__, 'config/motion/dab.yaml')
    if len(sys.argv) > 4:
        retarget_cfg_fn = sys.argv[4]
    else:
        retarget_cfg_fn = resource_filename(__name__, 'config/retarget/fair1_ppf.yaml')

    image_to_animation(img_fn, char_anno_dir, motion_cfg_fn, retarget_cfg_fn)
    '''

    return render_template('index.html')

def generate_frames():
    while True:
        success, frame = cap.read()  # 웹캠에서 프레임 읽기
        if not success:
            break
        else:
            # 미디어 파이프를 사용하여 포즈 인식
            results = pose.process(frame)

            # 포즈 랜드마크를 이미지에 그리기
            
            if results.pose_landmarks:
                #mp.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                landmarks = results.pose_landmarks.landmark
                pose_data = []
                for landmark in landmarks:
                    pose_data.extend([landmark.x, landmark.y, landmark.z]) 

                bvh_frames.append(pose_data)

            # 프레임을 JPEG 이미지로 인코딩하여 반환
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')