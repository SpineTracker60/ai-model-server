#시작 할때 pip install mediapipe
from mediapipe.python.solutions import face_detection as fd, hands as hd, face_mesh as fm, drawing_utils, drawing_styles
from PIL import ImageFont,ImageDraw,Image
import mediapipe as mp
import cv2
import numpy as np
import time
import math
import os

import winsound as sd
def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 100     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

# 만약 하위 폴더가 없으면 하위 폴더 생성
if not os.path.exists('./face data'):
    os.makedirs('./face data')
else:
    pass

mp_face_mesh = fm
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

#원하는 지점의 숫자
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
FACE_LINE= [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
SHOULDER_LINE = [11,12]
# 시간비교를 위해 시작 시간 기록
startTime = time.time()
O_piv = 0
# 글씨 스타일
font = ImageFont.truetype('NanumGothic.ttf', 20)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        break

    # 보기 편하기 위해 이미지를 좌우를 반전하고, BGR 이미지를 RGB로 변환
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    img_h, img_w = image.shape[:2]

    with mp_face_mesh.FaceMesh(
        #얼굴 감지 인원 수
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            # 얼굴에 지점좌표를 np.array형태로 변환
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                                    results.multi_face_landmarks[0].landmark])

            # 얼굴 높이는 맨위의 y값과 맨아래의 y값의 차
            face_height = mesh_points[10][1] - mesh_points[152][1]
            # 얼굴 너비는 맨왼쪽의 x값과 맨오른쪽의 x값의 차
            face_width = mesh_points[234][0] - mesh_points[454][0]
            # 얼굴 넓이
            face_size = face_height * face_width
            # 눈이 감았는지 여부를 알기 위해 눈의 높이
            # 뒤에 (1/int(1e9))엄청 작은 값을 추가시킨것은 eye_height값이 eye_ratio에서 분모로 들어가는데 이가 0이됨을 방지하기 위함
            eye_height = ((mesh_points[159][1] - mesh_points[145][1]) + (
                        mesh_points[386][1] - mesh_points[374][1])) + (1 / int(1e9)) / 2
            # 얼굴에서 눈이 차지하는 비율 만약에 그냥 eye_height로 사용하게되면 얼굴이 뒤로 갔을때 눈이 작아지면 감은것으로 인식함
            # 하지만 얼굴비율로 하게 되면 뒤로가도 비율은 유지
            eye_ratio = (face_height / eye_height)
            # 현재 시간 갱신
            endTime = time.time()

            # 글씨를 넣을 위치
            org = (10, 30)
            # 시작한지 5초가 지나기전
            if endTime - startTime < 4.7:
                # 양눈, 얼굴을 그려줌
                cv2.polylines(image, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(image, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(image, [mesh_points[FACE_LINE]], True, (0, 255, 0), 1, cv2.LINE_AA)

                # 글씨 추가
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)
                # math.floor(내림)을 사용해 5초부터 CountDown
                text = f'{(5 - math.floor(endTime - startTime))}초후에 바른자세 사진을 찍습니다.'
                draw.text(org, text, font=font, fill=(0, 255, 0))
                # 5초가 지났을 때 이미지를 기본 얼굴로 저장
            elif O_piv == 0 and 4.7 <= endTime - startTime < 5:
                image = Image.fromarray(image)
                O_piv = 1
                default_eye = eye_ratio
                default_face = face_size
                image.save('./face data/Original.jpg', 'JPEG')
            # 5초가 지난후에는
            else:
                cv2.polylines(image, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(image, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(image, [mesh_points[FACE_LINE]], True, (0, 255, 0), 1, cv2.LINE_AA)
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)
                # 눈이 평소일때 반까지 뜬것까진 괜찮
                if eye_ratio < 2 * default_eye:
                    # 눈을 감고있던 시간 측정 을 위해 마지막 눈을 뜬 시간 기록
                    last_eye_open = endTime
                # 눈을 감았다!
                else:
                    # 눈을 감고있던 시간 측정을 위해 마지막 눈을 감은 시간 기록
                    last_eye_close = endTime
                    # 눈을 감았다면 ? (아직 3초 안됨)
                    if last_eye_close - last_eye_open < 3:
                        text = '졸음 의심'
                        draw.text(org, text, font=font, fill=(100, 0, 0))
                    # 눈을 감은지 3초가 지남...!
                    else:
                        beepsound()
                        text = '꾸벅꾸벅 조는중!!!'
                        draw.text(org, text, font=font, fill=(255, 0, 0))
                # 얼굴의 넓이가 기본사진에 비해 1.1배 커짐 > 얼굴이 앞으로 나와있음 > 거북목!
                if face_size < 1.1 * default_face:
                    # 거북목이 아닌상태
                    last_not_turtle = endTime
                    # 거북목이 아니라면 1
                    piv_turtle = 1
                else:
                    # 거북목인상태
                    last_turtle = endTime
                    # 거북목 의심
                    org = (10, 50)
                    if last_turtle - last_not_turtle < 3:
                        text = '거북목 의심'
                        draw.text(org, text, font=font, fill=(0, 100, 0))
                    else:
                        beepsound()
                        text = '거북거북 거북목'
                        draw.text(org, text, font=font, fill=(0, 255, 0))
                        # 사진은 거북목일때 한번만 찍기
                        if piv_turtle:
                            piv_turtle = 0
                            T = math.floor(endTime - startTime)
                            image.save(f'./face data/{T // 3600}시{(T % 3600) // 60}분{T % 60}초.jpg', 'JPEG')

            image = np.array(image)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        results = pose.process(image)
        # 추정된 포즈 결과가 있으면 화면에 그리기
        if results.pose_landmarks:
            #print(results.pose_landmarks)
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                                    results.pose_landmarks.landmark])
            # 어깨 높이 차
            shoulder_abs = abs(mesh_points[11][1] - mesh_points[12][1])
            cv2.polylines(image, [mesh_points[SHOULDER_LINE]], True, (0, 255, 0), 1, cv2.LINE_AA)
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            text = f'{shoulder_abs}'
            org = (10, 70)
            #draw.text(org, text, font=font, fill=(255, 0, 0))
            image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
