import tensorflow.keras
import numpy as np
import cv2
import winsound
import time

def beep(frequency,duration):
    winsound.Beep(frequency, duration)

frequency = 2500
duration = 100

# 모델 위치
model_filename ='F:\\converted_keras\\keras_model.h5'

# 케라스 모델 가져오기
model = tensorflow.keras.models.load_model(model_filename)

# 카메라를 제어할 수 있는 객체
capture = cv2.VideoCapture(0)

# 카메라 길이 너비 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 이미지 처리하기
def preprocessing(frame):
    #frame_fliped = cv2.flip(frame, 1)
    # 사이즈 조정 티쳐블 머신에서 사용한 이미지 사이즈로 변경해준다.
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    
    # 이미지 정규화
    # astype : 속성
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1

    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    # keras 모델에 공급할 올바른 모양의 배열 생성
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    #print(frame_reshaped)
    return frame_reshaped

# 예측용 함수
def predict(frame):
    prediction = model.predict(frame)
    return prediction

while True:
    ret, frame = capture.read()

    if cv2.waitKey(22) > 0: 
        break

    preprocessed = preprocessing(frame)
    prediction = predict(preprocessed)

    if (prediction[0,0] < prediction[0,1]):
        print('hand off')
        cv2.putText(frame, 'hand off', (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

    else:
        cv2.putText(frame, 'hand on', (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        print('hand on')
        beep(frequency, duration)

    cv2.imshow("VideoFrame", frame)
