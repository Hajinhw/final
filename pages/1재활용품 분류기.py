import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np


st.subheader('재활용품 분류기')
with st.container(border=False):
    st.info(''' AI를 활용하여 재활용품 항목을 분류합니다.  
    이미 저장된 이미지를 업로드하거나, 직접 실시간으로 사진을 찍어서 분류해볼 수 있습니다.''')

st.write('')
tab1, tab2 = st.tabs(['이미지 업로드', '사진 촬영'])

# 이미지 업로드
with tab1:
    model = YOLO('model/best.pt')  # 사전 학습된 모델 사용
    col1, col2 = st.columns(2)
    uploaded_file = col1.file_uploader("이미지 파일을 등록해주세요", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        col1.image(uploaded_file)

        # 이미지를 OpenCV 형식으로 읽기
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # 모델로 예측 수행
        results = model(image)

        # 결과를 이미지에 그리기
        annotated_image = results[0].plot()

        # 결과 이미지 출력
        col2.header('')
        col2.header('')
        col2.markdown('')
        col2.text('')
        col2.image(annotated_image, channels="BGR", caption="재활용 분류 결과")


# 사진 촬영
with tab2:
    picture = st.camera_input("사진을 찍어주세요")
    model = YOLO('model/best.pt')
    if picture:

        # 이미지를 OpenCV 형식으로 읽기
        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        image2 = cv2.imdecode(file_bytes, 1)

        # 모델로 예측 수행
        results2 = model(image2)

        # 결과를 이미지에 그리기
        annotated_image2 = results2[0].plot()

        # 결과 이미지 출력
        st.text('')
        st.image(annotated_image2, channels="BGR", caption="재활용 분류 결과")