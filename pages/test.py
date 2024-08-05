import streamlit as st
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

st.subheader('재활용품 분류기')
with st.container():
    st.info('''AI를 활용하여 재활용품 항목을 분류합니다.  
    이미 저장된 이미지를 업로드하거나, 직접 실시간으로 사진을 찍어서 분류해볼 수 있습니다.''')

st.write('')
tab1, tab2 = st.tabs(['이미지 업로드', '사진 촬영'])

# 이미지 업로드
with tab1:
    model = YOLO('model/best.pt')  # 사전 학습된 모델 사용
    uploaded_file = st.file_uploader("이미지 파일을 등록해주세요", type=['png', 'jpg', 'jpeg'])
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        # 이미지를 PIL 형식으로 읽기
        image = Image.open(uploaded_file)  # PIL을 사용해 이미지 읽기
        image_np = np.array(image)  # PIL 이미지를 NumPy 배열로 변환

        # 모델로 예측 수행
        results = model(image_np)

        # 결과를 이미지에 그리기
        annotated_image = results[0].plot()  # YOLOv8 모델이 그린 이미지

        # 예측 결과 후처리
        detections = results[0].boxes  # 각 객체의 박스 정보
        labels = model.names  # 클래스 이름

        # matplotlib로 순서 번호 추가
        fig, ax = plt.subplots()
        ax.imshow(annotated_image)

        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 박스의 좌표
            sequence_number = i + 1  # 이미지 안에서 순서에 따라 번호 부여
            text = f"{sequence_number}"

            # 텍스트 그리기: 박스 위에 순서 번호를 추가
            ax.text((x1 + x2) / 2, y1 - 10, text, color='red', fontsize=12, weight='bold',
                    ha='center', bbox=dict(facecolor='white', alpha=0.7))

        plt.axis('off')  # 축 숨기기

        # matplotlib 이미지를 파일로 저장 후 Streamlit에 표시
        img_path = "temp_annotated_image.png"
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)  # plt.close()로 메모리 해제

        # 원본 이미지와 결과 이미지 출력
        col1.image(image, caption="원본 이미지")  # 원본 이미지 표시
        col2.image(img_path, caption="재활용 분류 결과")  # 결과 이미지 표시
