# MLP (Multi-Layer Perceptron) 실습

## 개요
MNIST 손글씨 숫자 데이터를 이용해 다층 퍼셉트론(MLP)을 구현하여 분류 작업을 수행했습니다.  
TensorFlow를 활용해 모델을 구성하고, 정확도 및 손실을 시각화했습니다.

## 사용 기술
- Python
- TensorFlow / Keras
- MNIST Dataset
- Dense Layer, ReLU, Softmax

## 모델 구조
- 입력층: 784 (28x28 이미지 평탄화)
- 은닉층: Dense(128) + ReLU, Dropout(optional)
- 출력층: Dense(10) + Softmax

## 실행 방법
```bash
python main.py
