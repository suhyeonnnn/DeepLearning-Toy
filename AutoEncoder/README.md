# AutoEncoder 실습

## 개요
AutoEncoder 모델을 구현하여 이미지 데이터(MNIST)를 압축하고 복원하는 과정을 실습했습니다.  
입력과 출력이 동일한 구조의 비지도 학습 모델을 구성하였습니다.

## 사용 기술
- Python
- TensorFlow / Keras
- Dense Layer, ReLU, Sigmoid
- Mean Squared Error Loss

## 모델 구조
입력층 (784) → 인코더(Dense 128) → 디코더(Dense 784)

## 실행 방법
```bash
python main.py
