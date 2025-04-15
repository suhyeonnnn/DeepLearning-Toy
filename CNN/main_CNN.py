import tensorflow as tf
import matplotlib.pyplot as plt
from ImageClassifier_CNN import ImageClassifier_CNN, to_onehotvec_label
import numpy as np

def run_classifier():
    # 1. 데이터 로딩
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print("Train data shape:", train_images.shape)
    print("Train labels:", train_labels)
    print("Test data shape:", test_images.shape)
    print("Test labels:", test_labels)

    # 2. 이미지 시각화
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # 3. 정규화
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 4. CNN용 채널 추가 (28x28 → 28x28x1)
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    # 5. 레이블 원-핫 인코딩
    num_classes = 10
    train_labels_oh = to_onehotvec_label(train_labels, num_classes)
    test_labels_oh = to_onehotvec_label(test_labels, num_classes)

    # 6. 샘플 이미지 25장 시각화
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

    ##### classifier train and predict – begin
    classifier = ImageClassifier_CNN(28, 28, num_classes)
    classifier.build_CNN_model()
    classifier.fit(train_images, train_labels_oh, num_epochs=5)

    predictions = classifier.predict(test_images)
    print("첫 번째 테스트 이미지 예측 결과:")
    print(predictions[0])
    ##### classifier train and predict – end

# Entry point
if __name__ == "__main__":
    run_classifier()

