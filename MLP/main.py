import tensorflow as tf
import matplotlib.pyplot as plt
from ImageClassifier_MLP import ImageClassifier_MLP

def run_classifier():
    # 1. 데이터 로드
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # 2. 클래스 이름 정의
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 3. 정규화 (0 ~ 1)
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 4. One-hot encoding
    train_labels_oh = ImageClassifier_MLP.to_onehotvec_label(train_labels, dim=10)

    # 5. 모델 초기화 및 빌드
    my_classifier = ImageClassifier_MLP(28, 28, 10)
    my_classifier.build_MLP_model()

    # 6. 학습
    my_classifier.fit(train_imgs=train_images, train_labels=train_labels_oh, num_epochs=10)

    # 7. 예측
    predicted_probs = my_classifier.predict(test_imgs=test_images)
    predicted_labels = tf.math.argmax(predicted_probs, axis=1)

    # 8. 시각화 (25개 샘플)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[predicted_labels[i]])
    plt.tight_layout()
    plt.savefig("results/sample_predictions.png")  # 이미지 저장도 가능!
    plt.show()


if __name__ == "__main__":
    run_classifier()
