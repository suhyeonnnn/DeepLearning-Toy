import tensorflow as tf
from MLP import MLP

def xor_classifier_example():
    input_data = tf.constant([[0.0, 0.0],
                              [0.0, 1.0],
                              [1.0, 0.0],
                              [1.0, 1.0]], dtype=tf.float32)

    xor_labels = tf.constant([[0.0], [1.0], [1.0], [0.0]], dtype=tf.float32)  # reshape to match output shape

    batch_size = 1
    epochs = 1500

    mlp_classifier = MLP(hidden_layer_conf=[4], num_output_nodes=1)
    mlp_classifier.build_model()
    mlp_classifier.fit(x=input_data, y=xor_labels, batch_size=batch_size, epochs=epochs)

    # Prediction
    prediction = mlp_classifier.predict(x=input_data, batch_size=batch_size)

    print("====== MLP XOR classifier result =====")
    for i in range(len(input_data)):
        x = input_data[i].numpy()
        y = prediction[i][0]  # prediction is a 2D array
        result = 1 if y > 0.5 else 0
        print(f"{int(x[0])} XOR {int(x[1])} => {y:.2f} => {result}")

# Entry point
if __name__ == '__main__':
    xor_classifier_example()
