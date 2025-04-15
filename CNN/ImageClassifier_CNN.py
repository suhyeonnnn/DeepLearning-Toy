import tensorflow as tf

def to_onehotvec_label(label, num_classes):
    return tf.keras.utils.to_categorical(label, num_classes)

class ImageClassifier_CNN:
    def __init__(self, img_shape_x, img_shape_y, num_labels):
        self.img_shape_x = img_shape_x
        self.img_shape_y = img_shape_y
        self.num_labels = num_labels
        self.model = None

    def build_CNN_model(self):
        input_layer = tf.keras.Input(shape=[self.img_shape_x, self.img_shape_y, 1])
        hidden = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
        hidden = tf.keras.layers.MaxPooling2D((2, 2))(hidden)
        hidden = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(hidden)
        hidden = tf.keras.layers.MaxPooling2D((2, 2))(hidden)
        hidden = tf.keras.layers.Flatten()(hidden)
        hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)
        output = tf.keras.layers.Dense(self.num_labels, activation='softmax')(hidden)

        self.model = tf.keras.Model(inputs=input_layer, outputs=output)
        self.model.summary()

        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

    def fit(self, x, y, num_epochs=5):
        self.model.fit(x, y, batch_size=32, epochs=num_epochs, validation_split=0.1)

    def predict(self, x):
        return self.model.predict(x)
