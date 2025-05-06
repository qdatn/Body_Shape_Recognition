import tensorflow as tf
from app.data_preprocessing import prepare_data
from app.body_shape_detector import BodyShapeDetector

def train_model(train_dir, test_dir):
    # Chuẩn bị dữ liệu huấn luyện và kiểm tra
    train_generator, validation_generator = prepare_data(train_dir, test_dir)

    # Xây dựng mô hình
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')  # Số lớp đầu ra = số kiểu dáng cơ thể
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Huấn luyện mô hình
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )

    # Lưu mô hình đã huấn luyện
    model.save('models/body_shape_model.h5')

def main():
    # Đường dẫn đến dữ liệu huấn luyện và kiểm tra
    train_dir = 'data/train'
    test_dir = 'data/test'

    # Huấn luyện mô hình
    train_model(train_dir, test_dir)

    # Sử dụng mô hình đã huấn luyện để nhận diện kiểu dáng cơ thể
    detector = BodyShapeDetector('models/body_shape_model.h5')
    image_path = 'data/test/sample_image.jpg'  # Hình ảnh cần dự đoánw
    body_shape = detector.detect(image_path)
    print(f'Predicted Body Shape: {body_shape}')

if __name__ == '__main__':
    main()


# from app.body_shape_detector import BodyShapeDetector

# def main():
#     detector = BodyShapeDetector()
    
#     # image_path = '/assets/example.jpg'  # or your own uploaded image
#     image_path = "D:/AIBuild/BodyShapeRecognition/assets/example.png"
#     try:
#         body_shape = detector.detect(image_path)
#         print(f"Predicted Body Shape: {body_shape}")
#     except ValueError as e:
#         print(e)

# if __name__ == "__main__":
#     main()
