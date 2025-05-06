# import tensorflow as tf
# from app.data_preprocessing import prepare_data
# from app.body_shape_detector import BodyShapeDetector

# def train_model(train_dir, test_dir):
#     # Chuẩn bị dữ liệu huấn luyện và kiểm tra
#     train_generator, validation_generator = prepare_data(train_dir, test_dir)

#     # Xây dựng mô hình
#     base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#     base_model.trainable = False

#     model = tf.keras.Sequential([
#         base_model,
#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')  # Số lớp đầu ra = số kiểu dáng cơ thể
#     ])

#     model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

#     # Huấn luyện mô hình
#     model.fit(
#         train_generator,
#         steps_per_epoch=train_generator.samples // train_generator.batch_size,
#         epochs=10,
#         validation_data=validation_generator,
#         validation_steps=validation_generator.samples // validation_generator.batch_size
#     )

#     # Lưu mô hình đã huấn luyện
#     model.save('models/body_shape_model.h5')

# def main():
#     # Đường dẫn đến dữ liệu huấn luyện và kiểm tra
#     train_dir = 'data/train'
#     test_dir = 'data/test'

#     # Huấn luyện mô hình
#     train_model(train_dir, test_dir)

#     # Sử dụng mô hình đã huấn luyện để nhận diện kiểu dáng cơ thể
#     detector = BodyShapeDetector('models/body_shape_model.h5')
#     image_path = 'data/test/sample_image.jpg'  # Hình ảnh cần dự đoánw
#     body_shape = detector.detect(image_path)
#     print(f'Predicted Body Shape: {body_shape}')

# if __name__ == '__main__':
#     main()


# # from app.body_shape_detector import BodyShapeDetector

# # def main():
# #     detector = BodyShapeDetector()
    
# #     # image_path = '/assets/example.jpg'  # or your own uploaded image
# #     image_path = "D:/AIBuild/BodyShapeRecognition/assets/example.png"
# #     try:
# #         body_shape = detector.detect(image_path)
# #         print(f"Predicted Body Shape: {body_shape}")
# #     except ValueError as e:
# #         print(e)

# # if __name__ == "__main__":
# #     main()


## ----------- CHAY TRONG MÔI TRƯỜNG FASTAPI -------------- ##

import tensorflow as tf
import json
from tensorflow.keras.callbacks import EarlyStopping
from app.data_preprocessing import prepare_data
from app.body_shape_detector import BodyShapeDetector
from app.utils import download_image_from_url
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Khởi tạo FastAPI
app = FastAPI()

# Cho phép gọi API từ frontend (nếu có)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model một lần khi start server
model_path = "models/body_shape_model.h5"
detector = BodyShapeDetector(model_path)

def train_model(train_dir, test_dir, model_save_path='models/body_shape_model.h5'):
    train_generator, validation_generator = prepare_data(train_dir, test_dir)

    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True

    # Fine-tune một phần cuối của base model
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[early_stop]
    )

    # Lưu mô hình
    model.save(model_save_path)

    # Lưu class indices để dự đoán sau này
    class_indices = list(train_generator.class_indices.keys())
    with open(model_save_path.replace('.h5', '_classes.json'), 'w') as f:
        json.dump(class_indices, f)


# Schema để nhận đầu vào
class ImageURLRequest(BaseModel):
    image_url: str

@app.post("/predict")
async def predict_body_shape(request: ImageURLRequest):
    try:
        image_path = download_image_from_url(request.image_url)
        result = detector.detect(image_path)
        return {"predicted_body_shape": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Tuỳ chọn chạy trực tiếp file này bằng python app/main.py
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    
# def main():
#     train_dir = 'data/train'
#     test_dir = 'data/test'

#     model_path = 'models/body_shape_model.h5'

#     # Huấn luyện
#     # train_model(train_dir, test_dir, model_path)

#     # Dự đoán
#     detector = BodyShapeDetector(model_path)
#     # Nhập link ảnh
#     while True:
#         image_url = input("Insert your body image URL: ").strip()
#         if image_url.lower() in ['exit', 'quit']:
#             print("Exiting...")
#             break

#         #image_path = 'data/test/sample_image.jpg'  # Thay ảnh mẫu
#         try:
#             image_path = download_image_from_url(image_url)
#             result = detector.detect(image_path)
#             print(f'Predicted Body Shape: {result}')
#         except Exception as e:
#             print(f"Error: {e}")

# if __name__ == '__main__':
#     main()
