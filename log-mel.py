import librosa
import numpy as np
import cv2
import os

input_folder = r"D:\code\python\ADReSS-IS2020-data\test\Normalised_audio-chunks"  # Thư mục chứa file .wav
output_folder = "output_images_rgb"  # Thư mục lưu ảnh RGB
os.makedirs(output_folder, exist_ok=True)

def process_audio(file_path):
    # Tải file âm thanh
    y, sr = librosa.load(file_path, sr=None)
    # Trích xuất log mel spectrogram
    n_mels = 224
    hop_length = 1024
    win_length = 2048 
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length, window='hann')
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Tính delta và delta-delta
    delta = librosa.feature.delta(log_mel)
    delta_delta = librosa.feature.delta(log_mel, order=2)
    
    # Kết hợp các kênh thành ảnh RGB
    img_rgb = np.stack([log_mel_norm, delta_norm, delta_delta_norm], axis=-1)

    # Chuyển đổi giá trị sang [0, 255] và kiểu uint8
    img_rgb = (255 * img_rgb).astype(np.uint8)

    return img_rgb

# Duyệt qua các file trong thư mục đầu vào
for file_name in os.listdir(input_folder):
    if file_name.endswith(".wav"):
        file_path = os.path.join(input_folder, file_name)

        # Xử lý file âm thanh và tạo ảnh RGB
        img_rgb = process_audio(file_path)

        # Thay đổi kích thước ảnh về 224x224
        img_resized = cv2.resize(img_rgb, (224, 224))

        # Lưu ảnh RGB
        output_path = os.path.join(output_folder, file_name.replace(".wav", ".png"))
        cv2.imwrite(output_path, img_resized)

        print(f"✔ Đã xử lý: {file_name} → {output_path}")

print("\n🎉 Xong! Tất cả các file .wav đã được chuyển thành ảnh RGB.")
