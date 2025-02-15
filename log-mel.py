import librosa
import numpy as np
import cv2
import os
from tqdm import tqdm  # Hiển thị tiến trình

# Định nghĩa thư mục đầu vào
input_train_folder_CC = r"D:\code\python\ADReSS-IS2020-train\ADReSS-IS2020-data\train\Full_wave_enhanced_audio\cc"   # input for training cc 
input_train_folder_CD = r"D:\code\python\ADReSS-IS2020-train\ADReSS-IS2020-data\train\Full_wave_enhanced_audio\cd"   # input for training cd
input_test_folder = r"D:\code\python\ADReSS-IS2020-test\ADReSS-IS2020-data\test\Full_wave_enhanced_audio" #input for test

# Định nghĩa thư mục đầu ra
output_train_folder_CC = "output_images_rgb/train_CC"  # tên thư mực output training
output_train_folder_CD = "output_images_rgb/train_CD"  # tên thư mực output training
output_test_folder = "output_images_rgb/test" # tên thư mực output test

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(output_train_folder_CC, exist_ok=True)
os.makedirs(output_train_folder_CD, exist_ok=True)
os.makedirs(output_test_folder, exist_ok=True)

def process_audio(file_path):
    """Xử lý file âm thanh và trả về ảnh RGB"""
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

    # Chuyển giá trị về khoảng [0, 255]
    log_mel = np.interp(log_mel, (log_mel.min(), log_mel.max()), (0, 255))
    delta = np.interp(delta, (delta.min(), delta.max()), (0, 255))
    delta_delta = np.interp(delta_delta, (delta_delta.min(), delta_delta.max()), (0, 255))

    # Kết hợp các kênh thành ảnh RGB
    img_rgb = np.stack([log_mel, delta, delta_delta], axis=-1).astype(np.uint8)

    return img_rgb

def process_folder(input_folder, output_folder):
    """Duyệt qua thư mục đầu vào và xử lý các file .wav"""
    wav_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

    for file_name in tqdm(wav_files, desc=f"Đang xử lý {os.path.basename(input_folder)}", unit="file"):
        file_path = os.path.join(input_folder, file_name)

        # Xử lý file âm thanh và tạo ảnh RGB
        img_rgb = process_audio(file_path)

        # Thay đổi kích thước ảnh về 224x224
        img_resized = cv2.resize(img_rgb, (224, 224))

        # Lưu ảnh RGB
        output_path = os.path.join(output_folder, file_name.replace(".wav", ".png"))
        cv2.imwrite(output_path, img_resized)

# Xử lý cả hai thư mục train và test
process_folder(input_train_folder_CC, output_train_folder_CC)
process_folder(input_train_folder_CD, output_train_folder_CD)
process_folder(input_test_folder, output_test_folder)

print("\n🎉 Xong! Tất cả các file .wav đã được chuyển thành ảnh RGB.")
