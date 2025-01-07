import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.optimizers import Adam
import ast

# 文件路径
input_file = "output/base_character1.txt"
output_file = "output/new_character1.txt"

# 检查文件是否存在
if not os.path.exists(input_file) or not os.path.exists(output_file):
    raise FileNotFoundError("输入或输出文件未找到，请确认路径是否正确！")

# 加载数据
def load_data(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        data_list = ast.literal_eval(content)
        data = np.array(data_list, dtype=np.float32)
    return data

input_data = load_data(input_file)
output_data = load_data(output_file)

# 检查数据维度
if input_data.shape[1:] != (64, 64) or output_data.shape[1:] != (64, 64):
    raise ValueError("输入或输出数组的维度不正确，应为N x 64 x 64！")
if input_data.shape[0] != output_data.shape[0]:
    raise ValueError("输入和输出数据的数量不一致！")

# 数据预处理
x_train = input_data.reshape((input_data.shape[0], 64, 64, 1))  # 添加通道维度
y_train = output_data.reshape((output_data.shape[0], 64, 64, 1))

# 构建CNN模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(64 * 64, activation='sigmoid'),
    Reshape((64, 64, 1))
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

# 保存模型
model.save("hanzi_style_model.keras")

print("模型训练完成并已保存为 hanzi_style_model.keras")