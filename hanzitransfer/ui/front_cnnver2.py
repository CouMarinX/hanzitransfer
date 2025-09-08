"""Tkinter-based user interface for Hanzi style transfer."""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk
import numpy as np
from io import BytesIO
import json

try:
    import win32clipboard  # type: ignore
except Exception:
    win32clipboard = None

from ..inference import load_cnn_model, render_strokes

# 运行时加载模型
model = None

# 将汉字转换为64x64的二维数组
def hanzi_to_array(hanzi):
    font = ImageFont.truetype('simsun.ttc', 64)
    image = Image.new('L', (64, 64), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), hanzi, font=font, fill=0)
    array = np.array(image)
    array = (array < 128).astype(int)
    return array

# 初始化总图片变量
total_image = None


# 处理输入的汉字
def process_hanzi():
    global total_image

    text = text_input.get('1.0', tk.END).strip()
    if not text:
        messagebox.showwarning("警告", "请输入汉字！")
        return
    if model is None:
        messagebox.showwarning("警告", "模型加载失败！")
        return

    hanzi_list = list(text)
    input_arrays = [hanzi_to_array(hanzi) for hanzi in hanzi_list]
    input_data = np.array(input_arrays)
    input_data = input_data.reshape(-1, 64, 64, 1)

    # 使用模型进行预测
    output_data = model.predict(input_data)

    # 将输出数组或笔画序列转为图片并拼接
    images = []
    for pred in output_data:
        img = None
        if isinstance(pred, (list, tuple)):
            img = render_strokes(pred)
        else:
            try:
                strokes = json.loads(pred)
                img = render_strokes(strokes)
            except Exception:
                arr = np.array(pred)
                if arr.ndim == 1:
                    arr = arr.reshape(64, 64)
                elif arr.ndim == 3:
                    arr = arr.squeeze(-1)
                img = Image.fromarray((arr * 255).astype(np.uint8))
        images.append(img)

    # 拼接图片
    width = 64 * len(images)
    height = 64
    total_image = Image.new('L', (width, height))
    for i, img in enumerate(images):
        total_image.paste(img, (i * 64, 0))

    # 显示图片
    photo = ImageTk.PhotoImage(total_image)
    image_label.config(image=photo, width=width, height=height)
    image_label.image = photo

    save_button.config(state='normal')
    copy_button.config(state='normal')

# 复制图片到剪贴板
def copy_to_clipboard():
    if total_image is None:
        messagebox.showwarning("警告", "没有图片可复制！")
        return
    if win32clipboard is None:
        messagebox.showwarning("警告", "当前环境不支持剪贴板复制")
        return

    output = BytesIO()
    total_image.save(output, format='BMP')
    data = output.getvalue()[14:]
    output.close()

    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    win32clipboard.CloseClipboard()

    messagebox.showinfo("提示", "图片已复制到剪贴板，可直接粘贴！")


# Ctrl + Enter 快捷键复制图片
def bind_shortcut(event=None):
    copy_to_clipboard()


def main() -> None:
    """Launch the Tkinter UI."""
    global model, text_input, image_label, save_button, copy_button, root

    try:
        model = load_cnn_model()
    except Exception as e:  # pragma: no cover - best effort loading
        print(f"Model loading failed: {e}")
        model = None

    # 创建主窗口
    root = tk.Tk()
    root.title("汉字转换器")
    root.geometry("600x200")  # 默认窗口大小

    # 主布局
    main_frame = tk.Frame(root, bg="#ffffff")
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # 左侧输入框区域
    text_frame = tk.Frame(main_frame, bg="#ffffff")
    text_frame.pack(side="left", fill="y", padx=10)

    # 圆角输入框背景
    text_canvas = tk.Canvas(text_frame, width=200, height=100, bg="#ffffff", highlightthickness=0)
    text_canvas.pack()

    # 绘制圆角矩形
    r = 20  # 圆角半径
    x1, y1, x2, y2 = 10, 10, 190, 90
    text_canvas.create_arc(x1, y1, x1 + r, y1 + r, start=90, extent=90, fill="#f5f5f5", outline="")
    text_canvas.create_arc(x2 - r, y1, x2, y1 + r, start=0, extent=90, fill="#f5f5f5", outline="")
    text_canvas.create_arc(x1, y2 - r, x1 + r, y2, start=180, extent=90, fill="#f5f5f5", outline="")
    text_canvas.create_arc(x2 - r, y2 - r, x2, y2, start=270, extent=90, fill="#f5f5f5", outline="")
    text_canvas.create_rectangle(x1 + r / 2, y1, x2 - r / 2, y2, fill="#f5f5f5", outline="")
    text_canvas.create_rectangle(x1, y1 + r / 2, x2, y2 - r / 2, fill="#f5f5f5", outline="")

    # 文本输入
    text_input = tk.Text(text_frame, width=20, height=4, relief="flat", bg="#f5f5f5")
    text_input.place(x=20, y=20)

    # 中间按钮区域
    button_frame = tk.Frame(main_frame, bg="#ffffff")
    button_frame.pack(side="left", padx=10, fill="y")

    process_button = tk.Button(button_frame, text="生成图片", command=process_hanzi)
    process_button.pack(pady=5)

    save_button = tk.Button(button_frame, text="保存图片", state="disabled")
    save_button.pack(pady=5)

    copy_button = tk.Button(button_frame, text="复制图片", state="disabled", command=copy_to_clipboard)
    copy_button.pack(pady=5)

    # 右侧图片显示区域
    image_frame = tk.Frame(main_frame, bg="#ffffff", width=200, height=100, relief="groove")
    image_frame.pack(side="left", padx=10)

    image_label = tk.Label(image_frame, bg="#ffffff", width=20, height=10, relief="flat")
    image_label.pack(fill="both", expand=True)

    # 绑定快捷键 Ctrl + Enter
    root.bind('<Control-Return>', bind_shortcut)

    root.mainloop()


if __name__ == "__main__":
    main()
