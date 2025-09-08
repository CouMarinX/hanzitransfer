"""Tkinter interface for Hanzi style transfer."""

from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk

from ..inference.predict import load_model, predict_chars


def main(model_path: str) -> None:
    """Launch the Tkinter GUI using the model at ``model_path``."""
    model = load_model(model_path)

    def process() -> None:
        text = text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("警告", "请输入汉字！")
            return
        try:
            img = predict_chars(model, text)
        except Exception as e:  # pragma: no cover - UI error display
            messagebox.showerror("错误", str(e))
            return
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo, width=img.width, height=img.height)
        image_label.image = photo

    root = tk.Tk()
    root.title("汉字转换器")

    text_input = tk.Text(root, width=20, height=4)
    text_input.pack(side="left", padx=10, pady=10)

    button = tk.Button(root, text="生成图片", command=process)
    button.pack(side="left", padx=10, pady=10)

    image_label = tk.Label(root)
    image_label.pack(side="left", padx=10, pady=10)

    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hanzi Transfer GUI")
    parser.add_argument("--model", default="hanzi_style_model.keras", help="Model file path")
    args = parser.parse_args()
    main(args.model)
