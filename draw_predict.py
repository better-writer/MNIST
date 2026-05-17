import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
import subprocess
import os
import sys
from datetime import datetime

CANVAS_SIZE = 280
IMG_SIZE = 28
MODEL_PATH = "models/mnist_cnn.onnx"
INFERENCE_BIN = "./inference"
TEMP_RAW = "/tmp/predict_input.raw"
EXAMPLE_DIR = "example"


class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit - Draw & Predict")
        self.root.resizable(False, False)

        # Canvas (white background)
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="white", cursor="cross")
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Predict", command=self.predict).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side="left", padx=5)

        # Result label
        self.result_label = tk.Label(root, text="", font=("Arial", 18))
        self.result_label.pack(pady=5)

        self.last_x = None
        self.last_y = None
        self.drawing = False

    def reset_last(self, event=None):
        self.last_x = None
        self.last_y = None

    def draw(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    fill="black", width=12, capstyle="round", smooth=True)
        self.last_x = x
        self.last_y = y
        self.drawing = True

    def clear(self):
        self.canvas.delete("all")
        self.result_label.config(text="")
        self.last_x = None
        self.last_y = None
        self.drawing = False

    def predict(self):
        if not self.drawing:
            messagebox.showwarning("Warning", "Please draw a digit first!")
            return

        # Capture canvas as image using postscript (works on WSL2 / headless X)
        self.canvas.update()
        ps_path = "/tmp/canvas_capture.ps"
        self.canvas.postscript(file=ps_path, colormode="color")
        img = Image.open(ps_path)
        # Convert to grayscale, resize to 28x28
        img = img.convert("L")
        img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        # Save PNG to example directory
        os.makedirs(EXAMPLE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = os.path.join(EXAMPLE_DIR, f"digit_{timestamp}.png")
        img_resized.save(saved_path)

        # Normalize to float32: pixel 0-255 -> 0.0-1.0, then invert (MNIST is dark bg, light stroke)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = 1.0 - arr
        # Standardize with MNIST stats
        arr = (arr - 0.1307) / 0.3081

        # Save as raw float32 binary
        arr.tofile(TEMP_RAW)

        # Run C inference
        if not os.path.exists(INFERENCE_BIN):
            messagebox.showerror("Error", f"Inference binary not found: {INFERENCE_BIN}\nRun 'make' first.")
            return
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", f"Model not found: {MODEL_PATH}\nRun 'python train.py' first.")
            return

        try:
            result = subprocess.run(
                [INFERENCE_BIN, MODEL_PATH, TEMP_RAW],
                capture_output=True, text=True, check=True
            )
            # Parse output
            for line in result.stdout.splitlines():
                if "Predicted" in line:
                    digit = line.split(":")[1].strip()
                    self.result_label.config(text=f"Predicted: {digit}")
                    print(result.stdout)
                    print(f"Image saved to {saved_path}")
                    break
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Inference failed:\n{e.stderr}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run 'python train.py' first.")
        sys.exit(1)
    if not os.path.exists(INFERENCE_BIN):
        print(f"Inference binary not found at {INFERENCE_BIN}. Run 'make' first.")
        sys.exit(1)

    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
