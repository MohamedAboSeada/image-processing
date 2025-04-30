import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, simpledialog, messagebox
import customtkinter as ctk
import imutils
import os

class ImageProcessor:
    def __init__(self, ui):
        self.ui = ui
        self.cv_image = None
        self.original_image = None
        self.undo_stack = []
        self.history_list = []
        self.original_visible = True

    def open_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.cv_image = cv2.imread(path)
            self.original_image = self.cv_image.copy()
            self.undo_stack = []
            self.history_list.clear()
            self.ui.refresh_history(self.history_list)
            self.ui.update_display(self.cv_image, self.original_image)
            image_name = os.path.basename(path)
            self.ui.update_window_title(image_name)
            
    def save_image(self):
        if self.cv_image is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png", 
                                              filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg")])
            if path:
                cv2.imwrite(path, self.cv_image)
                
    def undo(self):
        if self.undo_stack:
            self.cv_image = self.undo_stack.pop()
            self.history_list.pop()
            self.ui.refresh_history(self.history_list)
            self.ui.update_display(self.cv_image, self.original_image)

    def reset_image(self):
        if self.original_image is not None:
            self.cv_image = self.original_image.copy()
            self.undo_stack.clear()
            self.history_list.clear()
            self.ui.refresh_history(self.history_list)
            self.ui.update_display(self.cv_image, self.original_image)

    def delete_image(self):
        if self.cv_image is not None:
            confirm = messagebox.askyesno("Confirm Delete", "Are you sure you want to delete the current image?")
            if confirm:
                self.cv_image = None
                self.original_image = None
                self.undo_stack.clear()
                self.history_list.clear()
                self.ui.refresh_history(self.history_list)
                self.ui.image_label.configure(image="", text="")
                self.ui.image_label.image = None
                self.ui.image_label_original.configure(image="", text="")
                self.ui.image_label_original.image = None

    def toggle_original_visibility(self):
        if self.original_visible:
            self.ui.image_label_original.pack_forget()
            self.original_visible = False
            self.ui.update_toggle_button(False)
        else:
            self.ui.image_label_original.pack(side="left", expand=True, fill="both", padx=5, pady=5)
            self.original_visible = True
            self.ui.update_toggle_button(True)

    def push_undo(self, operation=""):
        if self.cv_image is not None:
            self.undo_stack.append(self.cv_image.copy())
            self.history_list.append(operation)
            self.ui.refresh_history(self.history_list)

    def resize_image(self):
        values = simpledialog.askstring("Resize", "Enter width, height (comma separated):")
        w, h = map(int, values.split(","))
        if w and h:
            self.push_undo("Resize")
            self.cv_image = cv2.resize(self.cv_image, (w, h))
            self.ui.update_display(self.cv_image, self.original_image)

    def crop_image(self):
        values = simpledialog.askstring("Crop", "Enter x, y, width, height (comma separated):")
        if values:
            try:
                x, y, w, h = map(int, values.split(","))
                self.push_undo('Cro')
                self.cv_image = self.cv_image[y:y+h, x:x+w]
                self.update_display()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter four valid integers separated by commas.")


                


    def rotate_image_with_crop(self):
        angle = simpledialog.askfloat("Rotate with Crop", "Enter angle")
        if angle is not None:
            self.push_undo("Rotate with Crop")
            self.cv_image = imutils.rotate_bound(self.cv_image, angle)
            self.ui.update_display(self.cv_image, self.original_image)

    def rotate_image_without_crop(self):
        angle = simpledialog.askfloat("Rotate without Crop", "Enter angle")
        if angle is not None:
            self.push_undo("Rotate without Crop")
            self.cv_image = imutils.rotate(self.cv_image, angle)
            self.ui.update_display(self.cv_image, self.original_image)

    def flip_image(self, code):
        self.push_undo("Flip")
        self.cv_image = cv2.flip(self.cv_image, code)
        self.ui.update_display(self.cv_image, self.original_image)

    def adjust_brightness(self):
        factor = simpledialog.askfloat("Brightness", "Enter factor (e.g., 1.2)")
        if factor:
            self.push_undo("Adjust Brightness")
            hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
            self.cv_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.ui.update_display(self.cv_image, self.original_image)

    def convert_Rgb(self):
        self.push_undo("Convert to RGB")
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.ui.update_display(self.cv_image, self.original_image)

    def convert_Hsv(self):
        self.push_undo("Convert to HSV")
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        self.ui.update_display(self.cv_image, self.original_image)

    def convert_Gray(self):
        self.push_undo("Convert to Gray")
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        self.cv_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.ui.update_display(self.cv_image, self.original_image)

    def convert_Binary(self):
        self.push_undo("Convert to Binary")
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.cv_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.ui.update_display(self.cv_image, self.original_image)

    def split_channels(self):
        b, g, r = cv2.split(self.cv_image)
        cv2.imshow("Blue", b)
        cv2.imshow("Green", g)
        cv2.imshow("Red", r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def normalize_image(self):
        self.push_undo("Normalize")
        self.cv_image = cv2.normalize(self.cv_image, None, 0, 255, cv2.NORM_MINMAX)
        self.ui.update_display(self.cv_image, self.original_image)
        
    def show_histogram(self):
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([self.cv_image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.title("Histogram")
        plt.show()

    def equalize_histogram(self):
        self.push_undo("Equalize Histogram")
        yuv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        self.cv_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        self.ui.update_display(self.cv_image, self.original_image)

    def image_addition(self):
        path = filedialog.askopenfilename()
        if path:
            other = cv2.imread(path)
            other = cv2.resize(other, (self.cv_image.shape[1], self.cv_image.shape[0]))
            self.push_undo("Image Addition")
            self.cv_image = cv2.add(self.cv_image, other)
            self.ui.update_display(self.cv_image, self.original_image)

    def weighted_addition(self):
        path = filedialog.askopenfilename()
        if path:
            alpha = simpledialog.askfloat("Alpha", "Enter alpha")
            beta = 1 - alpha
            other = cv2.imread(path)
            other = cv2.resize(other, (self.cv_image.shape[1], self.cv_image.shape[0]))
            self.push_undo("Weighted Addition")
            self.cv_image = cv2.addWeighted(self.cv_image, alpha, other, beta, 0)
            self.ui.update_display(self.cv_image, self.original_image)

    def image_subtraction(self):
        path = filedialog.askopenfilename()
        if path:
            other = cv2.imread(path)
            other = cv2.resize(other, (self.cv_image.shape[1], self.cv_image.shape[0]))
            self.push_undo("Image Subtraction")
            self.cv_image = cv2.subtract(self.cv_image, other)
            self.ui.update_display(self.cv_image, self.original_image)

    def add_salt_pepper_noise(self, amount=0.02, salt_vs_pepper=0.5):
        self.push_undo("Add Salt & Pepper Noise")
        noisy = np.copy(self.cv_image)
        row, col, ch = noisy.shape
        num_salt = np.ceil(amount * noisy.size * salt_vs_pepper).astype(int)
        num_pepper = np.ceil(amount * noisy.size * (1.0 - salt_vs_pepper)).astype(int)

        # Salt (white) noise
        coords = [np.random.randint(0, i - 1, num_salt) for i in noisy.shape[:2]]
        noisy[coords[0], coords[1]] = [255, 255, 255]

        # Pepper (black) noise
        coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy.shape[:2]]
        noisy[coords[0], coords[1]] = [0, 0, 0]

        self.cv_image = noisy
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_averaging_filter(self):
        self.push_undo("Averaging Filter")
        kernel = np.ones((5, 5), np.float32) / 25
        self.cv_image = cv2.filter2D(self.cv_image, -1, kernel)
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_weighted_averaging_filter(self):
        self.push_undo("Weighted Averaging Filter")
        kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32)
        kernel /= np.sum(kernel)
        self.cv_image = cv2.filter2D(self.cv_image, -1, kernel)
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_gaussian_filter(self):
        self.push_undo("Gaussian Filter")
        self.cv_image = cv2.GaussianBlur(self.cv_image, (5, 5), 0)
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_median_filter(self):
        self.push_undo("Median Filter")
        self.cv_image = cv2.medianBlur(self.cv_image, 5)
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_max_filter(self):
        self.push_undo("Max Filter")
        self.cv_image = cv2.dilate(self.cv_image, np.ones((5,5), np.uint8))
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_min_filter(self):
        self.push_undo("Min Filter")
        self.cv_image = cv2.erode(self.cv_image, np.ones((5,5), np.uint8))
        self.ui.update_display(self.cv_image, self.original_image)

    def interactive_compare(self):
        if self.cv_image is None or self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        compare_window = ctk.CTkToplevel(self.ui.root)
        compare_window.title("Compare Original vs Edited")
        compare_window.geometry("800x600")

        canvas = ctk.CTkLabel(compare_window)
        canvas.pack(expand=True, fill="both")

        slider = ctk.CTkSlider(compare_window, from_=0, to=1, number_of_steps=100, orientation="horizontal")
        slider.set(0.5)
        slider.pack(pady=10)

        def update_image(value):
            alpha = float(value)
            beta = 1.0 - alpha

            img1 = cv2.resize(self.original_image.copy(), (600, 400))
            img2 = cv2.resize(self.cv_image.copy(), (600, 400))

            blended = cv2.addWeighted(img2, alpha, img1, beta, 0)
            blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            canvas.configure(image=blended)
            canvas.image = blended

        slider.configure(command=update_image)
        update_image(slider.get())

    def close_image(self):
        if self.cv_image is not None:
            confirm = messagebox.askyesno("Close Image", "Are you sure you want to close the current image?")
            if confirm:
                self.cv_image = None
                self.original_image = None
                self.undo_stack.clear()
                self.history_list.clear()
                self.ui.reset_interface() 