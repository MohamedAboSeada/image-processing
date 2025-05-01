import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, simpledialog, messagebox
import customtkinter as ctk
import imutils
import os
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, ui):
        self.ui = ui
        self.cv_image = None
        self.original_image = None
        self.undo_stack = []
        self.redo_stack = []  # Add redo stack
        self.history_list = []
        self.redo_history_list = []  # Add redo history list to track operation names
        self.original_visible = True

    def open_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.cv_image = cv2.imread(path)
            self.original_image = self.cv_image.copy()
            self.undo_stack = []
            self.redo_stack.clear()  # Clear redo stack on new image load
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
            self.redo_stack.append(self.cv_image.copy())  # Save current state to redo stack
            operation = self.history_list.pop()  # Get the operation name before removing it
            self.redo_history_list.append(operation)  # Save the operation name for redo
            self.cv_image = self.undo_stack.pop()
            self.ui.refresh_history(self.history_list)
            self.ui.update_display(self.cv_image, self.original_image)

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.cv_image.copy())  # Save current state to undo stack
            self.cv_image = self.redo_stack.pop()
            if self.redo_history_list:
                operation = self.redo_history_list.pop()  # Get the original operation name
                self.history_list.append(operation)  # Add the original operation name to history
            else:
                self.history_list.append("Redo")  # Fallback if no operation name is available
            self.ui.refresh_history(self.history_list)
            self.ui.update_display(self.cv_image, self.original_image)

    def reset_image(self):
        if self.original_image is not None:
            self.cv_image = self.original_image.copy()
            self.undo_stack.clear()
            self.redo_stack.clear()  # Clear redo stack on reset
            self.redo_history_list.clear()  # Clear redo history list on reset
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
                self.redo_stack.clear()  # Clear redo stack on delete
                self.redo_history_list.clear()  # Clear redo history list on delete
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
            self.redo_stack.clear()  # Clear redo stack on new operation
            self.redo_history_list.clear()  # Clear redo history list on new operation
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
                self.push_undo('Crop')
                self.cv_image = self.cv_image[y:y+h, x:x+w]
                self.ui.update_display(self.cv_image, self.original_image)  # Fixed method call
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter four valid integers separated by commas.")

    def rotate_image_window(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for rotation
        rotate_window = ctk.CTkToplevel(self.ui.root)
        rotate_window.title("Rotate Image")
        rotate_window.geometry("800x600")

        # Create a preview canvas
        canvas = ctk.CTkLabel(rotate_window, text="")
        canvas.pack(expand=True, fill="both")

        # Create controls frame
        controls_frame = ctk.CTkFrame(rotate_window)
        controls_frame.pack(pady=10)

        # Angle slider
        angle_frame = ctk.CTkFrame(controls_frame)
        angle_frame.pack(pady=5)

        angle_label = ctk.CTkLabel(angle_frame, text="Angle:")
        angle_label.pack(side="left", padx=5)

        angle_slider = ctk.CTkSlider(angle_frame, from_=-180, to=180, number_of_steps=360, orientation="horizontal", width=300)
        angle_slider.set(0)  # Default angle
        angle_slider.pack(side="left", padx=10)

        angle_value_label = ctk.CTkLabel(angle_frame, text="0°")
        angle_value_label.pack(side="left", padx=5)

        # Checkbox for crop/no crop
        crop_var = ctk.BooleanVar(value=True)
        crop_checkbox = ctk.CTkCheckBox(controls_frame, text="Maintain original dimensions (crop)", variable=crop_var)
        crop_checkbox.pack(pady=5)

        # Function to update the preview
        def update_preview(*args):
            angle = angle_slider.get()
            angle_value_label.configure(text=f"{int(angle)}°")

            preview_image = self.cv_image.copy()
            if crop_var.get():
                # Rotate with crop (maintain dimensions)
                preview_image = imutils.rotate_bound(preview_image, angle)
            else:
                # Rotate without crop (may show black corners)
                preview_image = imutils.rotate(preview_image, angle)

            preview_image = cv2.resize(preview_image, (600, 400))
            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
            preview_image = Image.fromarray(preview_image)
            preview_image = ImageTk.PhotoImage(preview_image)
            canvas.configure(image=preview_image)
            canvas.image = preview_image

        # Connect the slider and checkbox to the update function
        angle_slider.configure(command=lambda val: update_preview())
        crop_var.trace_add("write", update_preview)

        # Initial preview
        update_preview()

        # Confirm and Cancel buttons
        def confirm():
            angle = angle_slider.get()
            with_crop = crop_var.get()

            if with_crop:
                self.push_undo("Rotate with Crop")
                self.cv_image = imutils.rotate_bound(self.cv_image, angle)
            else:
                self.push_undo("Rotate without Crop")
                self.cv_image = imutils.rotate(self.cv_image, angle)

            self.ui.update_display(self.cv_image, self.original_image)
            rotate_window.destroy()

        def cancel():
            rotate_window.destroy()

        button_frame = ctk.CTkFrame(rotate_window)
        button_frame.pack(pady=10)

        confirm_button = ctk.CTkButton(button_frame, text="Confirm", command=confirm)
        confirm_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel)
        cancel_button.pack(side="right", padx=5)

    # Keep these methods for backward compatibility but make them use the new window
    def rotate_image_with_crop(self):
        self.rotate_image_window()

    def rotate_image_without_crop(self):
        self.rotate_image_window()

    def adjust_brightness_window(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for brightness control
        brightness_window = ctk.CTkToplevel(self.ui.root)
        brightness_window.title("Brightness Control")
        brightness_window.geometry("800x600")

        # Create a preview canvas
        canvas = ctk.CTkLabel(brightness_window, text="")  # Remove default "ctk label" text
        canvas.pack(expand=True, fill="both")

        # Create a slider for brightness adjustment
        slider = ctk.CTkSlider(brightness_window, from_=0.1, to=3.0, number_of_steps=290, orientation="horizontal")
        slider.set(1.0)  # Default brightness factor
        slider.pack(pady=10)

        # Function to update the preview
        def update_preview(value):
            factor = float(value)
            hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
            preview_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            preview_image = cv2.resize(preview_image, (600, 400))
            preview_image = Image.fromarray(preview_image)
            preview_image = ImageTk.PhotoImage(preview_image)
            canvas.configure(image=preview_image)
            canvas.image = preview_image

        slider.configure(command=update_preview)
        update_preview(slider.get())

        # Confirm and Cancel buttons
        def confirm():
            factor = float(slider.get())
            self.push_undo("Adjust Brightness")
            hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
            self.cv_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.ui.update_display(self.cv_image, self.original_image)
            brightness_window.destroy()

        def cancel():
            brightness_window.destroy()

        button_frame = ctk.CTkFrame(brightness_window)
        button_frame.pack(pady=10)

        confirm_button = ctk.CTkButton(button_frame, text="Confirm", command=confirm)
        confirm_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel)
        cancel_button.pack(side="right", padx=5)

    def color_conversion_window(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for color conversion
        color_window = ctk.CTkToplevel(self.ui.root)
        color_window.title("Color Conversion")
        color_window.geometry("900x700")

        # Create a title label
        title_label = ctk.CTkLabel(color_window, text="Click on an image to apply the conversion", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Create a frame for the image grid
        grid_frame = ctk.CTkFrame(color_window)
        grid_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Configure grid layout (2x2)
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)
        grid_frame.rowconfigure(0, weight=1)
        grid_frame.rowconfigure(1, weight=1)

        # Create frames for each color mode
        bgr_frame = ctk.CTkFrame(grid_frame)
        bgr_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        rgb_frame = ctk.CTkFrame(grid_frame)
        rgb_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        hsv_frame = ctk.CTkFrame(grid_frame)
        hsv_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        gray_frame = ctk.CTkFrame(grid_frame)
        gray_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # Create labels for each color mode
        bgr_label = ctk.CTkLabel(bgr_frame, text="Original (BGR)")
        bgr_label.pack(pady=5)

        rgb_label = ctk.CTkLabel(rgb_frame, text="RGB")
        rgb_label.pack(pady=5)

        hsv_label = ctk.CTkLabel(hsv_frame, text="HSV")
        hsv_label.pack(pady=5)

        gray_label = ctk.CTkLabel(gray_frame, text="Grayscale")
        gray_label.pack(pady=5)

        # Create image canvases
        bgr_canvas = ctk.CTkLabel(bgr_frame, text="")
        bgr_canvas.pack(expand=True, fill="both", padx=5, pady=5)

        rgb_canvas = ctk.CTkLabel(rgb_frame, text="")
        rgb_canvas.pack(expand=True, fill="both", padx=5, pady=5)

        hsv_canvas = ctk.CTkLabel(hsv_frame, text="")
        hsv_canvas.pack(expand=True, fill="both", padx=5, pady=5)

        gray_canvas = ctk.CTkLabel(gray_frame, text="")
        gray_canvas.pack(expand=True, fill="both", padx=5, pady=5)

        # Prepare the preview images
        # Original (BGR)
        bgr_preview = self.cv_image.copy()
        bgr_preview = cv2.resize(bgr_preview, (300, 200))
        # For display purposes, we need to convert BGR to RGB for PIL
        bgr_preview_rgb = cv2.cvtColor(bgr_preview, cv2.COLOR_BGR2RGB)
        bgr_preview_pil = Image.fromarray(bgr_preview_rgb)
        bgr_preview_tk = ImageTk.PhotoImage(bgr_preview_pil)
        bgr_canvas.configure(image=bgr_preview_tk)
        bgr_canvas.image = bgr_preview_tk

        # RGB
        # Create a true RGB image by swapping the channels
        rgb_preview = self.cv_image.copy()
        # Swap B and R channels to convert BGR to RGB
        rgb_preview = rgb_preview[:, :, ::-1]
        rgb_preview = cv2.resize(rgb_preview, (300, 200))
        # For display, we need RGB format which we already have
        rgb_preview_rgb = cv2.cvtColor(rgb_preview, cv2.COLOR_BGR2RGB)
        rgb_preview_pil = Image.fromarray(rgb_preview_rgb)
        rgb_preview_tk = ImageTk.PhotoImage(rgb_preview_pil)
        rgb_canvas.configure(image=rgb_preview_tk)
        rgb_canvas.image = rgb_preview_tk

        # HSV
        hsv_preview = cv2.cvtColor(self.cv_image.copy(), cv2.COLOR_BGR2HSV)
        # Enhance the HSV visualization to make it more distinct
        # Normalize H to 0-255 for better visualization
        h, s, v = cv2.split(hsv_preview)
        h_normalized = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX)
        s_normalized = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX)
        v_normalized = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)

        # Create a false-color HSV visualization
        hsv_display = cv2.merge([h_normalized, s_normalized, v_normalized])
        hsv_display = cv2.resize(hsv_display, (300, 200))
        # Convert to RGB for display
        hsv_display_rgb = cv2.cvtColor(hsv_display, cv2.COLOR_BGR2RGB)
        hsv_preview_pil = Image.fromarray(hsv_display_rgb)
        hsv_preview_tk = ImageTk.PhotoImage(hsv_preview_pil)
        hsv_canvas.configure(image=hsv_preview_tk)
        hsv_canvas.image = hsv_preview_tk

        # Grayscale
        gray_preview = cv2.cvtColor(self.cv_image.copy(), cv2.COLOR_BGR2GRAY)
        # Resize before converting back to RGB for display
        gray_preview = cv2.resize(gray_preview, (300, 200))
        # Convert to RGB for display (this will be a true grayscale image)
        gray_preview_display = cv2.cvtColor(gray_preview, cv2.COLOR_GRAY2RGB)
        gray_preview_pil = Image.fromarray(gray_preview_display)
        gray_preview_tk = ImageTk.PhotoImage(gray_preview_pil)
        gray_canvas.configure(image=gray_preview_tk)
        gray_canvas.image = gray_preview_tk

        # Define click handlers for each canvas
        def apply_bgr():
            # Original is already in BGR, so no conversion needed
            color_window.destroy()

        def apply_rgb():
            self.push_undo("Convert to RGB")
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            self.ui.update_display(self.cv_image, self.original_image)
            color_window.destroy()

        def apply_hsv():
            self.push_undo("Convert to HSV")
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            self.ui.update_display(self.cv_image, self.original_image)
            color_window.destroy()

        def apply_gray():
            self.push_undo("Convert to Grayscale")
            gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            self.cv_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.ui.update_display(self.cv_image, self.original_image)
            color_window.destroy()

        # Bind click events to canvases
        bgr_canvas.bind("<Button-1>", lambda e: apply_bgr())
        rgb_canvas.bind("<Button-1>", lambda e: apply_rgb())
        hsv_canvas.bind("<Button-1>", lambda e: apply_hsv())
        gray_canvas.bind("<Button-1>", lambda e: apply_gray())

        # Add cancel button at the bottom
        button_frame = ctk.CTkFrame(color_window)
        button_frame.pack(pady=10)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=color_window.destroy)
        cancel_button.pack(padx=5)

    # Keep these methods for backward compatibility but make them use the new window
    def convert_Rgb(self):
        self.color_conversion_window()

    def convert_Hsv(self):
        self.color_conversion_window()

    def convert_Gray(self):
        self.color_conversion_window()

    def split_channels(self):
        b, g, r = cv2.split(self.cv_image)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(b, cmap='gray')
        plt.title("Blue Channel")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(g, cmap='gray')
        plt.title("Green Channel")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(r, cmap='gray')
        plt.title("Red Channel")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

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

    def apply_speckle_noise(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        row, col, ch = self.cv_image.shape
        gauss = np.random.normal(0, 1, (row, col, ch)).reshape(row, col, ch)
        noisy = np.clip(self.cv_image + self.cv_image * gauss, 0, 255).astype(np.uint8)
        self.push_undo("Applied Speckle Noise")
        self.cv_image = noisy
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_poisson_noise(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        noisy = np.random.poisson(self.cv_image / 255.0 * 255).astype(np.uint8)
        self.push_undo("Applied Poisson Noise")
        self.cv_image = np.clip(noisy, 0, 255)
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_uniform_noise(self, low=0, high=50):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        row, col, ch = self.cv_image.shape
        noise = np.random.uniform(low, high, (row, col, ch)).reshape(row, col, ch)
        noisy = np.clip(self.cv_image + noise, 0, 255).astype(np.uint8)
        self.push_undo("Applied Uniform Noise")
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
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for median filter control
        median_window = ctk.CTkToplevel(self.ui.root)
        median_window.title("Median Filter Control")
        median_window.geometry("800x600")

        # Create a preview canvas
        canvas = ctk.CTkLabel(median_window, text="")
        canvas.pack(expand=True, fill="both")

        # Create a slider for kernel size adjustment
        kernel_frame = ctk.CTkFrame(median_window)
        kernel_frame.pack(pady=10)

        kernel_label = ctk.CTkLabel(kernel_frame, text="Kernel Size:")
        kernel_label.pack(side="left", padx=5)

        # Kernel size must be odd
        slider = ctk.CTkSlider(kernel_frame, from_=1, to=15, number_of_steps=7, orientation="horizontal")
        slider.set(5)  # Default kernel size
        slider.pack(side="left", padx=5)

        # Function to update the preview
        def update_preview(value):
            kernel_size = int(value) | 1  # Ensure kernel size is odd
            preview_image = self.cv_image.copy()
            preview_image = cv2.medianBlur(preview_image, kernel_size)
            preview_image = cv2.resize(preview_image, (600, 400))
            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
            preview_image = Image.fromarray(preview_image)
            preview_image = ImageTk.PhotoImage(preview_image)
            canvas.configure(image=preview_image)
            canvas.image = preview_image

        slider.configure(command=update_preview)
        update_preview(slider.get())  # Initial preview

        # Confirm and Cancel buttons
        def confirm():
            kernel_size = int(slider.get()) | 1  # Ensure kernel size is odd
            self.push_undo("Median Filter")
            self.cv_image = cv2.medianBlur(self.cv_image, kernel_size)
            self.ui.update_display(self.cv_image, self.original_image)
            median_window.destroy()

        def cancel():
            median_window.destroy()

        button_frame = ctk.CTkFrame(median_window)
        button_frame.pack(pady=10)

        confirm_button = ctk.CTkButton(button_frame, text="Confirm", command=confirm)
        confirm_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel)
        cancel_button.pack(side="right", padx=5)

    def apply_max_filter(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for max filter control
        max_window = ctk.CTkToplevel(self.ui.root)
        max_window.title("Max Filter Control")
        max_window.geometry("800x600")

        # Create a preview canvas
        canvas = ctk.CTkLabel(max_window, text="")
        canvas.pack(expand=True, fill="both")

        # Create a slider for kernel size adjustment
        kernel_frame = ctk.CTkFrame(max_window)
        kernel_frame.pack(pady=10)

        kernel_label = ctk.CTkLabel(kernel_frame, text="Kernel Size:")
        kernel_label.pack(side="left", padx=5)

        slider = ctk.CTkSlider(kernel_frame, from_=1, to=15, number_of_steps=7, orientation="horizontal")
        slider.set(5)  # Default kernel size
        slider.pack(side="left", padx=5)

        # Function to update the preview
        def update_preview(value):
            kernel_size = int(value) | 1  # Ensure kernel size is odd
            preview_image = self.cv_image.copy()
            preview_image = cv2.dilate(preview_image, np.ones((kernel_size, kernel_size), np.uint8))
            preview_image = cv2.resize(preview_image, (600, 400))
            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
            preview_image = Image.fromarray(preview_image)
            preview_image = ImageTk.PhotoImage(preview_image)
            canvas.configure(image=preview_image)
            canvas.image = preview_image

        slider.configure(command=update_preview)
        update_preview(slider.get())  # Initial preview

        # Confirm and Cancel buttons
        def confirm():
            kernel_size = int(slider.get()) | 1  # Ensure kernel size is odd
            self.push_undo("Max Filter")
            self.cv_image = cv2.dilate(self.cv_image, np.ones((kernel_size, kernel_size), np.uint8))
            self.ui.update_display(self.cv_image, self.original_image)
            max_window.destroy()

        def cancel():
            max_window.destroy()

        button_frame = ctk.CTkFrame(max_window)
        button_frame.pack(pady=10)

        confirm_button = ctk.CTkButton(button_frame, text="Confirm", command=confirm)
        confirm_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel)
        cancel_button.pack(side="right", padx=5)

    def apply_min_filter(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for min filter control
        min_window = ctk.CTkToplevel(self.ui.root)
        min_window.title("Min Filter Control")
        min_window.geometry("800x600")

        # Create a preview canvas
        canvas = ctk.CTkLabel(min_window, text="")
        canvas.pack(expand=True, fill="both")

        # Create a slider for kernel size adjustment
        kernel_frame = ctk.CTkFrame(min_window)
        kernel_frame.pack(pady=10)

        kernel_label = ctk.CTkLabel(kernel_frame, text="Kernel Size:")
        kernel_label.pack(side="left", padx=5)

        slider = ctk.CTkSlider(kernel_frame, from_=1, to=15, number_of_steps=7, orientation="horizontal")
        slider.set(5)  # Default kernel size
        slider.pack(side="left", padx=5)

        # Function to update the preview
        def update_preview(value):
            kernel_size = int(value) | 1  # Ensure kernel size is odd
            preview_image = self.cv_image.copy()
            preview_image = cv2.erode(preview_image, np.ones((kernel_size, kernel_size), np.uint8))
            preview_image = cv2.resize(preview_image, (600, 400))
            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
            preview_image = Image.fromarray(preview_image)
            preview_image = ImageTk.PhotoImage(preview_image)
            canvas.configure(image=preview_image)
            canvas.image = preview_image

        slider.configure(command=update_preview)
        update_preview(slider.get())  # Initial preview

        # Confirm and Cancel buttons
        def confirm():
            kernel_size = int(slider.get()) | 1  # Ensure kernel size is odd
            self.push_undo("Min Filter")
            self.cv_image = cv2.erode(self.cv_image, np.ones((kernel_size, kernel_size), np.uint8))
            self.ui.update_display(self.cv_image, self.original_image)
            min_window.destroy()

        def cancel():
            min_window.destroy()

        button_frame = ctk.CTkFrame(min_window)
        button_frame.pack(pady=10)

        confirm_button = ctk.CTkButton(button_frame, text="Confirm", command=confirm)
        confirm_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel)
        cancel_button.pack(side="right", padx=5)

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

    def adjust_threshold(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for threshold control
        threshold_window = ctk.CTkToplevel(self.ui.root)
        threshold_window.title("Threshold Control")
        threshold_window.geometry("800x600")

        # Create a preview canvas
        canvas = ctk.CTkLabel(threshold_window, text="")  # Remove default "ctk label" text
        canvas.pack(expand=True, fill="both")

        # Create a slider for threshold adjustment
        slider = ctk.CTkSlider(threshold_window, from_=0, to=255, number_of_steps=255, orientation="horizontal")
        slider.set(127)  # Default threshold value
        slider.pack(pady=10)

        # Function to update the preview
        def update_preview(value):
            threshold_value = int(float(value))
            gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            preview_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            preview_image = cv2.resize(preview_image, (600, 400))
            preview_image = Image.fromarray(preview_image)
            preview_image = ImageTk.PhotoImage(preview_image)
            canvas.configure(image=preview_image)
            canvas.image = preview_image

        slider.configure(command=update_preview)
        update_preview(slider.get())

        # Confirm and Cancel buttons
        def confirm():
            threshold_value = int(slider.get())
            self.push_undo("Adjust Threshold")
            gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            self.cv_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            self.ui.update_display(self.cv_image, self.original_image)
            threshold_window.destroy()

        def cancel():
            threshold_window.destroy()

        button_frame = ctk.CTkFrame(threshold_window)
        button_frame.pack(pady=10)

        confirm_button = ctk.CTkButton(button_frame, text="Confirm", command=confirm)
        confirm_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel)
        cancel_button.pack(side="right", padx=5)

    def adjust_flip(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for flip adjustment
        flip_window = ctk.CTkToplevel(self.ui.root)
        flip_window.title("Flip Adjustment")
        flip_window.geometry("800x600")

        # Create a preview canvas
        canvas = ctk.CTkLabel(flip_window, text="")  # Remove default "ctk label" text
        canvas.pack(expand=True, fill="both")

        # Create radio buttons for flip options
        flip_option = ctk.StringVar(value="horizontal")

        def update_preview(*args):
            option = flip_option.get()
            preview_image = self.cv_image.copy()
            if option == "horizontal":
                preview_image = cv2.flip(preview_image, 1)
            elif option == "vertical":
                preview_image = cv2.flip(preview_image, 0)
            elif option == "both":
                preview_image = cv2.flip(preview_image, -1)

            preview_image = cv2.resize(preview_image, (600, 400))
            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
            preview_image = Image.fromarray(preview_image)
            preview_image = ImageTk.PhotoImage(preview_image)
            canvas.configure(image=preview_image)
            canvas.image = preview_image

        # Trace the flip_option variable to call update_preview on change
        flip_option.trace_add("write", update_preview)

        horizontal_button = ctk.CTkRadioButton(flip_window, text="Flip Horizontally", variable=flip_option, value="horizontal")
        horizontal_button.pack(anchor="w", padx=10, pady=5)

        vertical_button = ctk.CTkRadioButton(flip_window, text="Flip Vertically", variable=flip_option, value="vertical")
        vertical_button.pack(anchor="w", padx=10, pady=5)

        both_button = ctk.CTkRadioButton(flip_window, text="Flip Both", variable=flip_option, value="both")
        both_button.pack(anchor="w", padx=10, pady=5)

        # Update preview immediately for the default option
        update_preview()

        # Confirm and Cancel buttons
        def confirm():
            option = flip_option.get()
            self.push_undo("Flip")
            if option == "horizontal":
                self.cv_image = cv2.flip(self.cv_image, 1)
            elif option == "vertical":
                self.cv_image = cv2.flip(self.cv_image, 0)
            elif option == "both":
                self.cv_image = cv2.flip(self.cv_image, -1)
            self.ui.update_display(self.cv_image, self.original_image)
            flip_window.destroy()

        def cancel():
            flip_window.destroy()

        button_frame = ctk.CTkFrame(flip_window)
        button_frame.pack(pady=10)

        confirm_button = ctk.CTkButton(button_frame, text="Confirm", command=confirm)
        confirm_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel)
        cancel_button.pack(side="right", padx=5)

    def double_exposure_window(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for double exposure
        double_exposure_window = ctk.CTkToplevel(self.ui.root)
        double_exposure_window.title("Double Exposure Effect")
        double_exposure_window.geometry("800x600")

        # Create a preview canvas
        canvas = ctk.CTkLabel(double_exposure_window, text="")
        canvas.pack(expand=True, fill="both")

        # Variable to store the second image
        self.second_image = None

        # Function to upload second image
        def upload_second_image():
            path = filedialog.askopenfilename()
            if path:
                second_img = cv2.imread(path)
                if second_img is not None:
                    # Resize second image to match dimensions of the first image
                    self.second_image = cv2.resize(second_img, (self.cv_image.shape[1], self.cv_image.shape[0]))
                    # Show mode selection after image is uploaded
                    modes_frame.pack(pady=10)
                    # Update preview with default mode
                    update_preview()
                else:
                    messagebox.showerror("Error", "Could not load the image.")

        # Upload button
        upload_button = ctk.CTkButton(double_exposure_window, text="Upload Second Image", command=upload_second_image)
        upload_button.pack(pady=20)

        # Mode selection frame (initially hidden)
        modes_frame = ctk.CTkFrame(double_exposure_window)

        # Mode variable
        mode_var = ctk.StringVar(value="add")

        # Alpha values frame for weighted mode (initially hidden)
        alpha_frame = ctk.CTkFrame(double_exposure_window)

        # Alpha entry fields
        alpha_label1 = ctk.CTkLabel(alpha_frame, text="Alpha for First Image:")
        alpha_label1.pack(side="left", padx=5)
        alpha_entry1 = ctk.CTkEntry(alpha_frame, width=60)
        alpha_entry1.insert(0, "0.5")
        alpha_entry1.pack(side="left", padx=5)

        alpha_label2 = ctk.CTkLabel(alpha_frame, text="Alpha for Second Image:")
        alpha_label2.pack(side="left", padx=5)
        alpha_entry2 = ctk.CTkEntry(alpha_frame, width=60)
        alpha_entry2.insert(0, "0.5")
        alpha_entry2.pack(side="left", padx=5)

        # Apply button for alpha values
        apply_alpha_button = ctk.CTkButton(alpha_frame, text="Apply", width=60, command=lambda: update_preview())
        apply_alpha_button.pack(side="left", padx=10)

        # Function to update preview based on selected mode
        def update_preview():
            if self.second_image is None:
                return

            mode = mode_var.get()
            if mode == "add":
                preview_image = cv2.add(self.cv_image, self.second_image)
            elif mode == "weighted":
                try:
                    alpha1 = float(alpha_entry1.get())
                    alpha2 = float(alpha_entry2.get())
                    preview_image = cv2.addWeighted(self.cv_image, alpha1, self.second_image, alpha2, 0)
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter valid alpha values.")
                    return
            elif mode == "subtract":
                preview_image = cv2.subtract(self.cv_image, self.second_image)

            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
            preview_image = cv2.resize(preview_image, (600, 400))
            preview_image = Image.fromarray(preview_image)
            preview_image = ImageTk.PhotoImage(preview_image)
            canvas.configure(image=preview_image)
            canvas.image = preview_image

        def on_mode_change():
            if mode_var.get() == "weighted":
                alpha_frame.pack(pady=10)
                double_exposure_window.geometry("900x650")  # Increase window size
            else:
                alpha_frame.pack_forget()
                double_exposure_window.geometry("800x600")  # Reset window size
            update_preview()

        modes = [
            ("Add Two Images", "add"),
            ("Add Weighted", "weighted"),
            ("Subtract", "subtract")
        ]

        for text, value in modes:
            radio = ctk.CTkRadioButton(modes_frame, text=text, variable=mode_var, value=value, command=on_mode_change)
            radio.pack(anchor="w", padx=10, pady=5)

        # Confirm and Cancel buttons
        def confirm():
            if self.second_image is None:
                messagebox.showwarning("No Second Image", "Please upload a second image first.")
                return

            mode = mode_var.get()
            self.push_undo("Double Exposure")

            if mode == "add":
                self.cv_image = cv2.add(self.cv_image, self.second_image)
            elif mode == "weighted":
                try:
                    alpha1 = float(alpha_entry1.get())
                    alpha2 = float(alpha_entry2.get())
                    self.cv_image = cv2.addWeighted(self.cv_image, alpha1, self.second_image, alpha2, 0)
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter valid alpha values.")
                    return
            elif mode == "subtract":
                self.cv_image = cv2.subtract(self.cv_image, self.second_image)

            self.ui.update_display(self.cv_image, self.original_image)
            double_exposure_window.destroy()

        def cancel():
            double_exposure_window.destroy()

        button_frame = ctk.CTkFrame(double_exposure_window)
        button_frame.pack(side="bottom", pady=10)

        confirm_button = ctk.CTkButton(button_frame, text="Confirm", command=confirm)
        confirm_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel)
        cancel_button.pack(side="right", padx=5)

    def close_image(self):
        if self.cv_image is not None:
            confirm = messagebox.askyesno("Close Image", "Are you sure you want to close the current image?")
            if confirm:
                self.cv_image = None
                self.original_image = None
                self.undo_stack.clear()
                self.redo_stack.clear()  # Clear redo stack on close
                self.history_list.clear()
                self.ui.reset_interface()