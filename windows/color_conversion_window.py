import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
from .base_window import BaseWindow

class ColorConversionWindow(BaseWindow):
    """Window for color conversion operations"""
    
    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Color Conversion", geometry="900x700")
        
        # Create a title label
        self.title_label = ctk.CTkLabel(self.window, 
                                      text="Click on an image to apply the conversion", 
                                      font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)
        
        # Create a frame for the image grid
        self.grid_frame = ctk.CTkFrame(self.window)
        self.grid_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Configure grid layout (2x2)
        self.grid_frame.columnconfigure(0, weight=1)
        self.grid_frame.columnconfigure(1, weight=1)
        self.grid_frame.rowconfigure(0, weight=1)
        self.grid_frame.rowconfigure(1, weight=1)
        
        # Create frames for each color mode
        self.bgr_frame = ctk.CTkFrame(self.grid_frame)
        self.bgr_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.rgb_frame = ctk.CTkFrame(self.grid_frame)
        self.rgb_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.hsv_frame = ctk.CTkFrame(self.grid_frame)
        self.hsv_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        self.gray_frame = ctk.CTkFrame(self.grid_frame)
        self.gray_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        # Create labels for each color mode
        self.bgr_label = ctk.CTkLabel(self.bgr_frame, text="Original (BGR)")
        self.bgr_label.pack(pady=5)
        
        self.rgb_label = ctk.CTkLabel(self.rgb_frame, text="RGB")
        self.rgb_label.pack(pady=5)
        
        self.hsv_label = ctk.CTkLabel(self.hsv_frame, text="HSV")
        self.hsv_label.pack(pady=5)
        
        self.gray_label = ctk.CTkLabel(self.gray_frame, text="Grayscale")
        self.gray_label.pack(pady=5)
        
        # Create image canvases
        self.bgr_canvas = ctk.CTkLabel(self.bgr_frame, text="")
        self.bgr_canvas.pack(expand=True, fill="both", padx=5, pady=5)
        
        self.rgb_canvas = ctk.CTkLabel(self.rgb_frame, text="")
        self.rgb_canvas.pack(expand=True, fill="both", padx=5, pady=5)
        
        self.hsv_canvas = ctk.CTkLabel(self.hsv_frame, text="")
        self.hsv_canvas.pack(expand=True, fill="both", padx=5, pady=5)
        
        self.gray_canvas = ctk.CTkLabel(self.gray_frame, text="")
        self.gray_canvas.pack(expand=True, fill="both", padx=5, pady=5)
        
        # Store original image
        self.original_image = self.image_processor.cv_image.copy()
        
        # Prepare and display the preview images
        self.prepare_previews()
        
        # Bind click events to canvases
        self.bgr_canvas.bind("<Button-1>", lambda e: self.apply_bgr())
        self.rgb_canvas.bind("<Button-1>", lambda e: self.apply_rgb())
        self.hsv_canvas.bind("<Button-1>", lambda e: self.apply_hsv())
        self.gray_canvas.bind("<Button-1>", lambda e: self.apply_gray())
        
        # Add cancel button at the bottom
        self.button_frame = ctk.CTkFrame(self.window)
        self.button_frame.pack(pady=10)
        
        self.cancel_button = ctk.CTkButton(self.button_frame, text="Cancel", 
                                         command=self.window.destroy)
        self.cancel_button.pack(padx=5)
        
    def prepare_previews(self):
        """Prepare and display preview images for all color modes"""
        # Original (BGR)
        bgr_preview = self.original_image.copy()
        bgr_preview = cv2.resize(bgr_preview, (300, 200))
        # For display purposes, we need to convert BGR to RGB for PIL
        bgr_preview_rgb = cv2.cvtColor(bgr_preview, cv2.COLOR_BGR2RGB)
        bgr_preview_pil = Image.fromarray(bgr_preview_rgb)
        bgr_preview_tk = ImageTk.PhotoImage(bgr_preview_pil)
        self.bgr_canvas.configure(image=bgr_preview_tk)
        self.bgr_canvas.image = bgr_preview_tk
        
        # RGB
        # Create a true RGB image by swapping the channels
        rgb_preview = self.original_image.copy()
        # Swap B and R channels to convert BGR to RGB
        rgb_preview = rgb_preview[:, :, ::-1]
        rgb_preview = cv2.resize(rgb_preview, (300, 200))
        # For display, we need RGB format which we already have
        rgb_preview_rgb = cv2.cvtColor(rgb_preview, cv2.COLOR_BGR2RGB)
        rgb_preview_pil = Image.fromarray(rgb_preview_rgb)
        rgb_preview_tk = ImageTk.PhotoImage(rgb_preview_pil)
        self.rgb_canvas.configure(image=rgb_preview_tk)
        self.rgb_canvas.image = rgb_preview_tk
        
        # HSV
        hsv_preview = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2HSV)
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
        self.hsv_canvas.configure(image=hsv_preview_tk)
        self.hsv_canvas.image = hsv_preview_tk
        
        # Grayscale
        gray_preview = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2GRAY)
        # Resize before converting back to RGB for display
        gray_preview = cv2.resize(gray_preview, (300, 200))
        # Convert to RGB for display (this will be a true grayscale image)
        gray_preview_display = cv2.cvtColor(gray_preview, cv2.COLOR_GRAY2RGB)
        gray_preview_pil = Image.fromarray(gray_preview_display)
        gray_preview_tk = ImageTk.PhotoImage(gray_preview_pil)
        self.gray_canvas.configure(image=gray_preview_tk)
        self.gray_canvas.image = gray_preview_tk
        
    def apply_bgr(self):
        """Apply BGR color mode (original)"""
        # Original is already in BGR, so no conversion needed
        self.window.destroy()
        
    def apply_rgb(self):
        """Apply RGB color mode"""
        result_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.image_processor.apply_operation(result_image, "Convert to RGB")
        self.window.destroy()
        
    def apply_hsv(self):
        """Apply HSV color mode"""
        result_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        self.image_processor.apply_operation(result_image, "Convert to HSV")
        self.window.destroy()
        
    def apply_gray(self):
        """Apply grayscale color mode"""
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.image_processor.apply_operation(result_image, "Convert to Grayscale")
        self.window.destroy()
