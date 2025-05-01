import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from .base_window import BaseWindow

class CompareWindow(BaseWindow):
    """Window for comparing original and edited images"""
    
    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Compare Original vs Edited", geometry="800x600")
        
        # Create a frame for the images
        self.images_frame = ctk.CTkFrame(self.window)
        self.images_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Create left frame for original image
        self.original_frame = ctk.CTkFrame(self.images_frame)
        self.original_frame.pack(side="left", expand=True, fill="both", padx=5, pady=5)
        
        # Create right frame for edited image
        self.edited_frame = ctk.CTkFrame(self.images_frame)
        self.edited_frame.pack(side="right", expand=True, fill="both", padx=5, pady=5)
        
        # Labels for the images
        self.original_label = ctk.CTkLabel(self.original_frame, text="Original")
        self.original_label.pack(pady=5)
        
        self.edited_label = ctk.CTkLabel(self.edited_frame, text="Edited")
        self.edited_label.pack(pady=5)
        
        # Canvases for the images
        self.original_canvas = ctk.CTkLabel(self.original_frame, text="")
        self.original_canvas.pack(expand=True, fill="both", padx=5, pady=5)
        
        self.edited_canvas = ctk.CTkLabel(self.edited_frame, text="")
        self.edited_canvas.pack(expand=True, fill="both", padx=5, pady=5)
        
        # Create close button
        self.button_frame = ctk.CTkFrame(self.window)
        self.button_frame.pack(pady=10)
        
        self.close_button = ctk.CTkButton(self.button_frame, text="Close", command=self.window.destroy)
        self.close_button.pack()
        
        # Display the images
        self.display_images()
        
    def display_images(self):
        """Display the original and edited images"""
        if self.image_processor.original_image is None or self.image_processor.cv_image is None:
            return
            
        # Get the images
        original_image = self.image_processor.original_image.copy()
        edited_image = self.image_processor.cv_image.copy()
        
        # Resize for display
        original_image = cv2.resize(original_image, (350, 300))
        edited_image = cv2.resize(edited_image, (350, 300))
        
        # Convert to RGB for display
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        edited_rgb = cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL images
        original_pil = Image.fromarray(original_rgb)
        edited_pil = Image.fromarray(edited_rgb)
        
        # Convert to Tkinter images
        self.original_tk = ImageTk.PhotoImage(original_pil)
        self.edited_tk = ImageTk.PhotoImage(edited_pil)
        
        # Display the images
        self.original_canvas.configure(image=self.original_tk)
        self.edited_canvas.configure(image=self.edited_tk)
