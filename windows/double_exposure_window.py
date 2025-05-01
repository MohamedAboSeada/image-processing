import customtkinter as ctk
import cv2
from tkinter import filedialog
from .base_window import BaseWindow

class DoubleExposureWindow(BaseWindow):
    """Window for double exposure effect"""
    
    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Double Exposure Effect", geometry="800x600")
        
        # Create a preview canvas
        self.create_preview_canvas()
        
        # Create controls frame
        self.controls_frame = ctk.CTkFrame(self.window)
        self.controls_frame.pack(pady=10)
        
        # Upload button for second image
        self.upload_button = ctk.CTkButton(
            self.controls_frame, 
            text="Upload Second Image", 
            command=self.upload_second_image
        )
        self.upload_button.pack(pady=5)
        
        # Alpha slider for blending
        self.alpha_frame = ctk.CTkFrame(self.controls_frame)
        self.alpha_frame.pack(pady=5)
        
        self.alpha_label = ctk.CTkLabel(self.alpha_frame, text="Blend Amount:")
        self.alpha_label.pack(side="left", padx=5)
        
        self.alpha_slider = ctk.CTkSlider(self.alpha_frame, from_=0.0, to=1.0, 
                                        number_of_steps=100, orientation="horizontal")
        self.alpha_slider.set(0.5)  # Default alpha value
        self.alpha_slider.pack(side="left", padx=10)
        
        self.alpha_value_label = ctk.CTkLabel(self.alpha_frame, text="0.5")
        self.alpha_value_label.pack(side="left", padx=5)
        
        # Create button frame
        self.create_button_frame()
        
        # Store original image
        self.original_image = self.image_processor.cv_image.copy()
        self.second_image = None
        
        # Connect the slider to the update function
        self.alpha_slider.configure(command=self.update_preview_blend)
        
        # Initial preview (just show the original image until second image is loaded)
        self.update_preview(self.original_image)
        
    def upload_second_image(self):
        """Upload a second image for blending"""
        path = filedialog.askopenfilename()
        if path:
            second_img = cv2.imread(path)
            if second_img is not None:
                # Resize second image to match dimensions of the first image
                self.second_image = cv2.resize(second_img, 
                                             (self.original_image.shape[1], 
                                              self.original_image.shape[0]))
                # Update preview with the blended image
                self.update_preview_blend(self.alpha_slider.get())
        
    def update_preview_blend(self, value):
        """Update the preview with the current blend setting"""
        alpha = float(value)
        self.alpha_value_label.configure(text=f"{alpha:.2f}")
        
        if self.second_image is None:
            # If no second image is loaded, just show the original
            self.update_preview(self.original_image)
            return
            
        # Blend the images
        beta = 1.0 - alpha
        preview_image = cv2.addWeighted(self.original_image, alpha, 
                                      self.second_image, beta, 0)
        
        self.update_preview(preview_image)
        
    def confirm(self):
        """Apply the double exposure effect and close the window"""
        if self.second_image is None:
            # If no second image is loaded, just close the window
            super().confirm()
            return
            
        alpha = float(self.alpha_slider.get())
        beta = 1.0 - alpha
        
        result_image = cv2.addWeighted(self.original_image, alpha, 
                                     self.second_image, beta, 0)
        
        # Apply the changes
        self.image_processor.apply_operation(result_image, "Double Exposure")
        
        # Close the window
        super().confirm()
