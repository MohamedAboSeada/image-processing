import customtkinter as ctk
import cv2
import imutils
from .base_window import BaseWindow

class RotateWindow(BaseWindow):
    """Window for rotating images"""
    
    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Rotate Image", geometry="800x600")
        
        # Create a preview canvas
        self.create_preview_canvas()
        
        # Create controls frame
        self.controls_frame = ctk.CTkFrame(self.window)
        self.controls_frame.pack(pady=10)
        
        # Angle slider
        self.angle_frame = ctk.CTkFrame(self.controls_frame)
        self.angle_frame.pack(pady=5)
        
        self.angle_label = ctk.CTkLabel(self.angle_frame, text="Angle:")
        self.angle_label.pack(side="left", padx=5)
        
        self.angle_slider = ctk.CTkSlider(self.angle_frame, from_=-180, to=180, 
                                         number_of_steps=360, orientation="horizontal", width=300)
        self.angle_slider.set(0)  # Default angle
        self.angle_slider.pack(side="left", padx=10)
        
        self.angle_value_label = ctk.CTkLabel(self.angle_frame, text="0°")
        self.angle_value_label.pack(side="left", padx=5)
        
        # Checkbox for crop/no crop
        self.crop_var = ctk.BooleanVar(value=True)
        self.crop_checkbox = ctk.CTkCheckBox(self.controls_frame, 
                                           text="Maintain original dimensions (crop)", 
                                           variable=self.crop_var)
        self.crop_checkbox.pack(pady=5)
        
        # Connect the slider and checkbox to the update function
        self.angle_slider.configure(command=lambda val: self.update_preview_rotation())
        self.crop_var.trace_add("write", self.update_preview_rotation)
        
        # Create button frame
        self.create_button_frame()
        
        # Store original image
        self.original_image = self.image_processor.cv_image.copy()
        
        # Initial preview
        self.update_preview_rotation()
        
    def update_preview_rotation(self, *args):
        """Update the preview with the current rotation settings"""
        angle = self.angle_slider.get()
        self.angle_value_label.configure(text=f"{int(angle)}°")
        
        preview_image = self.original_image.copy()
        if self.crop_var.get():
            # Rotate with crop (maintain dimensions)
            preview_image = imutils.rotate_bound(preview_image, angle)
        else:
            # Rotate without crop (may show black corners)
            preview_image = imutils.rotate(preview_image, angle)
        
        self.update_preview(preview_image)
        
    def confirm(self):
        """Apply the rotation and close the window"""
        angle = self.angle_slider.get()
        with_crop = self.crop_var.get()
        
        if with_crop:
            operation_name = "Rotate with Crop"
            result_image = imutils.rotate_bound(self.original_image, angle)
        else:
            operation_name = "Rotate without Crop"
            result_image = imutils.rotate(self.original_image, angle)
        
        # Apply the changes
        self.image_processor.apply_operation(result_image, operation_name)
        
        # Close the window
        super().confirm()
