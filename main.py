import customtkinter as ctk
from tkinter import filedialog, simpledialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Menu 
from ui import ImageEditorUI
from imageProcessing import ImageProcessor

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.image_processor = ImageProcessor(None)  # Initialize with None first
        self.ui = ImageEditorUI(root, self.image_processor)
        self.image_processor.ui = self.ui  # Set the UI reference after UI is created
            
    def open_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.image_processor.cv_image = cv2.imread(path)
            self.image_processor.original_image = self.image_processor.cv_image.copy()
            self.image_processor.undo_stack = []
            self.image_processor.history_list.clear()
            self.ui.refresh_history(self.image_processor.history_list)
            self.ui.update_display(self.image_processor.cv_image, self.image_processor.original_image)
            
    def save_image(self):
        if self.image_processor.cv_image is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png", 
                                              filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg")])
            if path:
                cv2.imwrite(path, self.image_processor.cv_image)
                
    def undo(self):
        if self.image_processor.undo_stack:
            self.image_processor.cv_image = self.image_processor.undo_stack.pop()
            self.image_processor.history_list.pop()
            self.ui.refresh_history(self.image_processor.history_list)
            self.ui.update_display(self.image_processor.cv_image, self.image_processor.original_image)

    def reset_image(self):
        if self.image_processor.original_image is not None:
            self.image_processor.cv_image = self.image_processor.original_image.copy()
            self.image_processor.undo_stack.clear()
            self.image_processor.history_list.clear()
            self.ui.refresh_history(self.image_processor.history_list)
            self.ui.update_display(self.image_processor.cv_image, self.image_processor.original_image)

if __name__ == '__main__':
    root = ctk.CTk()
    app = ImageEditorApp(root)
    root.mainloop()
