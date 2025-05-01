import customtkinter as ctk
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
        
if __name__ == '__main__':
    root = ctk.CTk()
    app = ImageEditorApp(root)
    root.mainloop()
