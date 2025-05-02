import customtkinter as ctk
from PIL import Image
import cv2

class BaseWindow:
    """Base class for all window dialogs in the image editor"""

    def __init__(self, root, image_processor, title="Window", geometry="800x600"):
        """
        Initialize the base window

        Args:
            root: The parent window
            image_processor: The image processor instance
            title: Window title
            geometry: Window geometry (size)
        """
        self.window = ctk.CTkToplevel(root)
        self.window.title(title)
        self.window.geometry(geometry)
        self.window.attributes('-topmost', True)  # Make window stay on top

        self.image_processor = image_processor
        self.original_image = None
        self.preview_image = None

    def create_preview_canvas(self):
        """Create a canvas for image preview"""
        # Create main container frame
        self.main_container = ctk.CTkFrame(self.window)
        self.main_container.pack(expand=True, fill="both", padx=10, pady=10)

        # Create left frame for preview
        self.preview_frame = ctk.CTkFrame(self.main_container)
        self.preview_frame.pack(side="left", expand=True, fill="both", padx=5, pady=5)

        # Create canvas in preview frame
        self.canvas = ctk.CTkLabel(self.preview_frame, text="")
        self.canvas.pack(expand=True, fill="both")

        # Create right frame for controls
        self.controls_container = ctk.CTkFrame(self.main_container)
        self.controls_container.pack(side="right", fill="y", padx=5, pady=5)

    def create_button_frame(self):
        """Create a frame for buttons"""
        self.button_frame = ctk.CTkFrame(self.window)
        self.button_frame.pack(pady=10)

        self.confirm_button = ctk.CTkButton(self.button_frame, text="Confirm", command=self.confirm)
        self.confirm_button.pack(side="left", padx=5)

        self.cancel_button = ctk.CTkButton(self.button_frame, text="Cancel", command=self.cancel)
        self.cancel_button.pack(side="right", padx=5)

    def update_preview(self, cv_image, size=(600, 400)):
        """
        Update the preview image

        Args:
            cv_image: OpenCV image to display
            size: Size to resize the image to
        """
        if cv_image is None:
            return

        preview = cv_image.copy()
        preview = cv2.resize(preview, size)
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        preview = Image.fromarray(preview)

        # Use CTkImage instead of ImageTk.PhotoImage
        ctk_image = ctk.CTkImage(preview, size=size)

        self.canvas.configure(image=ctk_image)
        self.canvas.image = ctk_image  # Keep a reference to prevent garbage collection

    def confirm(self):
        """Confirm the changes and close the window"""
        # To be implemented by subclasses
        self.window.destroy()

    def cancel(self):
        """Cancel the changes and close the window"""
        self.window.destroy()
