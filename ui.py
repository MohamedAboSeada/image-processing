import customtkinter as ctk
import cv2
from tkinter import Menu
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageEditorUI:
    def __init__(self, root, image_processor):
        self.root = root
        self.root.title("Advanced Image Editor")
        self.root.geometry("1000x700")
        self.image_processor = image_processor
        self.show_original = True  # Default to showing original image

        self.create_widgets()
        self.setup_menus()
        
        # Bind window resize event
        self.root.bind('<Configure>', self.on_window_resize)

    def create_widgets(self):
        # Frames
        self.frame_left = ctk.CTkFrame(self.root)
        self.frame_left.pack(side="left", fill="y", padx=10, pady=10)

        # Basic buttons
        basic_buttons = [
            ("Open Image", self.image_processor.open_image),
            ("Save Image", self.image_processor.save_image),
            ("Undo", self.image_processor.undo),
            ("Redo", self.image_processor.redo),
            ("Reset Image", self.image_processor.reset_image),
            ("Close Image", self.image_processor.close_image)
        ]

        for text, command in basic_buttons:
            ctk.CTkButton(self.frame_left, text=text, command=command).pack(pady=3)

        # Toggle original button
        self.toggle_button = ctk.CTkButton(
            self.frame_left, 
            text="Hide Original", 
            command=self.toggle_original,
            fg_color="#8B0000",  # Dark red when showing
            hover_color="#A52A2A"  # Lighter red on hover
        )
        self.toggle_button.pack(pady=3)

        # History listbox
        self.history_listbox = ctk.CTkTextbox(self.frame_left, height=150, width=180)
        self.history_listbox.pack(pady=10)
        self.history_listbox.insert("end", "History:\n")
        
        # Menu buttons
        self.edit_menu_button = ctk.CTkButton(self.frame_left, text="Edit", command=self.show_edit_menu)
        self.edit_menu_button.pack(pady=5)

        self.filter_menu_button = ctk.CTkButton(self.frame_left, text="Filter", command=self.show_filter_menu)
        self.filter_menu_button.pack(pady=5)

        # Main image container
        self.image_frame = ctk.CTkFrame(self.root)
        self.image_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        
        # Create container for edited image (left)
        self.edited_container = ctk.CTkFrame(self.image_frame)
        self.edited_container.pack(side="left", expand=True, fill="both", padx=5, pady=5)
        
        # Caption for edited image
        self.edited_caption = ctk.CTkLabel(self.edited_container, text="Edited", font=("Arial", 12))
        self.edited_caption.pack(side="bottom", pady=5)
        
        # Create container for original image (right)
        self.original_container = ctk.CTkFrame(self.image_frame)
        self.original_container.pack(side="right", expand=True, fill="both", padx=5, pady=5)
        
        # Caption for original image
        self.original_caption = ctk.CTkLabel(self.original_container, text="Original", font=("Arial", 12))
        self.original_caption.pack(side="bottom", pady=5)
        
        # Matplotlib figures and canvases
        self.figure_original = plt.Figure(figsize=(5, 5), dpi=100)
        self.figure_original.patch.set_facecolor("#333")  # Set figure background color

        self.figure_edited = plt.Figure(figsize=(5, 5), dpi=100)
        self.figure_edited.patch.set_facecolor("#333")  # Set figure background color

        self.canvas_original = FigureCanvasTkAgg(self.figure_original, self.original_container)
        self.canvas_original.get_tk_widget().configure(bg="#333")  # Set background color
        self.canvas_original.get_tk_widget().pack(expand=True, fill="both")

        self.canvas_edited = FigureCanvasTkAgg(self.figure_edited, self.edited_container)
        self.canvas_edited.get_tk_widget().configure(bg="#333")  # Set background color
        self.canvas_edited.get_tk_widget().pack(expand=True, fill="both")

    def on_window_resize(self, event):
        if hasattr(self, 'image_label') and hasattr(self, 'image_label_original'):
            if self.image_processor.cv_image is not None and self.image_processor.original_image is not None:
                self.update_display(self.image_processor.cv_image, self.image_processor.original_image)

    def update_display(self, cv_image, original_image):
        if cv_image is None or original_image is None:
            return

        # Clear previous plots
        self.figure_original.clf()
        self.figure_edited.clf()

        # Plot original image
        if self.show_original:
            ax_original = self.figure_original.add_subplot(111, facecolor="#333")  # Set axes background color
            ax_original.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            ax_original.axis('off')
            self.canvas_original.draw()

        # Plot edited image
        ax_edited = self.figure_edited.add_subplot(111, facecolor="#333")  # Set axes background color
        ax_edited.imshow(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        ax_edited.axis('off')
        self.canvas_edited.draw()

    def toggle_original(self):
        self.show_original = not self.show_original
        if self.show_original:
            self.original_container.pack(side="right", expand=True, fill="both", padx=5, pady=5)
            self.toggle_button.configure(text="Hide Original", fg_color="#8B0000", hover_color="#A52A2A")
        else:
            self.original_container.pack_forget()
            self.toggle_button.configure(text="Show Original", fg_color="green", hover_color="#228B22")
        # Update display to handle the new layout
        self.update_display(self.image_processor.cv_image, self.image_processor.original_image)

    def refresh_history(self, history_list):
        self.history_listbox.delete("1.0", "end")
        self.history_listbox.insert("end", "History:\n")
        for h in history_list:
            self.history_listbox.insert("end", f"- {h}\n")

    def update_window_title(self, image_name):
        if image_name:
            self.root.title(f"Advanced Image Editor - {image_name}")
        else:
            self.root.title("Advanced Image Editor")

    def reset_interface(self):
        # Clear image displays
        self.figure_original.clf()
        self.figure_edited.clf()
        self.canvas_original.draw()
        self.canvas_edited.draw()
        
        # Reset window title
        self.update_window_title(None)
        
        # Clear history
        self.history_listbox.delete("1.0", "end")
        self.history_listbox.insert("end", "History:\n")
        
        # Show both containers
        self.show_original = True
        self.original_container.pack(side="right", expand=True, fill="both", padx=5, pady=5)
        self.toggle_button.configure(text="Hide Original", fg_color="#8B0000", hover_color="#A52A2A")

    def setup_menus(self):
        # Edit menu
        self.edit_menu = Menu(self.root, tearoff=0)
        self.edit_menu.add_command(label="Resize", command=self.image_processor.resize_image)
        self.edit_menu.add_command(label="Crop", command=self.image_processor.crop_image)
        self.edit_menu.add_command(label="Rotate with Crop", command=self.image_processor.rotate_image_with_crop)
        self.edit_menu.add_command(label="Rotate without Crop", command=self.image_processor.rotate_image_without_crop)
        self.edit_menu.add_command(label="Flip", command=self.image_processor.adjust_flip)
        self.edit_menu.add_command(label="Brightness", command=self.image_processor.adjust_brightness_window)
        self.edit_menu.add_command(label="Convert to RGB", command=self.image_processor.convert_Rgb)
        self.edit_menu.add_command(label="Convert to HSV", command=self.image_processor.convert_Hsv)
        self.edit_menu.add_command(label="Convert to Gray", command=self.image_processor.convert_Gray)
        self.edit_menu.add_command(label="Convert to Binary", command=self.image_processor.adjust_threshold)
        self.edit_menu.add_command(label="Split Channels", command=self.image_processor.split_channels)
        self.edit_menu.add_command(label="Normalize", command=self.image_processor.normalize_image)
        self.edit_menu.add_command(label="Histogram", command=self.image_processor.show_histogram)
        self.edit_menu.add_command(label="Equalize Histogram", command=self.image_processor.equalize_histogram)
        self.edit_menu.add_command(label="Image Addition", command=self.image_processor.image_addition)
        self.edit_menu.add_command(label="Weighted Addition", command=self.image_processor.weighted_addition)
        self.edit_menu.add_command(label="Subtraction", command=self.image_processor.image_subtraction)

        # Filter menu
        self.filter_menu = Menu(self.root, tearoff=0)
        self.filter_menu.add_command(label="Add Salt & Pepper Noise", command=self.image_processor.add_salt_pepper_noise)
        self.filter_menu.add_command(label="Averaging Filter", command=self.image_processor.apply_averaging_filter)
        self.filter_menu.add_command(label="Weighted Averaging Filter", command=self.image_processor.apply_weighted_averaging_filter)
        self.filter_menu.add_command(label="Gaussian Filter", command=self.image_processor.apply_gaussian_filter)
        self.filter_menu.add_command(label="Median Filter", command=self.image_processor.apply_median_filter)
        self.filter_menu.add_command(label="Max Filter", command=self.image_processor.apply_max_filter)
        self.filter_menu.add_command(label="Min Filter", command=self.image_processor.apply_min_filter)

    def show_edit_menu(self):
        x = self.edit_menu_button.winfo_rootx()
        y = self.edit_menu_button.winfo_rooty() + self.edit_menu_button.winfo_height()
        self.edit_menu.tk_popup(x, y)

    def show_filter_menu(self):
        x = self.filter_menu_button.winfo_rootx()
        y = self.filter_menu_button.winfo_rooty() + self.filter_menu_button.winfo_height()
        self.filter_menu.tk_popup(x, y)
