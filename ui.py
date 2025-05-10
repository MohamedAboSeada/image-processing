import customtkinter as ctk
import cv2
from tkinter import Menu
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

class ImageEditorUI:
    def __init__(self, root, app):
        self.root = root
        self.root.title("Advanced Image Editor")
        self.root.geometry("1000x700")
        self.app = app
        self.show_original = True  # Default to showing original image

        self.create_widgets()
        self.setup_menus()

    def create_widgets(self):
        # Frames
        self.frame_left = ctk.CTkFrame(self.root)
        self.frame_left.pack(side="left", fill="y", padx=10, pady=10)

        # Basic buttons
        basic_buttons = [
            ("Open Image", self.app.open_image),
            ("Save Image", self.app.save_image),
            ("Undo", self.app.undo),
            ("Redo", self.app.redo),
            ("Reset Image", self.app.reset_image),
            ("Close Image", self.app.close_image)
        ]

        for text, command in basic_buttons:
            ctk.CTkButton(self.frame_left, text=text, command=command).pack(pady=3)

        # Toggle original button
        self.toggle_button = ctk.CTkButton(
            self.frame_left,
            text="Hide Original",
            command=self.toggle_original,
            fg_color="green",  # Green when showing
            hover_color="#228B22"  # Lighter green on hover
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
        
        self.matplot_menu_button = ctk.CTkButton(self.frame_left, text="Matplot", command=self.show_matplot_menu)
        self.matplot_menu_button.pack(pady=5)

        # Add logo at the bottom of frame_left
        self.logo_image = ctk.CTkImage(Image.open("assets/logo-trans.png"),size=(70, 70))
        self.logo_label = ctk.CTkLabel(self.frame_left, image=self.logo_image, text="")
        self.logo_label.pack(side="bottom", pady=10)

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

        # Increase the zoom level for the matplotlib figures
        self.figure_original.set_size_inches(6, 6)
        self.figure_edited.set_size_inches(6, 6)

        self.canvas_original = FigureCanvasTkAgg(self.figure_original, self.original_container)
        self.canvas_original.get_tk_widget().configure(bg="#333")  # Set background color
        self.canvas_original.get_tk_widget().pack(expand=True, fill="both")

        self.canvas_edited = FigureCanvasTkAgg(self.figure_edited, self.edited_container)
        self.canvas_edited.get_tk_widget().configure(bg="#333")  # Set background color
        self.canvas_edited.get_tk_widget().pack(expand=True, fill="both")

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
            self.toggle_button.configure(text="Hide Original", fg_color="green", hover_color="#228B22")
        else:
            self.original_container.pack_forget()
            self.toggle_button.configure(text="Show Original", fg_color="green", hover_color="#228B22")
        # Update display to handle the new layout
        self.update_display(self.app.cv_image, self.app.original_image)

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
        self.toggle_button.configure(text="Hide Original", fg_color="green", hover_color="#228B22")

    def setup_menus(self):
        # Edit menu
        self.edit_menu = Menu(self.root, tearoff=0)
        self.edit_menu.add_command(label="Resize", command=self.app.resize_image)
        self.edit_menu.add_command(label="Crop", command=self.app.crop_image)
        self.edit_menu.add_command(label="Rotate", command=self.app.open_rotate_window)
        self.edit_menu.add_command(label="Flip", command=self.app.open_flip_window)
        self.edit_menu.add_command(label="Brightness", command=self.app.open_brightness_window)
        self.edit_menu.add_command(label="Color Conversion", command=self.app.open_color_conversion_window)
        self.edit_menu.add_command(label="Threshold", command=self.app.open_threshold_window)
        self.edit_menu.add_command(label="Double Exposure", command=self.app.open_double_exposure_window)
        self.edit_menu.add_command(label="Compare", command=self.app.open_compare_window)

        # Filter menu
        self.filter_menu = Menu(self.root, tearoff=0)
        self.filter_menu.add_command(label="Add Salt & Pepper Noise", command=self.app.add_salt_pepper_noise)
        self.filter_menu.add_command(label="Speckle Noise", command=self.app.apply_speckle_noise)
        self.filter_menu.add_command(label="Poisson Noise", command=self.app.apply_poisson_noise)
        self.filter_menu.add_command(label="Uniform Noise", command=self.app.apply_uniform_noise)
        self.filter_menu.add_command(label="Normalize", command=self.app.normalize_image)
        self.filter_menu.add_command(label="Equalize Histogram", command=self.app.equalize_histogram)
        self.filter_menu.add_command(label="Averaging Filter", command=self.app.apply_averaging_filter)
        self.filter_menu.add_command(label="Weighted Averaging Filter", command=self.app.apply_weighted_averaging_filter)
        self.filter_menu.add_command(label="Gaussian Filter", command=self.app.apply_gaussian_filter)
        self.filter_menu.add_command(label="Median Filter", command=self.app.open_median_filter_window)
        self.filter_menu.add_command(label="Max Filter", command=self.app.open_max_filter_window)
        self.filter_menu.add_command(label="Min Filter", command=self.app.open_min_filter_window)
        self.filter_menu.add_command(label="Laplacian", command=self.app.open_laplacian_window)
        self.filter_menu.add_command(label="LoG (Laplacian of Gaussian)", command=self.app.open_log_window)
        self.filter_menu.add_command(label="Custom Laplacian", command=self.app.open_custom_laplacian_window)
        
        # Matplot menu
        self.matplot_menu = Menu(self.root, tearoff=0)
        self.matplot_menu.add_command(label="Split Channels", command=self.app.split_channels)
        self.matplot_menu.add_command(label="Histogram", command=self.app.show_histogram)

    def show_edit_menu(self):
        x = self.edit_menu_button.winfo_rootx()
        y = self.edit_menu_button.winfo_rooty() + self.edit_menu_button.winfo_height()
        self.edit_menu.tk_popup(x, y)

    def show_filter_menu(self):
        x = self.filter_menu_button.winfo_rootx()
        y = self.filter_menu_button.winfo_rooty() + self.filter_menu_button.winfo_height()
        self.filter_menu.tk_popup(x, y)
        
    def show_matplot_menu(self):
        x = self.matplot_menu_button.winfo_rootx()
        y = self.matplot_menu_button.winfo_rooty() + self.matplot_menu_button.winfo_height()
        self.matplot_menu.tk_popup(x, y)
