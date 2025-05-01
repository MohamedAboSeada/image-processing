# Advanced Image Editor

A feature-rich image editor built with Python, OpenCV, and CustomTkinter.

## Features

- Basic image operations: open, save, resize, crop, rotate, flip
- Color adjustments: brightness, color conversion, threshold, normalize
- Filters: averaging, Gaussian, median, max, min
- Noise generation: salt & pepper, speckle, Poisson, uniform
- Special effects: double exposure
- Analysis tools: histogram, channel splitting
- Undo/redo functionality
- Side-by-side comparison of original and edited images

## Project Structure

The application is organized into the following components:

- `main.py`: The main application entry point and app logic
- `ui.py`: User interface components
- `imageProcessing.py`: Core image processing functions
- `windows/`: Dialog windows for various operations
  - `base_window.py`: Base class for all windows
  - `rotate_window.py`: Window for rotation operations
  - `brightness_window.py`: Window for brightness adjustments
  - `color_conversion_window.py`: Window for color conversions
  - `threshold_window.py`: Window for threshold adjustments
  - `flip_window.py`: Window for flip operations
  - `filter_windows.py`: Windows for filter operations
  - `double_exposure_window.py`: Window for double exposure effect
  - `compare_window.py`: Window for comparing original and edited images
- `tests/`: Unit tests for the application

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Matplotlib
- CustomTkinter
- Pillow (PIL)
- imutils

## Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install opencv-python numpy matplotlib customtkinter pillow imutils
   ```
3. Run the application:
   ```
   python main.py
   ```

## Testing

The application includes a comprehensive test suite to ensure functionality works as expected.

To run the tests:

```
python run_tests.py
```

See the `tests/README.md` file for more information about the test suite.

## Keyboard Shortcuts

- `Ctrl+Z`: Undo
- `Ctrl+Y`: Redo

## License

This project is open source and available under the MIT License.
