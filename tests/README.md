# Image Editor Test Suite

This directory contains unit tests for the Image Editor application.

## Test Files

- `test_image_processor.py`: Tests for the `ImageProcessor` class, which contains the core image processing functionality.
- `test_app.py`: Tests for the main application initialization.
- `test_ui.py`: Tests for the UI components.
- `test_windows.py`: Tests for the window classes.

## Running Tests

To run all tests, use the `run_tests.py` script in the root directory:

```bash
python run_tests.py
```

To run a specific test file:

```bash
python -m unittest tests/test_image_processor.py
```

To run a specific test case:

```bash
python -m unittest tests.test_image_processor.TestImageProcessor.test_resize
```

## Test Coverage

The tests cover the following functionality:

### Image Processing Tests
- Basic operations: resize, crop, rotate, flip
- Color operations: brightness adjustment, color conversion, thresholding, normalization
- Noise operations: salt & pepper, speckle, Poisson, uniform
- Filter operations: averaging, weighted averaging, Gaussian, median, max, min
- Blending operations: image blending

### Application Tests
- Application initialization
- UI component initialization
- Window class functionality

## Adding New Tests

When adding new functionality to the application, please add corresponding tests to maintain code quality and prevent regressions.

1. For new image processing functions, add tests to `test_image_processor.py`
2. For new UI components, add tests to `test_ui.py`
3. For new window classes, add tests to `test_windows.py`

## Test Images

Test images are created programmatically in the test setup methods. This ensures that the tests are self-contained and don't rely on external files.
