import unittest
import cv2
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add the parent directory to the path so we can import the ImageProcessor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imageProcessing import ImageProcessor

class TestImageProcessor(unittest.TestCase):
    """Test cases for the ImageProcessor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.processor = ImageProcessor()

        # Create a test directory if it doesn't exist
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_images')
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a simple test image (100x100 with a white square in the middle)
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = [255, 255, 255]  # White square

        # Save the test image
        self.test_image_path = os.path.join(self.test_dir, 'test_image.png')
        cv2.imwrite(self.test_image_path, self.test_image)

        # Create a gradient test image
        self.gradient_image = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            self.gradient_image[:, i] = [i * 2.55, i * 2.55, i * 2.55]

        # Save the gradient test image
        self.gradient_image_path = os.path.join(self.test_dir, 'gradient_image.png')
        cv2.imwrite(self.gradient_image_path, self.gradient_image)

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove test images
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        if os.path.exists(self.gradient_image_path):
            os.remove(self.gradient_image_path)

    def test_resize(self):
        """Test the resize method"""
        # Resize to 50x50
        resized = self.processor.resize(self.test_image, 50, 50)
        self.assertEqual(resized.shape, (50, 50, 3))

        # Resize to 200x200
        resized = self.processor.resize(self.test_image, 200, 200)
        self.assertEqual(resized.shape, (200, 200, 3))

    def test_crop(self):
        """Test the crop method"""
        # Crop the center square
        cropped = self.processor.crop(self.test_image, 25, 25, 50, 50)
        self.assertEqual(cropped.shape, (50, 50, 3))

        # Check that the cropped image is all white
        self.assertTrue(np.all(cropped == [255, 255, 255]))

    def test_rotate(self):
        """Test the rotate method"""
        # Rotate 90 degrees with crop
        rotated = self.processor.rotate(self.test_image, 90, True)
        self.assertEqual(rotated.shape, (100, 100, 3))

        # Rotate 45 degrees without crop
        rotated = self.processor.rotate(self.test_image, 45, False)
        self.assertEqual(rotated.shape, (100, 100, 3))

    def test_adjust_brightness(self):
        """Test the adjust_brightness method"""
        # Increase brightness
        brightened = self.processor.adjust_brightness(self.test_image, 1.5)
        self.assertEqual(brightened.shape, self.test_image.shape)

        # Check that the white square is still white
        self.assertTrue(np.all(brightened[25:75, 25:75] == [255, 255, 255]))

        # Decrease brightness
        darkened = self.processor.adjust_brightness(self.test_image, 0.5)
        self.assertEqual(darkened.shape, self.test_image.shape)

        # Check that the white square is now darker
        self.assertTrue(np.all(darkened[25:75, 25:75] < [255, 255, 255]))

    def test_convert_color(self):
        """Test the convert_color method"""
        # Convert to RGB
        rgb = self.processor.convert_color(self.test_image, 'rgb')
        self.assertEqual(rgb.shape, self.test_image.shape)

        # Convert to HSV
        hsv = self.processor.convert_color(self.test_image, 'hsv')
        self.assertEqual(hsv.shape, self.test_image.shape)

        # Convert to grayscale
        gray = self.processor.convert_color(self.test_image, 'gray')
        self.assertEqual(gray.shape, self.test_image.shape)

    def test_apply_threshold(self):
        """Test the apply_threshold method"""
        # Apply threshold at 128
        thresholded = self.processor.apply_threshold(self.gradient_image, 128)
        self.assertEqual(thresholded.shape, self.gradient_image.shape)

        # Check that pixels with value > 128 are white and others are black
        gray = cv2.cvtColor(self.gradient_image, cv2.COLOR_BGR2GRAY)
        expected = np.zeros_like(gray)
        expected[gray > 128] = 255
        expected = cv2.cvtColor(expected, cv2.COLOR_GRAY2BGR)

        # Allow for small differences due to conversion
        self.assertTrue(np.allclose(thresholded, expected, atol=1))

    def test_flip(self):
        """Test the flip method"""
        # Horizontal flip
        flipped_h = self.processor.flip(self.test_image, 1)
        self.assertEqual(flipped_h.shape, self.test_image.shape)

        # Vertical flip
        flipped_v = self.processor.flip(self.test_image, 0)
        self.assertEqual(flipped_v.shape, self.test_image.shape)

        # Both horizontal and vertical flip
        flipped_both = self.processor.flip(self.test_image, -1)
        self.assertEqual(flipped_both.shape, self.test_image.shape)

    def test_normalize(self):
        """Test the normalize method"""
        # Create an image with low contrast
        low_contrast = np.zeros((100, 100, 3), dtype=np.uint8)
        low_contrast[25:75, 25:75] = [50, 50, 50]  # Gray square

        # Normalize the image
        normalized = self.processor.normalize(low_contrast)
        self.assertEqual(normalized.shape, low_contrast.shape)

        # Check that the normalized image has higher contrast
        self.assertTrue(np.max(normalized) > np.max(low_contrast))

    def test_add_salt_pepper_noise(self):
        """Test the add_salt_pepper_noise method"""
        # Add salt and pepper noise
        noisy = self.processor.add_salt_pepper_noise(self.test_image)
        self.assertEqual(noisy.shape, self.test_image.shape)

        # Check that some pixels are now 0 (pepper) or 255 (salt)
        self.assertTrue(np.any(noisy[0:25, 0:25] == 0) or np.any(noisy[0:25, 0:25] == 255))

    def test_apply_speckle_noise(self):
        """Test the apply_speckle_noise method"""
        # Add speckle noise
        noisy = self.processor.apply_speckle_noise(self.test_image)
        self.assertEqual(noisy.shape, self.test_image.shape)

        # Check that the image has changed
        self.assertTrue(np.any(noisy != self.test_image))

    def test_apply_poisson_noise(self):
        """Test the apply_poisson_noise method"""
        # Add Poisson noise
        noisy = self.processor.apply_poisson_noise(self.test_image)
        self.assertEqual(noisy.shape, self.test_image.shape)

        # Check that the image has changed
        self.assertTrue(np.any(noisy != self.test_image))

    def test_apply_uniform_noise(self):
        """Test the apply_uniform_noise method"""
        # Add uniform noise
        noisy = self.processor.apply_uniform_noise(self.test_image)
        self.assertEqual(noisy.shape, self.test_image.shape)

        # Check that the image has changed
        self.assertTrue(np.any(noisy != self.test_image))

    def test_apply_averaging_filter(self):
        """Test the apply_averaging_filter method"""
        # Apply averaging filter
        filtered = self.processor.apply_averaging_filter(self.test_image)
        self.assertEqual(filtered.shape, self.test_image.shape)

    def test_apply_weighted_averaging_filter(self):
        """Test the apply_weighted_averaging_filter method"""
        # Apply weighted averaging filter
        filtered = self.processor.apply_weighted_averaging_filter(self.test_image)
        self.assertEqual(filtered.shape, self.test_image.shape)

    def test_apply_gaussian_filter(self):
        """Test the apply_gaussian_filter method"""
        # Apply Gaussian filter
        filtered = self.processor.apply_gaussian_filter(self.test_image)
        self.assertEqual(filtered.shape, self.test_image.shape)

    def test_apply_median_filter(self):
        """Test the apply_median_filter method"""
        # Add salt and pepper noise
        noisy = self.processor.add_salt_pepper_noise(self.test_image, 0.1, 0.1)

        # Apply median filter
        filtered = self.processor.apply_median_filter(noisy)
        self.assertEqual(filtered.shape, noisy.shape)

        # Check that the filter has reduced the noise
        # (This is a simple check - the filtered image should be closer to the original)
        diff_noisy = np.sum(np.abs(noisy - self.test_image))
        diff_filtered = np.sum(np.abs(filtered - self.test_image))
        self.assertTrue(diff_filtered < diff_noisy)

    def test_apply_max_filter(self):
        """Test the apply_max_filter method"""
        # Apply max filter
        filtered = self.processor.apply_max_filter(self.test_image)
        self.assertEqual(filtered.shape, self.test_image.shape)

    def test_apply_min_filter(self):
        """Test the apply_min_filter method"""
        # Apply min filter
        filtered = self.processor.apply_min_filter(self.test_image)
        self.assertEqual(filtered.shape, self.test_image.shape)

    def test_blend_images(self):
        """Test the blend_images method"""
        # Create a second test image (100x100 with a black square in the middle)
        test_image2 = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White background
        test_image2[25:75, 25:75] = [0, 0, 0]  # Black square

        # Blend the images
        blended = self.processor.blend_images(self.test_image, test_image2, 0.5)
        self.assertEqual(blended.shape, self.test_image.shape)

        # Check that the blended image has gray in the middle (blend of white and black)
        # Allow for small differences due to rounding
        self.assertTrue(np.allclose(blended[25:75, 25:75], [127, 127, 127], atol=1))

if __name__ == '__main__':
    unittest.main()
