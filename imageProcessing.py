import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    """Class for image processing operations"""

    def __init__(self):
        """Initialize the image processor"""
        pass

    def resize(self, image, width, height):
        """
        Resize an image to the specified dimensions

        Args:
            image: The image to resize
            width: The target width
            height: The target height

        Returns:
            The resized image
        """
        return cv2.resize(image, (width, height))

    def crop(self, image, x, y, width, height):
        """
        Crop an image to the specified region

        Args:
            image: The image to crop
            x: The x-coordinate of the top-left corner
            y: The y-coordinate of the top-left corner
            width: The width of the region
            height: The height of the region

        Returns:
            The cropped image
        """
        return image[y:y+height, x:x+width]

    def rotate(self, image, angle, with_crop=True):
        """
        Rotate an image by the specified angle

        Args:
            image: The image to rotate
            angle: The rotation angle in degrees
            with_crop: Whether to crop the image to maintain dimensions

        Returns:
            The rotated image
        """
        import imutils
        if with_crop:
            return imutils.rotate_bound(image, angle)
        else:
            return imutils.rotate(image, angle)

    def adjust_brightness(self, image, factor):
        """
        Adjust the brightness of an image

        Args:
            image: The image to adjust
            factor: The brightness factor (0.1 to 3.0)

        Returns:
            The brightness-adjusted image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def convert_color(self, image, conversion_type):
        """
        Convert an image to a different color space

        Args:
            image: The image to convert
            conversion_type: The type of conversion ('rgb', 'hsv', 'gray')

        Returns:
            The converted image
        """
        if conversion_type == 'rgb':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif conversion_type == 'hsv':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif conversion_type == 'gray':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return image

    def apply_threshold(self, image, threshold_value):
        """
        Apply a threshold to an image

        Args:
            image: The image to threshold
            threshold_value: The threshold value (0-255)

        Returns:
            The thresholded image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def flip(self, image, flip_code):
        """
        Flip an image

        Args:
            image: The image to flip
            flip_code: The flip code (1=horizontal, 0=vertical, -1=both)

        Returns:
            The flipped image
        """
        return cv2.flip(image, flip_code)

    def normalize(self, image):
        """
        Normalize an image

        Args:
            image: The image to normalize

        Returns:
            The normalized image
        """
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    def split_channels(self, image):
        """
        Split an image into its color channels and display them

        Args:
            image: The image to split
        """
        b, g, r = cv2.split(image)
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(b, cmap='gray')
        plt.title("Blue Channel")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(g, cmap='gray')
        plt.title("Green Channel")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(r, cmap='gray')
        plt.title("Red Channel")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def show_histogram(self, image):
        """
        Display the histogram of an image

        Args:
            image: The image to analyze
        """
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.title("Histogram")
        plt.show()

    def equalize_histogram(self, image):
        """
        Equalize the histogram of an image

        Args:
            image: The image to equalize

        Returns:
            The equalized image
        """
        # Convert to YUV color space
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Equalize the Y channel
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        # Convert back to BGR
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def add_salt_pepper_noise(self, image, salt_prob=0.03, pepper_prob=0.03):
        """
        Add salt and pepper noise to an image

        Args:
            image: The image to add noise to
            salt_prob: Probability of salt noise
            pepper_prob: Probability of pepper noise

        Returns:
            The noisy image
        """
        noisy = image.copy()
        # Get image dimensions
        height, width = image.shape[:2]

        # Handle both color and grayscale images
        if len(image.shape) == 3:  # Color image
            # Salt noise
            num_salt = int(height * width * salt_prob)
            for _ in range(num_salt):
                y = np.random.randint(0, height)
                x = np.random.randint(0, width)
                noisy[y, x] = [255, 255, 255]

            # Pepper noise
            num_pepper = int(height * width * pepper_prob)
            for _ in range(num_pepper):
                y = np.random.randint(0, height)
                x = np.random.randint(0, width)
                noisy[y, x] = [0, 0, 0]
        else:  # Grayscale image
            # Salt noise
            salt_mask = np.random.random(image.shape[:2]) < salt_prob
            noisy[salt_mask] = 255
            # Pepper noise
            pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
            noisy[pepper_mask] = 0

        return noisy

    def apply_speckle_noise(self, image, intensity=0.3):
        """
        Add speckle noise to an image

        Args:
            image: The image to add noise to
            intensity: The noise intensity

        Returns:
            The noisy image
        """
        # Convert to float for calculations
        image_float = image.astype(float)

        # Create noise with same shape as image
        if len(image.shape) == 3:  # Color image
            height, width, channels = image.shape
            noise = np.random.normal(0, intensity, (height, width, channels))
        else:  # Grayscale image
            noise = np.random.normal(0, intensity, image.shape)

        # Apply speckle noise formula: J = I + I*n
        noisy = image_float + image_float * noise

        # Clip values and convert back to uint8
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_poisson_noise(self, image):
        """
        Add Poisson noise to an image

        Args:
            image: The image to add noise to

        Returns:
            The noisy image
        """
        # Create a copy of the image to avoid modifying the original
        noisy = np.zeros_like(image, dtype=np.float32)

        # Scale factor to make noise more visible
        scale_factor = 0.5  # Lower values make noise more pronounced

        # Handle color and grayscale images
        if len(image.shape) == 3:  # Color image
            # Process each channel separately
            for c in range(image.shape[2]):
                # Scale values for Poisson distribution (which works better with larger values)
                vals = image[..., c].astype(np.float32) * scale_factor
                # Generate noise using Poisson distribution
                noise = np.random.poisson(np.maximum(vals, 1)).astype(np.float32) / scale_factor
                # Add some contrast to make noise more visible
                noisy[..., c] = noise
        else:  # Grayscale image
            vals = image.astype(np.float32) * scale_factor
            noise = np.random.poisson(np.maximum(vals, 1)).astype(np.float32) / scale_factor
            noisy = noise

        # Normalize back to 0-255 range and convert to uint8
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_uniform_noise(self, image, low=-50, high=50):
        """
        Add uniform noise to an image

        Args:
            image: The image to add noise to
            low: Lower bound of noise
            high: Upper bound of noise

        Returns:
            The noisy image
        """
        # Convert to float for calculations
        image_float = image.astype(float)

        # Generate uniform noise with same shape as image
        if len(image.shape) == 3:  # Color image
            height, width, channels = image.shape
            noise = np.random.uniform(low, high, (height, width, channels))
        else:  # Grayscale image
            noise = np.random.uniform(low, high, image.shape)

        # Add noise to image
        noisy = image_float + noise

        # Clip values and convert back to uint8
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_averaging_filter(self, image, kernel_size=3):
        """
        Apply an averaging filter to an image

        Args:
            image: The image to filter
            kernel_size: The size of the kernel

        Returns:
            The filtered image
        """
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        return cv2.filter2D(image, -1, kernel)

    def apply_weighted_averaging_filter(self, image, kernel_size=3):
        """
        Apply a weighted averaging filter to an image

        Args:
            image: The image to filter
            kernel_size: The size of the kernel

        Returns:
            The filtered image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def apply_gaussian_filter(self, image, kernel_size=3, sigma=1.0):
        """
        Apply a Gaussian filter to an image

        Args:
            image: The image to filter
            kernel_size: The size of the kernel
            sigma: The standard deviation

        Returns:
            The filtered image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def apply_median_filter(self, image, kernel_size=3):
        """
        Apply a median filter to an image

        Args:
            image: The image to filter
            kernel_size: The size of the kernel

        Returns:
            The filtered image
        """
        return cv2.medianBlur(image, kernel_size)

    def apply_max_filter(self, image, kernel_size=3):
        """
        Apply a max filter (dilation) to an image

        Args:
            image: The image to filter
            kernel_size: The size of the kernel

        Returns:
            The filtered image
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(image, kernel)

    def apply_min_filter(self, image, kernel_size=3):
        """
        Apply a min filter (erosion) to an image

        Args:
            image: The image to filter
            kernel_size: The size of the kernel

        Returns:
            The filtered image
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(image, kernel)

    def apply_laplacian(self, image, kernel_size=3, alpha=0.5):
        """
        Apply Laplacian sharpening: add the Laplacian result to the original image.
        Args:
            image: The image to process
            kernel_size: The size of the Laplacian kernel (must be odd)
            alpha: The scaling factor for the Laplacian result
        Returns:
            The sharpened image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        if len(image.shape) == 3:
            # Color: apply Laplacian to each channel
            channels = cv2.split(image)
            sharp_channels = []
            for ch in channels:
                lap = cv2.Laplacian(ch, cv2.CV_16S, ksize=kernel_size)
                lap = cv2.convertScaleAbs(lap)
                sharp = cv2.addWeighted(ch, 1.0, lap, alpha, 0)
                sharp_channels.append(sharp)
            out = cv2.merge(sharp_channels)
        else:
            lap = cv2.Laplacian(image, cv2.CV_16S, ksize=kernel_size)
            lap = cv2.convertScaleAbs(lap)
            out = cv2.addWeighted(image, 1.0, lap, alpha, 0)
        return np.clip(out, 0, 255).astype(np.uint8)

    def blend_images(self, image1, image2, alpha=0.5):
        """
        Blend two images together

        Args:
            image1: The first image
            image2: The second image
            alpha: The blending factor (0.0 to 1.0)

        Returns:
            The blended image
        """
        beta = 1.0 - alpha
        return cv2.addWeighted(image1, alpha, image2, beta, 0)

    def apply_log(self, image, kernel_size=3, sigma=1.0, alpha=0.9):
        """
        Apply Laplacian of Gaussian (LoG) filter to an image, matching the provided code logic.

        Args:
            image: The image to process
            kernel_size: The size of the Laplacian kernel (must be odd)
            sigma: The standard deviation for Gaussian blur
            alpha: The blending factor for addWeighted

        Returns:
            The processed image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        # Gaussian blur
        g = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        # Convert to grayscale
        if len(g.shape) == 3:
            img_g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        else:
            img_g = g.copy()
        # Laplacian
        shar = cv2.Laplacian(img_g, -1, ksize=kernel_size)
        # Blend
        out = cv2.addWeighted(img_g, alpha, shar, alpha, 0)
        # Convert back to BGR if original was color
        if len(image.shape) == 3:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        return out

    def apply_custom_laplacian(self, image, alpha=0.5):
        """
        Apply a custom Laplacian kernel [[1,1,1],[1,-8,1],[1,1,1]] and add the result to the original image for sharpening.
        Args:
            image: The image to process
            alpha: The scaling factor for the Laplacian result
        Returns:
            The sharpened image
        """
        kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
        if len(image.shape) == 3:
            channels = cv2.split(image)
            sharp_channels = []
            for ch in channels:
                lap = cv2.filter2D(ch, -1, kernel)
                sharp = cv2.addWeighted(ch, 1.0, lap, alpha, 0)
                sharp_channels.append(sharp)
            out = cv2.merge(sharp_channels)
        else:
            lap = cv2.filter2D(image, -1, kernel)
            out = cv2.addWeighted(image, 1.0, lap, alpha, 0)
        return np.clip(out, 0, 255).astype(np.uint8)
