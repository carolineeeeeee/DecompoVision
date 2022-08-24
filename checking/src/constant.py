import pathlib2

ROOT = pathlib2.Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT / 'data'
BOOTSTRAP_DIR = DATA_DIR / 'bootstrap'
# transformations
GAUSSIAN_NOISE = "gaussian_noise"
DEFOCUS_BLUR = "defocus_blur"
FROST = "frost"
BRIGHTNESS = "brightness"
CONTRAST = "contrast"
JPEG_COMPRESSION = "jpeg_compression"
RGB = "RGB"
COLOR_JITTER = "color_jitter"
TRANSFORMATIONS = [JPEG_COMPRESSION, GAUSSIAN_NOISE, FROST, BRIGHTNESS, CONTRAST, DEFOCUS_BLUR]
TRANSFORMATION_LEVEL = 1000
