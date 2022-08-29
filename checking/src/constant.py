import os

import pathlib2

ROOT = pathlib2.Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT / 'data'
BOOTSTRAP_DIR = DATA_DIR / 'bootstrap'
VOC_ROOT =  pathlib2.Path('VOCdevkit/')

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

# list of thresholds and path to MVC output
human_thld_D = {'frost': {}, 'brightness': {}}

human_thld_D['frost']['person'] = {}
human_thld_D['frost']['person']['L'] = {}
human_thld_D['frost']['person']['L']['cp'] = {0.9:'frost_0.9'}
human_thld_D['frost']['person']['L']['pp'] = {0.7:'frost_0.9', 0.3:'frost_0.3'}

human_thld_D['frost']['person']['C|L'] = {}
human_thld_D['frost']['person']['C|L']['cp'] = {0.9:'frost_0.9'}
human_thld_D['frost']['person']['C|L']['pp'] = {0.3:'frost_0.3'}

human_thld_D['frost']['person']['D'] = {}
human_thld_D['frost']['person']['D']['cp'] = {0.9:'frost_0.9'}
human_thld_D['frost']['person']['D']['pp'] = {0.3:'frost_0.3'}

human_thld_D['frost']['bus'] = {}
human_thld_D['frost']['bus']['L'] = {}
human_thld_D['frost']['bus']['L']['cp'] = {0.9:'frost_0.9', 0.7:'frost_0.9'}
human_thld_D['frost']['bus']['L']['pp'] = {0.3:'frost_0.3'}

human_thld_D['frost']['bus']['C|L'] = {}
human_thld_D['frost']['bus']['C|L']['cp'] = {0.7:'frost_0.9'}
human_thld_D['frost']['bus']['C|L']['pp'] = {0.3:'frost_0.3'}

human_thld_D['frost']['bus']['D'] = {}
human_thld_D['frost']['bus']['D']['cp'] = {0.7:'frost_0.9'}
human_thld_D['frost']['bus']['D']['pp'] = {0.3:'frost_0.3'}

human_thld_D['brightness']['person'] = {}
human_thld_D['brightness']['person']['L'] = {}
human_thld_D['brightness']['person']['L']['cp'] = {0.9:'brightness_0.9'}
human_thld_D['brightness']['person']['L']['pp'] = {0.2:'brightness_0.2'}

human_thld_D['brightness']['person']['C|L'] = {}
human_thld_D['brightness']['person']['C|L']['cp'] = {0.9:'brightness_0.9'}
human_thld_D['brightness']['person']['C|L']['pp'] = {0.8:'brightness_0.9', 0.2:'brightness_0.2'}

human_thld_D['brightness']['person']['D'] = {}
human_thld_D['brightness']['person']['D']['cp'] = {0.9:'brightness_0.9'}
human_thld_D['brightness']['person']['D']['pp'] = {0.2:'brightness_0.2'}

human_thld_D['brightness']['bird'] = {}
human_thld_D['brightness']['bird']['L'] = {}
human_thld_D['brightness']['bird']['L']['cp'] = {0.8:'brightness_0.9'}
human_thld_D['brightness']['bird']['L']['pp'] = {0.9:'brightness_0.9', 0.8:'brightness_0.9'}

human_thld_D['brightness']['bird']['C|L'] = {}
human_thld_D['brightness']['bird']['C|L']['cp'] = {0.9:'brightness_0.9', 0.8:'brightness_0.9'}
human_thld_D['brightness']['bird']['C|L']['pp'] = {0.8:'brightness_0.9'}

human_thld_D['brightness']['bird']['D'] = {}
human_thld_D['brightness']['bird']['D']['cp'] = {0.8:'brightness_0.9'}
human_thld_D['brightness']['bird']['D']['pp'] = {0.8:'brightness_0.9'}


human_thld_I = {'frost': {}, 'brightness': {}}

human_thld_I['frost']['person'] = {}
human_thld_I['frost']['person']['L'] = {}
human_thld_I['frost']['person']['L']['cp'] = {0.9:'frost_0.9', 0.6:'frost_0.9'}
human_thld_I['frost']['person']['L']['pp'] = {0.7:'frost_0.9', 0.3:'frost_0.3'}

human_thld_I['frost']['person']['C|L'] = {}
human_thld_I['frost']['person']['C|L']['cp'] = {0.9:'frost_0.9', 0.6:'frost_0.9'}
human_thld_I['frost']['person']['C|L']['pp'] = {0.3:'frost_0.3'}

human_thld_I['frost']['person']['S|C,L'] = {}
human_thld_I['frost']['person']['S|C,L']['cp'] = {0.6:'frost_0.9'}
human_thld_I['frost']['person']['S|C,L']['pp'] = {0.3:'frost_0.3'}

human_thld_I['frost']['person']['I'] = {}
human_thld_I['frost']['person']['I']['cp'] = {0.6:'frost_0.9'}
human_thld_I['frost']['person']['I']['pp'] = {0.3:'frost_0.3'}


human_thld_I['frost']['bus'] = {}
human_thld_I['frost']['bus']['L'] = {}
human_thld_I['frost']['bus']['L']['cp'] = {0.9:'frost_0.9', 0.7:'frost_0.9'}
human_thld_I['frost']['bus']['L']['pp'] = {0.3:'frost_0.3'}

human_thld_I['frost']['bus']['C|L'] = {}
human_thld_I['frost']['bus']['C|L']['cp'] = {0.7:'frost_0.9'}
human_thld_I['frost']['bus']['C|L']['pp'] = {0.3:'frost_0.9'}

human_thld_I['frost']['bus']['S|C,L'] = {}
human_thld_I['frost']['bus']['S|C,L']['cp'] = {0.7:'frost_0.9'}
human_thld_I['frost']['bus']['S|C,L']['pp'] = {0.3:'frost_0.9'}

human_thld_I['frost']['bus']['I'] = {}
human_thld_I['frost']['bus']['I']['cp'] = {0.7:'frost_0.9'}
human_thld_I['frost']['bus']['I']['pp'] = {0.3:'frost_0.3'}


human_thld_I['brightness']['person'] = {}
human_thld_I['brightness']['person']['L'] = {}
human_thld_I['brightness']['person']['L']['cp'] = {0.9:'brightness_0.9', 0.5:'brightness_0.9'}
human_thld_I['brightness']['person']['L']['pp'] = {0.2:'brightness_0.2'}

human_thld_I['brightness']['person']['C|L'] = {}
human_thld_I['brightness']['person']['C|L']['cp'] = {0.9:'brightness_0.9', 0.5:'brightness_0.9'}
human_thld_I['brightness']['person']['C|L']['pp'] = {0.8:'brightness_0.9', 0.2:'brightness_0.2'}

human_thld_I['brightness']['person']['S|C,L'] = {}
human_thld_I['brightness']['person']['S|C,L']['cp'] = {0.5:'brightness_0.9'}
human_thld_I['brightness']['person']['S|C,L']['pp'] = {0.2:'brightness_0.2'}

human_thld_I['brightness']['person']['I'] = {}
human_thld_I['brightness']['person']['I']['cp'] = {0.5:'brightness_0.9'}
human_thld_I['brightness']['person']['I']['pp'] = {0.2:'brightness_0.2'}


human_thld_I['brightness']['bird'] = {}
human_thld_I['brightness']['bird']['L'] = {}
human_thld_I['brightness']['bird']['L']['cp'] = {0.8:'brightness_0.9'}
human_thld_I['brightness']['bird']['L']['pp'] = {0.9:'brightness_0.9', 0.8:'brightness_0.9'}

human_thld_I['brightness']['bird']['C|L'] = {}
human_thld_I['brightness']['bird']['C|L']['cp'] = {0.9:'brightness_0.9', 0.8:'brightness_0.9'}
human_thld_I['brightness']['bird']['C|L']['pp'] = {0.8:'brightness_0.9'}

human_thld_I['brightness']['bird']['S|C,L'] = {}
human_thld_I['brightness']['bird']['S|C,L']['cp'] = {0.8:'brightness_0.9'}
human_thld_I['brightness']['bird']['S|C,L']['pp'] = {0.9:'brightness_0.9', 0.8:'brightness_0.9'}

human_thld_I['brightness']['bird']['I'] = {}
human_thld_I['brightness']['bird']['I']['cp'] = {0.8:'brightness_0.9'}
human_thld_I['brightness']['bird']['I']['pp'] = {0.8:'brightness_0.9'}


