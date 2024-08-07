from ocr import *
import easyocr
import warnings
import spacy
import json
import yaml

# Ignore all warnings
warnings.simplefilter(action='ignore')

# Load configuration from a YAML file
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Retrieve configuration parameters
outputFilePath = config.get('outputFilePath', './output.json')  # Path for output JSON file
rootDir = config.get('rootDir', './images')  # Directory containing images
easyocrLang = config.get('easyocrLanguages', ['en'])  # Languages for EasyOCR
spacyModel = config.get('spacyModel', 'en_core_web_trf')  # SpaCy model for NER
model = config.get('model', 'Yolo')  # Model to use for speech bubble detection
modelpath = config.get('modelpath', "./runs/detect/train/weights/best.pt")  # Path to the YOLO model weights

# Initialize YOLO model if not using the 'simple' approach
if model != 'simple':
    model = YOLO(modelpath)

# Load SpaCy model for NER
nlp = spacy.load(spacyModel)

# Initialize EasyOCR reader with specified languages
reader = easyocr.Reader(easyocrLang)

# Process images to extract and preprocess speech bubbles
comic = preprocessor(rootDir, reader, model)

# Perform Named Entity Recognition (NER) and update the output JSON file
result = NER(comic, nlp, outputFilePath)
