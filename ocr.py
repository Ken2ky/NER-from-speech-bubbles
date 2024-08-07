import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import easyocr
import json
from ultralytics import YOLO

def findSpeechBubbles(imagePath, model):
    speechBubbles = []
    
    # Check if the chosen model is 'simple'
    if model == 'simple':
        image = cv2.imread(imagePath)
        
        # Verify if the image is loaded correctly
        if image is None:
            raise ValueError(f"Image not found or unable to load: {imagePath}")
        
        # Convert image to grayscale
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        imageGrayBlur = cv2.GaussianBlur(imageGray, (3, 3), 0)
        
        # Apply binary thresholding to highlight speech bubbles
        _, binary = cv2.threshold(imageGrayBlur, 235, 255, cv2.THRESH_BINARY)
        
        # Find contours of the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
        for contour in contours:
            try:
                # Get bounding rectangle for each contour
                [x, y, w, h] = cv2.boundingRect(contour)
                
            except Exception as e:
                print(f"Error processing contour: {e}")
            
            # Filter out speech bubble candidates based on size
            if w < 500 and w > 60 and h < 500 and h > 25:
                croppedImage = image[y:y+h, x:x+w]
                speechBubbles.append(croppedImage)
    
    else:
        # Use YOLO model for speech bubble detection
        results = model.predict(imagePath, save=False, imgsz=960, conf=0.5, task='detect')
        for result in results:
            boxes = result.boxes.xyxy  # Get bounding box coordinates
            image = cv2.imread(imagePath)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                croppedImage = image[y1:y2, x1:x2]
                speechBubbles.append(croppedImage)
          
    return speechBubbles

def image_to_text(imageList, reader):
    op = []
    for image in imageList:
        result = reader.readtext(image, detail=0)
        if len(result) > 0:
            text = ' '.join(result)
            op.append(text)
        
    return '. '.join(op)
    
def looper(rootDir):
    fileNameList = []
    filePathList = []
    
    # Traverse directory and gather image file paths
    for subDir, dirs, files in os.walk(rootDir):
        for file in files:
            fileInfo = file.split('.')
            fileName, fileExten = fileInfo[0], fileInfo[-1].lower()  # Convert extension to lowercase
            filePath = os.path.join(subDir, file)
            if fileExten in ['jpg', 'png']:  # Check for image file extensions
                if fileName not in fileNameList:
                    fileNameList.append(fileName)
                    filePathList.append(filePath)
    return filePathList

def denoise(image, n):
    for i in range(n):
        image = cv2.fastNlMeansDenoisingColored(image)

    return image
    
def preprocessor(rootDir, reader, model):
    comic = []
    for imagePath in looper(rootDir):
        print(imagePath)
        
        # Find speech bubbles in each image
        try:  
            croppedImageList = findSpeechBubbles(imagePath, model)
        except Exception as e:
            print(e)
            continue
        
        croppedImageList = croppedImageList[::-1]
        speechBubbles = []
        
        for croppedImage in croppedImageList:
            # Enlarge image
            croppedImage = cv2.resize(croppedImage, (0, 0), fx=2, fy=2)
            # Denoise image
            croppedImage = denoise(croppedImage, 2)
            kernel = np.ones((1, 1), np.uint8)
            croppedImage = cv2.dilate(croppedImage, kernel, iterations=50)
            croppedImage = cv2.erode(croppedImage, kernel, iterations=50)

            # Convert to grayscale
            croppedImageGray = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian filter
            croppedImageGrayBlur = cv2.GaussianBlur(croppedImageGray, (5, 5), 0)
            # Perform edge detection
            croppedImageGrayBlurLaplacian = cv2.Laplacian(croppedImageGrayBlur, cv2.CV_64F)
            # Adjust contrast and brightness
            croppedImageGrayBlurLaplacian = np.uint8(np.clip((10 * croppedImageGrayBlurLaplacian + 10), 0, 255))
            
            # Append processed image for OCR
            speechBubbles.append(croppedImageGrayBlurLaplacian)

        text = image_to_text(speechBubbles, reader)
        comic.append(text)
    return comic

def NER(comic, nlp, outputFilePath):
    full_text = '. '.join(comic)

    # Perform Named Entity Recognition
    doc = nlp(full_text)
    new_entities = {ent.text for ent in doc.ents if ent.label_ == 'PERSON'}
    result = [{'tagValue': entity, 'tagType': 'CHARACTER'} for entity in new_entities]

    # Update or create the output file with new entities
    if os.path.exists(outputFilePath):
        with open(outputFilePath, 'r') as json_file:
            existing_data = json.load(json_file)
        
        existing_entities = {item['tagValue'] for item in existing_data}
        all_entities = existing_entities.union(new_entities)
        result = [{'tagValue': entity, 'tagType': 'CHARACTER'} for entity in all_entities]

    with open(outputFilePath, 'w') as json_file:
        json.dump(result, json_file, indent=4)

    return result
