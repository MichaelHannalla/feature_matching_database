import cv2
import pytesseract

# Get the main tesseract library path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('capt.png')

h, w, c = img.shape
boxes = pytesseract.image_to_boxes(img) 
strings = pytesseract.image_to_string(img)

for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

#img = cv2.resize(img, (img.shape[1]//3, img.shape[0]//3))
#print(strings)
#exit()
cv2.imshow('img', img)
cv2.waitKey(0)