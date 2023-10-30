# from paddleocr import PaddleOCR
# import cv2

# # Initialize the PaddleOCR instance
# ocr = PaddleOCR()

# # Load the image you want to process
# image_path = '/home/premal/Downloads/massachusetts-drivers-license.jpg'

# # Perform text detection and recognition
# result = ocr.ocr(image_path)

# # Load the image to draw bounding boxes on
# image = cv2.imread(image_path)

# # Iterate through detected text regions and draw bounding boxes
# for res in result[0]:
#     if isinstance(res, list):
#         text = res[1][0]
#         confidence = res[1][1]
#         box = res[0]
#     else:
#         continue  # Skip non-list results

#     if isinstance(box, list):
#         box = box[0]  # Take the first element if it's a list

#     if isinstance(box, float):
#         continue  # Skip non-iterable float objects

#     # Convert the box coordinates to integers
#     x, y = int(box[0]), int(box[1])
#     cv2.rectangle(image, (x, y), (x + 100, y + 30), (0, 0, 255), 2)
#     cv2.putText(image, f'{text} ({confidence:.2f})', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#     # Print the recognized text to the console
#     print(f"Text: {text}")
# # Save the image with bounding boxes
# output_image_path = '/home/premal/Desktop/Python/ImageToText/image_with_boxes/image_with_boxes.jpg'
# cv2.imwrite(output_image_path, image)

# print(f"Image with bounding boxes saved at {output_image_path}")




from paddleocr import PaddleOCR
import cv2,os

# Initialize the PaddleOCR instance
ocr = PaddleOCR(lang='ml',det_pse_min_area=50,rec_algorithm='CRNN',rec_max_length=128,det_algorithm='DB',use_gpu=False,use_xpu=False,use_npu=False,ir_optim=True,use_tensorrt=False,  min_subgraph_size=15, precision='fp64',det_limit_side_len=960, det_limit_type='max', det_box_type='quad')


# use_gpu=False,          # Use GPU for processing (False)
#     use_xpu=False,          # Use XPU (Unknown accelerator) for processing (False)
#     use_npu=False,          # Use NPU (Neural Processing Unit) for processing (False)
#     ir_optim=True,          # Enable IR optimization (True)
#     use_tensorrt=False,     # Use TensorRT for optimization (False)
#     min_subgraph_size=15,   # Minimum subgraph size for optimization (15)
#     precision='fp32', 

# ocr = PaddleOCR(det_algorithm='PSENet')
# ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, det_model_dir="./PaddleOCR-release-2.3/inference/det_sast_tt/")


# ocr = PaddleOCR()
# ocr = PaddleOCR(rec_algorithm='SVTR_LCNet',lang='en')
# Load the image you want to process
image_path = '/home/premal/Downloads/us_license_img.jpeg'

# image_path = '/home/premal/Downloads/massachusetts-drivers-license.jpg'
#
# Set a confidence threshold (adjust this value as needed)
confidence_threshold = 0.7
# det_algorithm = 'EAST'

  # Example: filter regions with confidence >= 0.8
# Extract the original image name (without path)
image_name = os.path.basename(image_path)
# Perform text detection and recognition
result = ocr.ocr(image_path)

# Load the image to draw bounding boxes on
image = cv2.imread(image_path)

# Iterate through detected text regions and draw bounding boxes
for res in result[0]:
    if isinstance(res, list):
        text = res[1][0]
        confidence = res[1][1]
        box = res[0]
    else:
        continue  # Skip non-list results

    if isinstance(box, list):
        box = box[0]  # Take the first element if it's a list

    if isinstance(box, float):
        continue  # Skip non-iterable float objects

    # Check the confidence of the recognized text
    if confidence >= confidence_threshold:
        # Convert the box coordinates to integers
        x, y = int(box[0]), int(box[1])
        cv2.rectangle(image, (x, y), (x + 100, y + 30), (0, 0, 255), 2)
        cv2.putText(image, f'{text} ({confidence:.2f})', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Print the recognized text to the console
        # print(f"Text: {text} | Confidence: {confidence:.2f}")
        print(f"Text: {text}")

# Save the image with bounding boxes
# output_image_path = '/home/premal/Desktop/Python/ImageToText/image_with_boxes/1.jpg'
output_image_path = os.path.join('/home/premal/Desktop/Python/ImageToText/image_with_boxes/', f'Image_with_box_{image_name}')
cv2.imwrite(output_image_path, image)

print(f"Image with bounding boxes saved at {output_image_path}")





# /home/premal/Downloads/massachusetts-drivers-license.jpg