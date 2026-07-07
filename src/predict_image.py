"""
predict_image.py

Professional Version 2.0
CNN based Handwritten Digit Recognition
"""

import cv2
import time
import numpy as np
import os

from tensorflow.keras.models import load_model

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading CNN Model...")

model = load_model("models/cnn_mnist.keras")

print("Model Loaded Successfully!\n")


# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs("output", exist_ok=True)
os.makedirs("output/digits", exist_ok=True)


# ==========================================
# PREPROCESS DIGIT
# ==========================================

def preprocess_digit(thresh, x, y, w, h):
    """
    Convert cropped digit into MNIST format.
    """

    digit = thresh[y:y+h, x:x+w]

    # Make image square

    size = max(w, h)

    canvas = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    canvas[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = digit

    # Resize

    resized = cv2.resize(
        canvas,
        (20, 20)
    )

    final = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    final[
        4:24,
        4:24
    ] = resized

    final = final.astype("float32") / 255.0

    final = final.reshape(
        1,
        28,
        28,
        1
    )

    return final


# ==========================================
# MAIN FUNCTION
# ==========================================

def predict_uploaded_image():

    print("=" * 60)
    print("      HANDWRITTEN DIGIT RECOGNITION")
    print("=" * 60)

    image_path = input(
        "\nEnter Image Path : "
    )

    image = cv2.imread(image_path)

    if image is None:

        print("\nImage not found!")

        return
    
    start_time = time.time()

    print("\nImage Loaded Successfully!")

    output = image.copy()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ======================================
    # OTSU THRESHOLD
    # ======================================

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV +
        cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    cv2.imwrite(
        "output/threshold.png",
        thresh
    )

    # ======================================
    # FIND CONTOURS
    # ======================================

    contours, _ = cv2.findContours(

        thresh,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )

    digit_boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < 10 or h < 10:

            continue

        digit_boxes.append(

            (x, y, w, h)

        )

    digit_boxes = sorted(

        digit_boxes,

        key=lambda b: (

            b[1] // 80,

            b[0]

        )

    )

    print(
        f"\nDigits Detected : {len(digit_boxes)}"
    )

    predictions = []

    digit_index = 1

    # ======================================
    # PREDICTION LOOP
    # ======================================

    for (x, y, w, h) in digit_boxes:

        digit = preprocess_digit(

            thresh,

            x,

            y,

            w,

            h

        )

        digit_image = (

            digit.reshape(
                28,
                28
            ) * 255

        ).astype(np.uint8)

        cv2.imwrite(

            f"output/digits/digit_{digit_index}.png",

            digit_image

        )

        prediction = model.predict(

            digit,

            verbose=0

        )

        predicted_digit = int(

            np.argmax(

                prediction

            )

        )

        confidence = float(

            np.max(

                prediction

            )

        ) * 100

                # ======================================
        # STORE PREDICTION
        # ======================================

        predictions.append({

            "id": digit_index,

            "x": x,

            "y": y,

            "w": w,

            "h": h,

            "digit": predicted_digit,

            "confidence": confidence

        })

        # ======================================
        # DRAW PROFESSIONAL BOUNDING BOX
        # ======================================

        cv2.rectangle(

            output,

            (x, y),

            (x + w, y + h),

            (0, 200, 0),

            3

        )

        # ======================================
        # LABEL
        # ======================================

        label = f"{predicted_digit} ({confidence:.1f}%)"

        (text_width, text_height), _ = cv2.getTextSize(

            label,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            2

        )

        # Green filled rectangle

        cv2.rectangle(

            output,

            (x, y - 32),

            (x + text_width + 8, y),

            (0, 255, 0),

            -1

        )

        # Black Text

        cv2.putText(

            output,

            label,

            (x + 4, y - 8),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 0, 0),

            2

        )

        # ======================================
        # TERMINAL OUTPUT
        # ======================================

        print(

            f"[{digit_index}] "

            f"Digit : {predicted_digit}    "

            f"Confidence : {confidence:.2f}%"

        )

        digit_index += 1

    # ======================================
    # SAVE IMAGE WITH BOXES
    # ======================================

    cv2.imwrite(

        "output/prediction.png",

        output

    )

    print("\nPrediction image saved.")

    # ======================================
    # SORT PREDICTIONS
    # ======================================

    predictions = sorted(

        predictions,

        key=lambda p: (

            p["y"] // 80,

            p["x"]

        )

    )

    grouped_numbers = []

    current_group = []

    previous_y = None

    line_threshold = 60

    for prediction in predictions:

        if previous_y is None:

            current_group.append(

                prediction

            )

            previous_y = prediction["y"]

            continue

        if abs(

            prediction["y"] -

            previous_y

        ) < line_threshold:

            current_group.append(

                prediction

            )

        else:

            grouped_numbers.append(

                current_group

            )

            current_group = [

                prediction

            ]

            previous_y = prediction["y"]

    if current_group:

        grouped_numbers.append(

            current_group

        )

    # ======================================
    # SORT DIGITS INSIDE EACH LINE
    # ======================================

    for group in grouped_numbers:

        group.sort(

            key=lambda p: p["x"]

        )

    print("\n")

    print("=" * 60)

    print("             OCR RESULT")

    print("=" * 60)

        # ======================================
    # PRINT DETECTED NUMBERS
    # ======================================

    total_confidence = 0
    total_digits = 0

    detected_numbers = []

    for group in grouped_numbers:

        number = ""

        group_confidence = []

        for digit in group:

            number += str(digit["digit"])

            group_confidence.append(
                digit["confidence"]
            )

            total_confidence += digit["confidence"]

            total_digits += 1

        detected_numbers.append(number)

        print(f"\nDetected Number : {number}")

        print(
            f"Average Confidence : "
            f"{sum(group_confidence)/len(group_confidence):.2f}%"
        )

    average_confidence = (
        total_confidence / total_digits
        if total_digits > 0 else 0
    )

    end_time = time.time()

    processing_time = end_time - start_time

    print("\n" + "=" * 60)

    print("              STATISTICS")

    print("=" * 60)

    print(f"Total Contours        : {len(digit_boxes)}")

    print(f"Total Digits          : {total_digits}")

    print(f"Detected Numbers      : {len(detected_numbers)}")

    print(
        f"Average Confidence    : "
        f"{average_confidence:.2f}%"
    )

    print(
        f"Processing Time       : "
        f"{processing_time:.3f} seconds"
    )

    print("=" * 60)

    # ======================================
    # SAVE REPORT
    # ======================================

    report_path = "output/prediction_report.txt"

    with open(report_path, "w") as report:

        report.write("=" * 60 + "\n")
        report.write("HANDWRITTEN DIGIT RECOGNITION REPORT\n")
        report.write("=" * 60 + "\n\n")

        report.write(f"Image Path : {image_path}\n\n")

        report.write("Detected Numbers\n")
        report.write("-------------------------\n")

        for number in detected_numbers:

            report.write(number + "\n")

        report.write("\n")

        report.write(
            f"Total Contours : {len(digit_boxes)}\n"
        )

        report.write(
            f"Total Digits : {total_digits}\n"
        )

        report.write(
            f"Detected Numbers : {len(detected_numbers)}\n"
        )

        report.write(
            f"Average Confidence : "
            f"{average_confidence:.2f}%\n"
        )

        report.write(
            f"Processing Time : "
            f"{processing_time:.3f} seconds\n"
        )

    print("\nPrediction report saved!")

    print(
        f"\nReport Location : {report_path}"
    )

    print(
        "\nPrediction image saved to "
        "output/prediction.png"
    )

    print(
        "Threshold image saved to "
        "output/threshold.png"
    )

    print(
        "Individual digits saved to "
        "output/digits/"
    )

    print("\n" + "=" * 60)

    print("      PREDICTION COMPLETED SUCCESSFULLY")

    print("=" * 60)