from resistor_detection_new_logic.py import load_and_detect_resistors, crop_resistor, preprocess_image, compute_vertical_medians

def main(image_path):
    img, resistors = load_and_detect_resistors(image_path)
    if img is None or resistors is None:
        print("No resistors detected.")
        return

    for x, y, w, h in resistors:
        cropped_img = crop_resistor(img, x, y, w, h)
        preprocessed_img = preprocess_image(cropped_img)
        median_img = compute_vertical_medians(preprocessed_img)
        bands = findBands(median_img, DEBUG=True)
        print("Detected bands:", bands)

        # Check the structure of color_code_positions, excluding 'last_pos'
        for key, value in bands.items():
            if key == 'last_pos':
                continue
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                print(f"Error: Value for {key} is not a list or tuple with at least two elements.")
                return

        printResult(bands, img, (x, y, w, h), DEBUG=True)
        display_images([cropped_img, preprocessed_img, median_img], ['Cropped', 'Preprocessed', 'Median'])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main('pic4.jpg')