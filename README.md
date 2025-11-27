# ToolBox

A small collection of utility scripts for document processing and image processing. This repository is intended to grow with additional tools for document cleanup, evaluation (boundary box F1), and Handwritten Text Recognition (HTR) related utilities.

## Current files

- `Image_border_cleaner.py` — tools to detect and remove or clean image borders. The main function provided is:
  - `find_edge(image, user_threshold=None, show_density=False, action="Crop")`
    - Description: Detects document region by scanning density from the image center toward all sides. It can either crop out the detected document region (`action="Crop"`) or paint the detected border areas with the document's minimum pixel value (usually black).
    - Arguments:
      - `image`: OpenCV BGR image (numpy array).
      - `user_threshold`: Optional numeric threshold to override the automatic threshold that detects edges.
      - `show_density`: If `True`, plots the row/column density graph using `matplotlib` to help tune thresholds.
      - `action`: `"Crop"` to remove borders; any other value paints the borders with the minimum pixel value.
    - Returns: the processed image (cropped or cleaned).
    - Quick example:
      ```python
      import cv2
      from Image_border_cleaner import find_edge

      img = cv2.imread('scanned_doc.jpg')
      out = find_edge(img, show_density=True, action='Crop')
      cv2.imwrite('scanned_doc_cropped.jpg', out)
      ```

- `f1_score_for_boundaryBoxes.py` — evaluation utilities for boundary box detection (filename indicates it computes F1 for boundary boxes). This script is intended to provide metrics (precision, recall, F1) for predicted vs ground-truth bounding boxes used in document layout and detection tasks. Import or run the script to compute F1 given predicted and reference boxes. (Open the file for the exact function names and usage.)

## Requirements

Install the main Python dependencies used by current tools:

```powershell
pip install opencv-python numpy matplotlib
```

If you use other parts of the toolbox later, additional dependencies may be required — check each script's imports.

## Notes & Tips

- The `find_edge` method converts the input to grayscale and analyzes the pixel intensity sums across rows/columns to detect strong changes. If results are too aggressive or too permissive, try adjusting `user_threshold` or enabling `show_density` to visualize the profile.
- For images with unusual lighting or heavy noise, consider pre-processing (deskew, histogram equalization, bilateral filter) before running border cleaning.

## Future Work

Planned additions to this repository (ToolBox):

- More document-processing utilities (deskew, binarization, layout analysis).
- Image-processing helpers (de-noising, contrast enhancement, morphological operations).
- HTR-related helpers and evaluation tools (alignment, CER/WER calculators, training data pipelines).

Contributions and suggestions are welcome — open an issue or submit a PR with new tools.

## License

This repository is licensed under the MIT License — see the `LICENSE` file for details.

