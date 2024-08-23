import streamlit as st
import numpy as np
import cv2
import fitz  # PyMuPDF
from io import BytesIO


def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def deskew_pdf(input_pdf_data):
    doc = fitz.open(stream=input_pdf_data, filetype="pdf")
    output = BytesIO()
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        deskewed_img = deskew_image(img)
        deskewed_pix = fitz.Pixmap(fitz.csRGB, fitz.Image(deskewed_img.tobytes(), pix.width, pix.height))
        page.insert_image(page.rect, pixmap=deskewed_pix)

    doc.save(output)
    output.seek(0)

    return output


def process_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        deskewed_pdf = deskew_pdf(uploaded_file.read())
        return deskewed_pdf, "application/pdf", "deskewed_output.pdf"
    else:
        # Handle image deskewing
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        deskewed_image = deskew_image(image)
        _, buffer = cv2.imencode('.jpg', deskewed_image)
        deskewed_image_data = BytesIO(buffer)
        deskewed_image_data.seek(0)
        return deskewed_image_data, "image/jpeg", "deskewed_output.jpg"


if __name__ == '__main__':
    # Streamlit UI
    st.title("File Deskewing App")
    st.write("Upload a PDF or image file, and this app will deskew the content.")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.write("Processing your file...")
        processed_file, mime_type, output_filename = process_file(uploaded_file)

        st.success("File deskewed successfully!")
        st.download_button(
            label="Download Deskewed File",
            data=processed_file,
            file_name=output_filename,
            mime=mime_type
        )