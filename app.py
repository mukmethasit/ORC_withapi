import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import cv2
import numpy as np

st.set_page_config(page_title="Gemini Smart OCR", layout="wide", page_icon="📄")

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def auto_scan_image(pil_image):
    try:
        opencv_image = np.array(pil_image)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        ratio = opencv_image.shape[0] / 500.0
        orig = opencv_image.copy()
        image = cv2.resize(opencv_image, (int(opencv_image.shape[1] / ratio), 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break
        if screenCnt is not None:
            warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            return Image.fromarray(warped_rgb), True
        else:
            return pil_image, False
    except Exception:
        return pil_image, False

def enhance_image(pil_image):
    try:
        opencv_image = np.array(pil_image)
        if len(opencv_image.shape) == 3:
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = opencv_image
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        result_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(result_rgb)
    except Exception:
        return pil_image

st.title("📄 Gemini Smart OCR")
st.markdown("ระบบแปลงภาพเป็นข้อความอัจฉริยะ ด้วย Gemini ")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.text_input("Google API Key", type="password")
    st.divider()
    crop_mode = st.selectbox("Crop Mode", ["None", "Auto Scan"], index=0)
    enable_enhance = st.checkbox("Enhance (B&W)", value=False)
    st.divider()
    output_format = st.selectbox("Output Format", ["Text", "JSON", "CSV (Table)"], index=0)
    custom_topic = st.text_input("Focus Topic", placeholder="Optional")

if not api_key:
    st.warning("Please enter API Key.")
    st.stop()

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    st.error(f"API Error: {e}")
    st.stop()

uploaded_files = st.file_uploader("Upload Images", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)
run_btn = st.button("Run OCR", type="primary", use_container_width=True)

if run_btn and uploaded_files:
    base_prompt = "Act as an expert OCR system. Analyze the image and extract text."
    format_instruction = ""
    if output_format == "JSON":
        format_instruction = "Output Requirement: Return the result strictly in JSON format. Identify keys based on the document labels. Do not use Markdown code blocks, just raw JSON text."
    elif output_format == "CSV (Table)":
        format_instruction = "Output Requirement: If the image contains a table, extract it as CSV format. Use comma (,) as separator. Return only the CSV data."
    else: 
        format_instruction = "Output Requirement: Return plain text, preserving the layout using Markdown spacing."

    final_prompt = f"{base_prompt}\n{format_instruction}\n"
    if custom_topic:
        final_prompt += f"\nIMPORTANT FOCUS: {custom_topic}"

    for uploaded_file in uploaded_files:
        try:
            image_bytes = uploaded_file.getvalue()
            img = Image.open(io.BytesIO(image_bytes))
            st.divider()
            st.subheader(f"File: {uploaded_file.name}")
            col1, col2 = st.columns([1, 1])
            
            processed_img = img
            if crop_mode == "Auto Scan":
                processed_img, success = auto_scan_image(processed_img)
            if enable_enhance:
                processed_img = enhance_image(processed_img)

            with col1:
                st.image(processed_img, caption="Processed Image", use_container_width=True)

            with col2:
                with st.spinner('Processing...'):
                    response = model.generate_content([final_prompt, processed_img])
                    result_text = response.text
                    st.success("Done!")
                    if output_format == "JSON":
                        st.json(result_text)
                    elif output_format == "CSV (Table)":
                        st.code(result_text, language='csv')
                    else:
                        st.markdown(result_text)
                    st.download_button(label="Download Result", data=result_text, file_name=f"ocr_result_{uploaded_file.name}.txt", mime="text/plain")

        except Exception as e:
            st.error(f"Error: {e}")

if not uploaded_files and run_btn:
    st.warning("Please upload images first.")