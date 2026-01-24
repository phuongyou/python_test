import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np
from rapidfuzz import fuzz

# ================= CONFIG =================

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INPUT_PDF = "palladium_fixed.pdf"
OUTPUT_PDF = "palladium_redacted_final.pdf"

TARGETS = [
    "Company B",
    "Company A",
    "123 North Street, 14th Floor",
    "Chicago, IL 60606",
]

TARGETS = [t.lower() for t in TARGETS]

FUZZY_THRESHOLD = 90        # match phrase
OCR_THRESHOLD = 200         # xoá highlight
LINE_Y_THRESHOLD = 12       # group words theo dòng

# ================= UTILS =================

def pdf_has_text_layer(doc, min_chars=50):
    for page in doc:
        if len(page.get_text().strip()) > min_chars:
            return True
    return False


def preprocess_image_for_ocr(pil_img):
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, OCR_THRESHOLD, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh)


def ocr_image(img):
    return pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT,
        config="--psm 6"
    )


def group_words_into_lines(data):
    """
    Group OCR words theo dòng (y gần nhau)
    """
    lines = []

    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue

        box = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i]
        )

        placed = False
        for line in lines:
            if abs(line["y"] - box[1]) < LINE_Y_THRESHOLD:
                line["words"].append((word, box))
                placed = True
                break

        if not placed:
            lines.append({
                "y": box[1],
                "words": [(word, box)]
            })

    # sort words trong dòng theo x
    for line in lines:
        line["words"].sort(key=lambda w: w[1][0])

    return lines


def redact_targets_from_lines(page, lines, targets, scale_x, scale_y):
    """
    ❗ CHỈ redact khi match ĐỦ PHRASE
    """
    for line in lines:
        words = line["words"]
        texts = [w[0] for w in words]

        for target in targets:
            target_words = target.split()
            t_len = len(target_words)

            for i in range(len(texts) - t_len + 1):
                phrase = " ".join(texts[i:i+t_len]).lower()
                score = fuzz.ratio(phrase, target)

                if score >= FUZZY_THRESHOLD:
                    boxes = [words[j][1] for j in range(i, i+t_len)]

                    x0 = min(b[0] for b in boxes)
                    y0 = min(b[1] for b in boxes)
                    x1 = max(b[0] + b[2] for b in boxes)
                    y1 = max(b[1] + b[3] for b in boxes)

                    rect = fitz.Rect(
                        x0 * scale_x,
                        y0 * scale_y,
                        x1 * scale_x,
                        y1 * scale_y
                    )

                    page.add_redact_annot(rect, fill=(0, 0, 0))


# ================= MAIN =================

doc = fitz.open(INPUT_PDF)
has_text = pdf_has_text_layer(doc)

print(f"📄 PDF has text layer: {has_text}")

for page_index, page in enumerate(doc):
    print(f"🔍 Processing page {page_index + 1}")
    page_rect = page.rect

    if has_text:
        # ===== TEXT MODE (NO OCR) =====
        words = page.get_text("words")  # x0,y0,x1,y1,text
        texts = [w[4] for w in words]

        joined = " ".join(texts).lower()

        for target in TARGETS:
            target_words = target.split()
            t_len = len(target_words)

            for i in range(len(words) - t_len + 1):
                phrase = " ".join(w[4] for w in words[i:i+t_len]).lower()
                if fuzz.ratio(phrase, target) >= FUZZY_THRESHOLD:
                    x0 = min(w[0] for w in words[i:i+t_len])
                    y0 = min(w[1] for w in words[i:i+t_len])
                    x1 = max(w[2] for w in words[i:i+t_len])
                    y1 = max(w[3] for w in words[i:i+t_len])

                    page.add_redact_annot(
                        fitz.Rect(x0, y0, x1, y1),
                        fill=(0, 0, 0)
                    )

    else:
        # ===== OCR MODE =====
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes()))

        scale_x = page_rect.width / img.width
        scale_y = page_rect.height / img.height

        clean_img = preprocess_image_for_ocr(img)
        data = ocr_image(clean_img)

        lines = group_words_into_lines(data)

        redact_targets_from_lines(
            page,
            lines,
            TARGETS,
            scale_x,
            scale_y
        )

    page.apply_redactions()

doc.save(OUTPUT_PDF)
doc.close()

print("✅ DONE – Phrase-level redact, no single-character bug")
