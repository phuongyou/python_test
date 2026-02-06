import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import cv2
import datetime
import numpy as np
from rapidfuzz import fuzz
import openpyxl

import os
import google.generativeai as genai
from config import GEMINI_API_KEY


# ================= CONFIG =================


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ========== GEMINI CONFIG ==========
# Lấy GEMINI_API_KEY từ file config.py
genai.configure(api_key=GEMINI_API_KEY)


INPUT_PDF = "palladium_fixed.pdf"
CONTRACT_TYPE = "contract"  # Hoặc 'sow', 'co', 'od' tuỳ loại
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "palladium_redacted_final.pdf")
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "redacted_fields.xlsx")


# TARGETS sẽ được cập nhật tự động từ AI
TARGETS = []

FUZZY_THRESHOLD = 94        # match phrase
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
    # Detect blue background and whiten it
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    # Define blue color range (tune if needed)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # Set blue regions to white
    img_cv[mask_blue > 0] = [255, 255, 255]
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
    import string
    def strip_punct(s):
        return s.rstrip(string.punctuation)

    for line in lines:
        words = line["words"]
        texts = [w[0] for w in words]

        for target in targets:
            target_words = target.split()
            t_len = len(target_words)

            for i in range(len(texts) - t_len + 1):
                # Chuẩn hóa: loại bỏ dấu câu cuối mỗi từ
                phrase = " ".join(strip_punct(t) for t in texts[i:i+t_len]).lower()
                target_norm = " ".join(strip_punct(t) for t in target_words).lower()
                score = fuzz.ratio(phrase, target_norm)

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

def extract_all_text_from_pdf(doc):
    """
    Trả về toàn bộ text của PDF (ưu tiên text layer, nếu không có thì OCR)
    """
    all_text = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            all_text.append(text)
        else:
            # OCR nếu không có text layer
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes()))
            clean_img = preprocess_image_for_ocr(img)
            ocr_txt = pytesseract.image_to_string(clean_img)
            all_text.append(ocr_txt)
    return "\n".join(all_text)


def get_redact_fields_from_gemini(text, contract_type="contract"):
    """
    Gửi text lên Gemini để lấy các field cần redact theo loại hợp đồng, chỉ lấy đúng các field mẫu cho từng loại.
    """
    FIELD_TEMPLATES = {
        "contract": '''{
  "contract_code": "Contract number/code",
  "contract_name": "Contract name/title",
  "contract_type": "NDA|MSA|Subcontract|Partnership|VendorAgreement (choose the most appropriate one, default VendorAgreement)",
  "contract_summary": "Brief summary of contract content",
  "description": "Detailed description of the contract",
  "status": "Active|Expired|Pending|Terminated|Incomplete (default Incomplete if unclear)",
   "client_name": "Client name or organization associated with the contract",
  "contract_author_id": "Author or drafter of the contract if mentioned",
  "contract_manager_id": "Contract owner or manager if mentioned",
  "start_date": "Start date (YYYY-MM-DD)",
  "end_date": "End date (YYYY-MM-DD)",
  "effective_date": "Effective date (YYYY-MM-DD)",
  "execution_date": "Execution/signing date (YYYY-MM-DD)",
  "date_signed": "Date when contract was signed (YYYY-MM-DD)",
  "expiration_date": "Expiration date (YYYY-MM-DD)",
  "duration_months": "Contract duration in months if explicitly stated",
  "coi_required": "true/false (is certificate of insurance required)",
  "coi_expiry_date": "Insurance expiry date (YYYY-MM-DD)",
  "insurance_requirements": "Insurance requirements description",
  "renewal_terms": "Renewal terms description",
  "auto_renew": "true/false (automatic renewal)",
  "renewal_term_months": "Renewal term in months (number)",
  "term_frequency_id": "Renewal frequency reference if specified",
  "governing_law": "Governing law",
  "jurisdiction": "Jurisdiction",
  "dispute_resolution": "Dispute resolution method",
  "termination_terms": "Termination terms",
  "termination_notice_days": "Notice period before termination in days (number)",
  "currency_code": "Currency used in the contract (ISO code like USD, VND, EUR)",
  "work_type": "Type of work (only Partnership, Agreement)",
  "product_service_id": "Referenced product or service if mentioned",
  "signed_copy": "true/false (is a signed copy available)",
  "docusign": "true/false (signed via DocuSign or similar electronic signature platform)",
  "alert_frequency_id": "Alert or reminder frequency if defined",
  "notes": "Other notes or additional extracted information",
  "client_address": "Client address associated with the contract",
  "signed_name": "Name of the person who signed the contract",
  "signed_title": "Title of the person who signed the contract",
  "customer_name": "Client / Customer name associated with the contract",
  "customer_address": "Client / Customer address associated with the contract",
  "director_name": "Director or authorized signatory name",
  "client_authorized_name": "Name of client authorized signatory",
  "supplier_name": "Supplier or vendor name associated with the contract",
  "supplier_authorized_name": "Name of supplier authorized signatory",
  "supplier_address": "Supplier or vendor address associated with the contract",

  "payment_terms": "Payment terms specified in contract",
  "invoice_frequency": "Invoice frequency (e.g. monthly, quarterly)",
  "audit_rights": "Audit rights description",
  "audit_retention": "Audit retention period or terms",
  "coi_renewal": "Certificate of insurance renewal terms",
  "general_liability": "General liability insurance details",
  "errors_omissions": "Errors & omissions insurance details",
  "cyber_liability": "Cyber liability insurance details",
  "additional_insured": "true/false (is additional insured required)",
  "waiver_of_subrogation": "true/false (is waiver of subrogation required)",
  "tail_coverage": "Tail coverage insurance details",
  "agreement": "Type of agreement, corresponds to contract_type"
  
}''',
        "sow": '''{
  "sow_name": "SOW name/title",
  "sow_type": "SOW|CO|OD (Statement of Work, Change Order, or Order Document - for SOW always use 'SOW')",
  "status": "Active|Completed|Expired (default Active if unclear)",
  "line_of_business": "Business line or division (e.g., IT Services, Consulting, Software Development)",
  "work_type": "Professional Services|Managed Services (type of work being performed)",
  "start_date": "SOW start date (YYYY-MM-DD)",
  "end_date": "SOW end date (YYYY-MM-DD)",
  "effective_date": "Effective date when SOW becomes active (YYYY-MM-DD)",
  "date_signed": "Date when SOW was signed (YYYY-MM-DD)",
  "po_required": "true/false (is purchase order required for this SOW)",
  "po_number": "Purchase order number if po_required is true",
  "budget": "Budget amount for this SOW (number)",
  "renewal_terms": "Renewal terms description or frequency",
  "alert_status": "Alert days (comma-separated: 1,14,35,45,50,60,70,90,120)",
  "signed_copy": "true/false (is a signed copy available)",
  "docusign": "true/false (signed via DocuSign or similar electronic signature platform)",
  "terms": "SOW terms, conditions, scope of work, deliverables, or other details",
  "client_name": "Client / Customer name associated with the SOW",
  "client_email": "Primary client email for the SOW",
  "client_phone": "Primary client phone number for the SOW",
  "application_name": "Extract the core application/system name only. If the project name contains words 
    like 'Application', 'System', 'Platform', 'Services', 'Support', or 'Remote Services', exclude them. 
    Prefer the main product name appearing before such keywords.",
  "contact_title": "Primary contact's job title for the SOW",
  "contact_name": "Primary contact's full name for the SOW",
  "contact_email": "Primary contact's email address for the SOW",
  "contact_phone": "Primary contact's phone number for the SOW",
  "consultant_name": "Consultant or service provider name associated with the SOW",
  "consultant_email": "Primary consultant email for the SOW",
  "consultant_phone": "Primary consultant phone number for the SOW",
  "supplier_name": "Supplier or vendor name associated with the SOW",
  "supplier_email": "Primary supplier email for the SOW",
  "supplier_phone": "Primary supplier phone number for the SOW",
  
  "project": "Project name associated with the SOW",
  "supplier_tech_contact_name": "Supplier technical contact's full name for the SOW",
  "supplier_tech_contact_email": "Supplier technical contact's email address for the SOW",
  "supplier_tech_contact_phone": "Supplier technical contact's phone number for the SOW",

  "client_tech_contact_name": "Client technical contact's full name for the SOW",
  "client_tech_contact_email": "Client technical contact's email address for the SOW",
  "client_tech_contact_phone": "Client technical contact's phone number for the SOW",

  "accepted_client_name": "Name of client representative who accepted the SOW",
  "accepted_client_title": "Title of client representative who accepted the SOW",
  "accepted_client_signed_date": "Date when client accepted and signed the SOW (YYYY-MM-DD)",

  "accepted_supplier_name": "Name of supplier representative who accepted the SOW",
  "accepted_supplier_title": "Title of supplier representative who accepted the SOW",
  "accepted_supplier_signed_date": "Date when supplier accepted and signed the SOW (YYYY-MM-DD)",

  "term_start_date": "Start date of the term (YYYY-MM-DD)",
  "term_duration": "Duration of the term (number, e.g., months)",
  "term_end_date": "End date of the term (YYYY-MM-DD)",
  "doc_id": "Document identifier or reference number for the SOW",
  "consultant_fee_schedule_Investment": "Fee schedule related to investment consulting services, including rates, pricing, or payment structure defined in the SOW",
  "retainer_summary_annual_retainer_fee": "Total annual retainer fee amount stated in the contract for ongoing consulting or support services",
  "services_fee_schedule_investment": "Detailed fee schedule for investment-related services, including service types, rates, and billing terms",
  "monthly_support_schedule_term_length": "Duration or length of the monthly support agreement, expressed in months or years",
  "monthly_support_schedule_term_end": "End date of the monthly support term (YYYY-MM-DD)",
  "monthly_support_schedule_hours_per_month": "Number of support or consulting hours provided per month under the support agreement",
  "monthly_support_schedule_discount": "Discount applied to the monthly support fees, as a percentage or fixed amount",
  "monthly_support_schedule_monthly_rate": "Monthly fee or rate charged for support services under the monthly support schedule"

  

}''',
        "co": '''{
  "sow_name": "CO name/title (Change Order title)",
  "sow_type": "CO (always use 'CO' for Change Orders)",
  "status": "Active|Completed|Expired (default Active if unclear)",
  "change_type": "Scope Change|Budget Adjustment|Schedule Change|Resource Change|Other (type of change being requested)",
  "approval_status": "Pending|Approved|Rejected (approval status of the change order)",
  "change_order_date": "Date of the change order (YYYY-MM-DD)",
  "co_effective_date": "Effective date when change order becomes active (YYYY-MM-DD)",
  "cost_impact": "Cost impact in dollars (number, e.g., 5000.00 for $5,000 increase)",
  "effort_impact_hours": "Effort impact in hours (number, e.g., 120.5 for 120.5 hours)",
  "revised_scope": "Description of the revised scope of work after the change",
  "schedule_impact": "Description of schedule impact (e.g., 'Delayed by 2 weeks', 'No schedule impact')",
  "billing_impact": "Description of billing impact (e.g., 'Additional $10K in next invoice', 'Time & Materials basis')",
  "co_budget": "true/false (does this change impact the budget)",
  "co_schedule": "true/false (does this change impact the schedule)",
  "co_scope": "true/false (does this change impact the scope)",
  "signed_change_order": "true/false (is the change order signed)",
  "line_of_business": "Business line or division (e.g., IT Services, Consulting)",
  "work_type": "Professional Services|Managed Services",
  "start_date": "Change order start date (YYYY-MM-DD)",
  "end_date": "Change order end date (YYYY-MM-DD)",
  "po_required": "true/false (is purchase order required for this change)",
  "po_number": "Purchase order number if po_required is true",
  "budget": "Updated budget amount after change (number)",
  "renewal_terms": "Renewal terms if applicable",
  "alert_status": "Alert days (comma-separated: 1,14,35,45,50,60,70,90,120)",
  "signed_copy": "true/false (is a signed copy available)",
  "docusign": "true/false (signed via DocuSign)",
  "terms": "Change order terms, conditions, justification, or other details",
  "client_name": "Client / Customer name associated with the change order",
  "supplier_name": "Supplier or vendor name associated with the change order",
  "sow_title": "Title of the original SOW associated with this change order",
  "change_number": "Unique identifier or number for this change order"
}''',
        "od": '''{
  "sow_name": "OD name/title (Order Document title)",
  "sow_type": "OD (always use 'OD' for Order Documents)",
  "status": "Active|Completed|Expired (default Active if unclear)",
  "start_date": "OD start date (YYYY-MM-DD)",
  "end_date": "OD end date (YYYY-MM-DD)",
  "od_execution_date": "Execution date when OD was executed (YYYY-MM-DD)",
  "od_support_term": "Support term in months (number, e.g., 12 for 12 months)",
  "provisioning_date": "Provisioning date (YYYY-MM-DD)",
  "ship_date": "Product ship date (YYYY-MM-DD)",
  "support_end_date": "Support end date (YYYY-MM-DD)",
  "od_support_provider": "Support provider vendor ID or name (vendor providing support services)",
  "support_provider": "Support provider name or ID",
  "support_term_months": "Support term length in months",
  "line_of_business": "Business line or division (e.g., IT Services, Infrastructure Support)",
  "work_type": "Professional Services|Managed Services",
  "order_amount": "Total order amount",
  "budget": "Budget amount for this order document (number)",
  "vendor_amount": "Vendor total amount",
  "fixed_bid_services_amount": "Fixed bid services amount",
  "travel_expenses_amount": "Travel expenses amount",
  "freight": "Freight or shipping cost",
  "oem_list_price": "OEM list price",
  "vendor_discount_percent": "Vendor discount percentage",
  "customer_discount_percent": "Customer discount percentage",
  "estimated_gross_margin_percent": "Estimated gross margin percentage",
  "po_required": "true/false (is purchase order required for this order)",
  "po_number": "Purchase order number if po_required is true",
  "customer_po_number": "Customer purchase order number",
  "vendor_quote": "Vendor quote reference or identifier",
  "oem": "OEM name",
  "oem_opportunity_id": "OEM opportunity or reference ID",
  "authorized_signer_name": "Authorized signer full name",
  "authorized_signer_email": "Authorized signer email address",
  "product_service_id": "Product or service ID linked to this order document",
  "renewal_terms": "Renewal terms or frequency (e.g., Annual, Quarterly)",
  "alert_status": "Alert days (comma-separated: 1,14,35,45,50,60,70,90,120)",
  "signed_copy": "true/false (is a signed copy available)",
  "docusign": "true/false (signed via DocuSign or electronic signature)",
  "terms": "Order document terms, conditions, scope, deliverables, SLA, or other details"
}'''
    }
    contract_type_key = contract_type.lower()
    if contract_type_key not in FIELD_TEMPLATES:
        contract_type_key = "contract"
    field_template = FIELD_TEMPLATES[contract_type_key]
    prompt = f"""
You are an AI that extracts sensitive information for redaction. Below is the content of a PDF file of type '{contract_type}'. Only extract fields exactly as in the following JSON template. For each field, return the field name and the actual value found in the document (leave value empty if not found). Return the result as a JSON list: [{{'field': ..., 'value': ...}}, ...].

FIELD TEMPLATE FOR TYPE '{contract_type}':
{field_template}

PDF CONTENT:
{text}
"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    import json
    try:
        # Tìm đoạn JSON trong response
        import re
        match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if match:
            fields = json.loads(match.group(0))
        else:
            fields = json.loads(response.text)
        return fields
    except Exception as e:
        print("Gemini response parse error:", e)
        print("Raw response:", response.text)
        return []

def save_fields_to_excel(fields, xlsx_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["field", "value"])
    for item in fields:
        ws.append([item.get("field", ""), item.get("value", "")])
    wb.save(xlsx_path)



def main():
    doc = fitz.open(INPUT_PDF)
    has_text = pdf_has_text_layer(doc)
    print(f"📄 PDF has text layer: {has_text}")

    # 1. Trích xuất toàn bộ text
    all_text = extract_all_text_from_pdf(doc)


    # 2. Gửi lên Gemini để lấy các field cần redact
    fields = get_redact_fields_from_gemini(all_text, contract_type=CONTRACT_TYPE)
    # print("Fields from Gemini:", fields)

    # 2.1. Filter only redact fields by type
    REDACT_FIELDS = {
        "contract": [
            "agreement", "client_name", "client_authorized_name", "client_address", "supplier_name","supplier_address",
            "supplier_authorized_name","doc_id","effective_date","sign_date","term_start_date","term_duration",
            "term_end_date","msa_renewals","payment_terms","invoice_frequency","audit_rights","audit_retention",
            "coi_required","coi_renewal","errors_omissions","general_liability","cyber_liability",
            "additional_insured","waiver_of_subrogation","tail_coverage"
        ],
        "sow": [
            "project","client_name","supplier_name",
            "supplier_tech_contact_name","supplier_tech_contact_email","supplier_tech_contact_phone",
            "supplier_tech_contact_name","supplier_tech_contact_email","supplier_tech_contact_phone",
            "accepted_client_name","accepted_client_title","application_name",
            "accepted_client_signed_date", "accepted_supplier_name" ,"accepted_supplier_title","accepted_supplier_signed_date",
            "doc_id","effective_date","term_start_date", "term_duration", "term_end_date",
            "consultant_fee_schedule_Investment",
            "retainer_summary_annual_retainer_fee","services_fee_schedule_investment",
            "monthly_support_schedule_term_length","monthly_support_schedule_term_end",
            "monthly_support_schedule_hours_per_month","monthly_support_schedule_discount",
            "monthly_support_schedule_monthly_rate"
            # "sow_name", "client_name", "client_email", "client_phone", "application_name", "contact_title", "contact_name", "contact_email", "contact_phone", "consultant_name", "consultant_phone", "consultant_email", "supplier_name", "supplier_email", "supplier_phone"
        ],
        "co": [
            "sow_name", "client_name", "supplier_name", "sow_title", "change_number", "change_order_date"
        ]
    }
    type_key = CONTRACT_TYPE.lower()
    field_dict = {item.get("field", ""): item.get("value", "") for item in fields}
    if type_key == "contract":
        CONTRACT_ALL_FIELDS = [
            "agreement", "client_name", "client_authorized_name", "client_address", "supplier_name","supplier_address",
            "supplier_authorized_name","doc_id","effective_date","sign_date","term_start_date","term_duration",
            "term_end_date","msa_renewals","payment_terms","invoice_frequency","audit_rights","audit_retention",
            "coi_required","coi_renewal","errors_omissions","general_liability","cyber_liability",
            "additional_insured","waiver_of_subrogation","tail_coverage"
        ]
        redact_keys = [
            "client_name", "client_authorized_name", "client_address", "supplier_name","supplier_address", "supplier_authorized_name"
        ]
        filtered_fields = [
            {"field": k, "value": field_dict.get(k, "")}
            for k in CONTRACT_ALL_FIELDS if k != "application_name"
        ]
    elif type_key == "sow":
        SOW_ALL_FIELDS = [
            "project","client_name","supplier_name",
            "supplier_tech_contact_name","supplier_tech_contact_email","supplier_tech_contact_phone",
            "supplier_tech_contact_name","supplier_tech_contact_email","supplier_tech_contact_phone",
            "accepted_client_name","accepted_client_title","application_name",
            "accepted_client_signed_date", "accepted_supplier_name" ,"accepted_supplier_title","accepted_supplier_signed_date",
            "doc_id","effective_date","term_start_date", "term_duration", "term_end_date",
            "consultant_fee_schedule_Investment",
            "retainer_summary_annual_retainer_fee","services_fee_schedule_investment",
            "monthly_support_schedule_term_length","monthly_support_schedule_term_end",
            "monthly_support_schedule_hours_per_month","monthly_support_schedule_discount",
            "monthly_support_schedule_monthly_rate"
        ]
        SOW_REDACT_FIELDS = [
            "project","client_name","supplier_name",
            "supplier_tech_contact_name","supplier_tech_contact_email","supplier_tech_contact_phone",
            "supplier_tech_contact_name","supplier_tech_contact_email","supplier_tech_contact_phone",
            "accepted_client_name","accepted_client_title","application_name",
            "accepted_client_signed_date", "accepted_supplier_name" ,"accepted_supplier_title","accepted_supplier_signed_date"
        ]
        redact_keys = SOW_REDACT_FIELDS
        filtered_fields = [
            {"field": k, "value": field_dict.get(k, "")}
            for k in SOW_ALL_FIELDS if k != "application_name"
        ]
    else:
        redact_keys = REDACT_FIELDS.get(type_key, [])
        filtered_fields = [
            {"field": k, "value": field_dict.get(k, "")}
            for k in redact_keys if k != "application_name"
        ]
    save_fields_to_excel(filtered_fields, OUTPUT_XLSX)

    # 4. Chuẩn bị TARGETS (có cả application_name nếu có value)
    global TARGETS
    base_targets = []
    for k in redact_keys:
        value = str(field_dict.get(k, "")).lower()
        if value:
            if "address" in k:
                # Nếu là trường address, chia nhỏ từng từ để redact riêng
                base_targets.extend(value.split())
            else:
                base_targets.append(value)
    # Nếu từ dài hơn 2 từ, tách thành các cụm 2 từ liên tiếp
    def split_bigrams(text):
        words = text.split()
        if len(words) <= 2:
            return []
        first = " ".join(words[:2])
        rest = " ".join(words[2:]) if len(words) > 2 else ""
        result = [first]
        if rest:
            result.append(rest)
        return result
    bigram_targets = []
    for t in base_targets:
        bigram_targets.extend(split_bigrams(t))
    TARGETS = base_targets + bigram_targets

    # 5. Redact PDF như cũ
    for page_index, page in enumerate(doc):
        print(f"🔍 Processing page {page_index + 1}")
        page_rect = page.rect

        if has_text:
            words = page.get_text("words")  # x0,y0,x1,y1,text
            texts = [w[4] for w in words]

            import string
            def strip_punct(s):
                return s.rstrip(string.punctuation)

            for target in TARGETS:
                target_words = target.split()
                t_len = len(target_words)

                for i in range(len(words) - t_len + 1):
                    phrase = " ".join(strip_punct(w[4]) for w in words[i:i+t_len]).lower()
                    target_norm = " ".join(strip_punct(t) for t in target_words).lower()
                    if fuzz.ratio(phrase, target_norm) >= FUZZY_THRESHOLD:
                        x0 = min(w[0] for w in words[i:i+t_len])
                        y0 = min(w[1] for w in words[i:i+t_len])
                        x1 = max(w[2] for w in words[i:i+t_len])
                        y1 = max(w[3] for w in words[i:i+t_len])

                        page.add_redact_annot(
                            fitz.Rect(x0, y0, x1, y1),
                            fill=(0, 0, 0)
                        )

        else:
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
    print(f"✅ DONE – Redacted PDF and Excel saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python nhien_ocr_redact_pdf.py <contract_type> <input_pdf>")
        print("Ví dụ: python nhien_ocr_redact_pdf.py contract myfile.pdf")
        sys.exit(1)
    CONTRACT_TYPE = sys.argv[1]
    INPUT_PDF = sys.argv[2]
    # Tạo thư mục con theo kiểu type_ID (type + timestamp)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(OUTPUT_DIR, f"{CONTRACT_TYPE}_{run_id}")
    os.makedirs(run_folder, exist_ok=True)
    OUTPUT_PDF = os.path.join(run_folder, f"{os.path.splitext(os.path.basename(INPUT_PDF))[0]}_redacted_final.pdf")
    OUTPUT_XLSX = os.path.join(run_folder, f"{os.path.splitext(os.path.basename(INPUT_PDF))[0]}_redacted_fields.xlsx")
    main()
