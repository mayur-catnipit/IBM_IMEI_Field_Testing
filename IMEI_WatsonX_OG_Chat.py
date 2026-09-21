# """
# Extract the IMEI number(s) from scanned mobile-shop invoices using
# watsonx.ai Llama 4 Maverick (to READ the value) + Tesseract OCR
# (to LOCATE the value's pixels) + Pillow (to CROP).

# Why this can be trusted:
#   An IMEI is 15 digits and its LAST digit is a Luhn check digit over the
#   first 14. So unlike a plain invoice number or a date, an IMEI can be
#   MATHEMATICALLY VERIFIED. Any single-digit misread almost always breaks
#   the checksum, so we can detect it instead of returning a wrong value.

# Accuracy strategy (highest-confidence first):
#   1. Maverick returns STRUCTURED JSON: imei1 / imei2 (dual-SIM phones
#      print two) plus the raw text, with explicit digit-confusion warnings.
#   2. We read the image up to N_READS times and take the majority vote per
#      IMEI (consensus). Temperature 0 keeps each read deterministic; the
#      repeat is a guard against occasional flakiness.
#   3. Every candidate is validated: exactly 15 digits AND passes Luhn.
#      - VERIFIED  -> checksum passed (trustworthy)
#      - UNVERIFIED-> 15 digits but checksum failed (flagged, not trusted)
#      - INVALID   -> not 15 digits (flagged)
#   4. Each IMEI is located on the page via OCR and cropped (best-effort);
#      the value + verification status are ALWAYS written to results.csv.

# Nothing is silently guessed: an IMEI that fails the checksum is reported
# as UNVERIFIED so a human can glance at it, rather than being trusted.
# """

# import os
# import csv
# import json
# import base64
# import difflib
# from collections import Counter
# from pathlib import Path

# from PIL import Image
# import pytesseract
# from dotenv import load_dotenv

# from ibm_watsonx_ai import Credentials
# from ibm_watsonx_ai.foundation_models import ModelInference

# # ---------------------------------------------------------------------------
# # CONFIG
# # ---------------------------------------------------------------------------
# INPUT_DIR = r"D:\Extra\TVS\Crop_field\input_files"
# OUTPUT_DIR = r"D:\Extra\TVS\Crop_field\output_files"

# VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# # Pixels of padding to add around the located value before cropping.
# PAD_X = 30
# PAD_Y = 15

# # Minimum fuzzy-match ratio (0-1) to accept an OCR match for the value.
# MATCH_THRESHOLD = 0.75

# # How many times to read each image; the majority answer per IMEI wins.
# # 1 = single read (fastest). 3 is a good accuracy/latency trade-off.
# N_READS = 3

# # If Tesseract is not on PATH (common on Windows), point to the exe:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # ---------------------------------------------------------------------------
# # watsonx model setup
# # ---------------------------------------------------------------------------
# load_dotenv()  # reads the .env file in the same folder

# API_KEY = os.getenv("IBM_API_KEY")
# SERVICE_URL = os.getenv("IBM_SERVICE_URL")
# PROJECT_ID = os.getenv("IBM_PROJECT_ID")
# MODEL_ID = os.getenv("IBM_MODEL_ID", "meta-llama/llama-4-maverick-17b-128e-instruct-fp8")

# if not all([API_KEY, SERVICE_URL, PROJECT_ID]):
#     raise SystemExit("Missing IBM_API_KEY / IBM_SERVICE_URL / IBM_PROJECT_ID in .env")

# _model = ModelInference(
#     model_id=MODEL_ID,
#     credentials=Credentials(url=SERVICE_URL, api_key=API_KEY),
#     project_id=PROJECT_ID,
# )

# # temperature 0 -> deterministic, most accurate for extraction
# _CHAT_PARAMS = {"temperature": 0, "max_tokens": 300}

# # PROMPT = (
# #     "You are reading a mobile-shop TAX INVOICE image from India. "
# #     "Find the phone's IMEI number(s). The label may be 'IMEI', 'IMEI No', "
# #     "'IMEI1 / IMEI2', 'IMEI 1', 'IMEI 2', 'I.M.E.I', 'MEID', 'S/N', or the "
# #     "digits may sit directly under a barcode. "
# #     "An IMEI is EXACTLY 15 digits (digits only, no letters). Dual-SIM phones "
# #     "print TWO IMEIs; single-SIM phones print one. There may also be a longer "
# #     "16-digit IMEISV - if so, ignore the extra 2 digits and keep the 15-digit "
# #     "IMEI, or return the 16 digits in 'raw' and the 15-digit IMEI in imei1.\n"
# #     "Read the digits EXACTLY as printed. Do NOT normalize, reorder, or 'fix' "
# #     "them to look nicer. Do NOT invent digits that are not clearly visible.\n"
# #     "Pay special attention to these visually similar digit pairs - inspect "
# #     "each digit individually and do NOT confuse them:\n"
# #     "1 vs 7\n"
# #     "2 vs 8\n"
# #     "2 vs 9\n"
# #     "3 vs 8\n"
# #     "7 vs 8\n"
# #     "0 vs 8\n"
# #     "6 vs 8\n"
# #     "Also watch: 5 vs 6, 5 vs 8, 4 vs 9, 0 vs 6, 0 vs O(letter), 1 vs I(letter).\n"
# #     "For each digit, decide it on its own shape - do NOT let a neighbouring "
# #     "digit or the desire for a 'nice' number influence your reading.\n"
# #     "If a digit is genuinely unreadable, do NOT guess it - return NONE for "
# #     "that IMEI instead of a half-guessed value.\n"
# #     "IMPORTANT about the second IMEI: only return imei2 if a SECOND IMEI "
# #     "VALUE is actually printed on the invoice. If the image shows an 'IMEI2' "
# #     "or 'IMEI 2' LABEL but there is NO digits next to it (blank, empty, or a "
# #     "dash), you MUST return \"imei2\": \"NONE\". Never copy imei1 into imei2 "
# #     "and never invent a second IMEI just because a label exists.\n"
# #     "Return ONLY a JSON object, no markdown, no extra text, in exactly this shape:\n"
# #     '{"imei1": "<15 digits or NONE>", "imei2": "<15 digits or NONE>", '
# #     '"raw": "<the IMEI text exactly as printed, including any label>"}\n'
# #     'If you cannot find any IMEI, return '
# #     '{"imei1": "NONE", "imei2": "NONE", "raw": "NONE"}.'
# # )

# PROMPT = (
#     "You are reading an Indian mobile-shop TAX INVOICE image. "
#     "Your task is to extract the phone's IMEI number(s) only.\n\n"

#     "An IMEI is normally EXACTLY 15 digits (digits only). "
#     "A dual-SIM phone may have TWO IMEIs, while a single-SIM phone may have ONE. "
#     "The IMEI may appear next to labels such as 'IMEI', 'IMEI No', 'IMEI1', "
#     "'IMEI 1', 'IMEI2', 'IMEI 2', 'I.M.E.I', or may appear as digits directly "
#     "below/next to a barcode.\n\n"

#     "IMPORTANT: Do not confuse an IMEI with other numbers printed on an invoice, "
#     "such as invoice numbers, GSTIN, product codes, SKU numbers, serial numbers, "
#     "order numbers, phone numbers, or barcode numbers. Extract a number as IMEI "
#     "only when the surrounding context indicates that it is an IMEI, or when the "
#     "barcode/text clearly represents an IMEI.\n\n"

#     "IMEI FORMAT:\n"
#     "- Valid IMEI: exactly 15 digits.\n"
#     "- If a clearly identified IMEISV is printed as 16 digits, treat the final "
#     "2 digits as the software-version digits and return the corresponding "
#     "15-digit IMEI when it can be determined clearly.\n"
#     "- Do NOT invent or reconstruct missing digits.\n"
#     "- Do NOT change any digit to make the number look valid or to satisfy a "
#     "Luhn checksum.\n"
#     "- Luhn/checksum validation may be used only as a secondary validation "
#     "signal, never as a reason to change the visually read digits.\n\n"

#     "READ THE IMAGE EXACTLY:\n"
#     "Read every digit exactly as printed. Do NOT normalize, reorder, correct, "
#     "or 'fix' the number.\n\n"

#     "Pay special attention to these visually similar digits:\n"
#     "1 vs 7\n"
#     "2 vs 8\n"
#     "2 vs 9\n"
#     "3 vs 8\n"
#     "7 vs 8\n"
#     "0 vs 8\n"
#     "6 vs 8\n"
#     "5 vs 6\n"
#     "5 vs 8\n"
#     "4 vs 9\n"
#     "0 vs 6\n"
#     "0 vs O (letter)\n"
#     "1 vs I (letter)\n\n"

#     "Inspect each digit individually based on its visible shape. "
#     "Do not let neighbouring digits, common IMEI patterns, checksum validation, "
#     "or the desire for a valid-looking number influence the reading.\n\n"

#     "If ANY digit of an IMEI is genuinely unreadable or ambiguous, return "
#     "\"NONE\" for that IMEI rather than guessing or partially reconstructing it.\n\n"

#     "SECOND IMEI RULE:\n"
#     "Only return imei2 when a SECOND IMEI VALUE is actually printed in the image. "
#     "If the invoice contains an 'IMEI2' or 'IMEI 2' label but there are no digits "
#     "next to it (blank, empty, dash, etc.), return \"imei2\": \"NONE\". "
#     "NEVER copy imei1 into imei2. NEVER invent a second IMEI merely because "
#     "an IMEI2 label exists.\n\n"

#     "If there are multiple phones on the invoice, extract the IMEI(s) belonging "
#     "to the phone/product being invoiced. Do not combine IMEIs from unrelated "
#     "products or other sections of the document.\n\n"

#     "OUTPUT RULES:\n"
#     "Return ONLY a valid JSON object. No markdown, no explanation, no extra text.\n"
#     "Use exactly this structure:\n"
#     "{\"imei1\": \"<15 digits or NONE>\", "
#     "\"imei2\": \"<15 digits or NONE>\", "
#     "\"raw\": \"<IMEI text exactly as printed or NONE>\"}\n\n"

#     "For a single IMEI, put it in imei1 and set imei2 to NONE.\n"
#     "For two IMEIs, put the first in imei1 and the second in imei2.\n"
#     "The raw field should contain the IMEI text as visually printed, including "
#     "labels such as IMEI/IMEI1/IMEI2 when clearly present. If two IMEIs are "
#     "present, include both in their printed order.\n\n"

#     "If no IMEI can be reliably identified, return exactly:\n"
#     "{\"imei1\": \"NONE\", \"imei2\": \"NONE\", \"raw\": \"NONE\"}"
# )

# # ---------------------------------------------------------------------------
# # IMEI validation (Luhn checksum) - this is what makes IMEI verifiable
# # ---------------------------------------------------------------------------
# def only_digits(s: str) -> str:
#     return "".join(ch for ch in str(s) if ch.isdigit())


# def luhn_ok(number: str) -> bool:
#     """True if `number` (a digit string) passes the Luhn checksum."""
#     digits = [int(c) for c in number]
#     if not digits:
#         return False
#     checksum = 0
#     # Double every second digit from the right.
#     for i, d in enumerate(reversed(digits)):
#         if i % 2 == 1:
#             d *= 2
#             if d > 9:
#                 d -= 9
#         checksum += d
#     return checksum % 10 == 0


# def classify_imei(raw_value: str):
#     """
#     Return (clean_imei, status) where status is:
#       'verified'   -> 15 digits and Luhn passes
#       'unverified' -> 15 digits but Luhn fails (report but don't trust)
#       'invalid'    -> not usable (wrong length / empty / NONE)
#     """
#     if not raw_value or str(raw_value).strip().upper() == "NONE":
#         return "", "invalid"
#     d = only_digits(raw_value)
#     # Tolerate a 16-digit IMEISV by keeping the first 15 (the IMEI part).
#     if len(d) == 16:
#         d = d[:15]
#     if len(d) != 15:
#         return d, "invalid"
#     return (d, "verified") if luhn_ok(d) else (d, "unverified")


# # ---------------------------------------------------------------------------
# # STEP 1 - read the IMEI(s) (structured) with Llama 4 Maverick
# # ---------------------------------------------------------------------------
# def _read_once(b64: str, mime: str) -> dict:
#     messages = [{
#         "role": "user",
#         "content": [
#             {"type": "text", "text": PROMPT},
#             {"type": "image_url",
#              "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
#         ],
#     }]
#     resp = _model.chat(messages=messages, params=_CHAT_PARAMS)
#     text = resp["choices"][0]["message"]["content"].strip()

#     # be tolerant of stray markdown fences or text around the JSON
#     if "```" in text:
#         text = text.split("```")[1].replace("json", "", 1).strip()
#     start, end = text.find("{"), text.rfind("}")
#     if start != -1 and end != -1:
#         text = text[start:end + 1]
#     try:
#         return json.loads(text)
#     except Exception:
#         return {"imei1": "NONE", "imei2": "NONE", "raw": text}


# def read_imeis(image_path: Path) -> dict:
#     """
#     Read the image N_READS times and take a majority vote per slot.
#     Returns {"imei1": str, "imei2": str, "raw": str}.
#     """
#     with open(image_path, "rb") as f:
#         b64 = base64.b64encode(f.read()).decode("utf-8")
#     ext = image_path.suffix.lower().lstrip(".")
#     mime = "jpeg" if ext in ("jpg", "jpeg") else ext

#     votes1, votes2, raws = Counter(), Counter(), []
#     for _ in range(max(1, N_READS)):
#         obj = _read_once(b64, mime)
#         c1, _ = classify_imei(obj.get("imei1", ""))
#         c2, _ = classify_imei(obj.get("imei2", ""))
#         if c1:
#             votes1[c1] += 1
#         if c2:
#             votes2[c2] += 1
#         raws.append(str(obj.get("raw", "")).strip())

#     imei1 = votes1.most_common(1)[0][0] if votes1 else "NONE"
#     imei2 = votes2.most_common(1)[0][0] if votes2 else "NONE"

#     # Guard: never treat imei2 as present if it is blank or just a copy of
#     # imei1 (the model sometimes echoes imei1 when only an empty IMEI2 label
#     # exists). A real second IMEI must be a distinct 15-digit value.
#     if imei2 == imei1:
#         imei2 = "NONE"
#     raw = next((r for r in raws if r and r.upper() != "NONE"), raws[0] if raws else "NONE")
#     return {"imei1": imei1, "imei2": imei2, "raw": raw}


# # ---------------------------------------------------------------------------
# # STEP 2/3 - locate the value's pixels via OCR
# # ---------------------------------------------------------------------------
# def _clean(s: str) -> str:
#     return "".join(ch for ch in s if ch.isalnum()).lower()


# def locate_value_box(image: Image.Image, value: str):
#     """Return (left, top, right, bottom) of the value on the page, or None."""
#     target = _clean(value)
#     if not target:
#         return None

#     data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#     n = len(data["text"])

#     # Group word indices by (block, paragraph, line) so we only join words
#     # that sit on the same physical line.
#     lines = {}
#     for i in range(n):
#         if not data["text"][i].strip():
#             continue
#         try:
#             if float(data["conf"][i]) < 0:
#                 continue
#         except (ValueError, TypeError):
#             continue
#         key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
#         lines.setdefault(key, []).append(i)

#     best = None  # (ratio, box)
#     for idxs in lines.values():
#         # An IMEI can be split across several OCR tokens; allow a wide window.
#         for start in range(len(idxs)):
#             for length in range(1, 7):
#                 window = idxs[start:start + length]
#                 if not window:
#                     continue
#                 joined = _clean("".join(data["text"][j] for j in window))
#                 if not joined:
#                     continue
#                 ratio = difflib.SequenceMatcher(None, joined, target).ratio()
#                 if target in joined or joined in target:
#                     ratio = max(ratio, 0.9)
#                 if best is None or ratio > best[0]:
#                     lefts = [data["left"][j] for j in window]
#                     tops = [data["top"][j] for j in window]
#                     rights = [data["left"][j] + data["width"][j] for j in window]
#                     bottoms = [data["top"][j] + data["height"][j] for j in window]
#                     box = (min(lefts), min(tops), max(rights), max(bottoms))
#                     best = (ratio, box)

#     if best and best[0] >= MATCH_THRESHOLD:
#         return best[1]
#     return None


# # ---------------------------------------------------------------------------
# # STEP 4 - crop and save
# # ---------------------------------------------------------------------------
# def crop_and_save(image: Image.Image, box, out_path: Path):
#     w, h = image.size
#     left = max(0, box[0] - PAD_X)
#     top = max(0, box[1] - PAD_Y)
#     right = min(w, box[2] + PAD_X)
#     bottom = min(h, box[3] + PAD_Y)
#     image.crop((left, top, right, bottom)).save(out_path)


# # ---------------------------------------------------------------------------
# # MAIN
# # ---------------------------------------------------------------------------
# def main():
#     in_dir, out_dir = Path(INPUT_DIR), Path(OUTPUT_DIR)
#     if not in_dir.exists():
#         raise FileNotFoundError(f"Input folder not found: {in_dir}")
#     out_dir.mkdir(parents=True, exist_ok=True)

#     images = [p for p in in_dir.iterdir()
#               if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
#     if not images:
#         print(f"No images found in {in_dir}")
#         return

#     results = []
#     print(f"Processing {len(images)} image(s)  (N_READS={N_READS})...\n")

#     for img_path in images:
#         print(f"- {img_path.name}")
#         try:
#             obj = read_imeis(img_path)
#             image = None  # opened lazily only if we have something to locate

#             for slot in ("imei1", "imei2"):
#                 clean, status = classify_imei(obj.get(slot, ""))
#                 if status == "invalid" and not clean:
#                     continue  # this slot genuinely had no IMEI

#                 print(f"    {slot}: {clean or '(unreadable)'}  [{status}]")

#                 crop_status = "not_attempted"
#                 if clean:
#                     if image is None:
#                         image = Image.open(img_path)
#                     box = locate_value_box(image, clean)
#                     if box:
#                         out_path = out_dir / f"{img_path.stem}_{slot}.png"
#                         crop_and_save(image, box, out_path)
#                         crop_status = "cropped"
#                         print(f"        cropped -> {out_path.name}")
#                     else:
#                         crop_status = "value_read_but_not_located"

#                 results.append({
#                     "file": img_path.name,
#                     "slot": slot,
#                     "imei": clean,
#                     "verification": status,   # verified / unverified / invalid
#                     "raw": obj.get("raw", ""),
#                     "crop_status": crop_status,
#                 })

#             # If neither slot produced anything, still log the miss.
#             if not any(r["file"] == img_path.name for r in results):
#                 results.append({
#                     "file": img_path.name, "slot": "-", "imei": "",
#                     "verification": "not_read_by_model",
#                     "raw": obj.get("raw", ""), "crop_status": "not_attempted",
#                 })

#         except Exception as e:
#             print(f"    ERROR: {e}")
#             results.append({
#                 "file": img_path.name, "slot": "-", "imei": "",
#                 "verification": f"error: {e}", "raw": "", "crop_status": "-",
#             })

#     csv_path = out_dir / "results_imei.csv"
#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=[
#             "file", "slot", "imei", "verification", "raw", "crop_status"])
#         writer.writeheader()
#         writer.writerows(results)

#     verified = sum(1 for r in results if r["verification"] == "verified")
#     print(f"\nDone. {verified} IMEI(s) checksum-verified.")
#     print(f"Log written to {csv_path}")


# if __name__ == "__main__":
#     main()


# ******************************************

# """
# Extract the IMEI number(s) from scanned mobile-shop invoices using
# watsonx.ai Llama 4 Maverick (to READ the value) + Tesseract OCR
# (to LOCATE the value's pixels) + Pillow (to CROP).

# Why this can be trusted:
#   An IMEI is 15 digits and its LAST digit is a Luhn check digit over the
#   first 14. So unlike a plain invoice number or a date, an IMEI can be
#   MATHEMATICALLY VERIFIED. Any single-digit misread almost always breaks
#   the checksum, so we can detect it instead of returning a wrong value.

# Accuracy strategy (highest-confidence first):
#   1. Maverick returns STRUCTURED JSON: imei1 / imei2 (dual-SIM phones
#      print two) plus the raw text, with explicit digit-confusion warnings.
#   2. We read the image up to N_READS times and take the majority vote per
#      IMEI (consensus). Temperature 0 keeps each read deterministic; the
#      repeat is a guard against occasional flakiness.
#   3. Every candidate is validated: exactly 15 digits AND passes Luhn.
#      - VERIFIED  -> checksum passed (trustworthy)
#      - UNVERIFIED-> 15 digits but checksum failed (flagged, not trusted)
#      - INVALID   -> not 15 digits (flagged)
#   4. Each IMEI is located on the page via OCR and cropped (best-effort);
#      the value + verification status are ALWAYS written to results.csv.

# Nothing is silently guessed: an IMEI that fails the checksum is reported
# as UNVERIFIED so a human can glance at it, rather than being trusted.
# """

# import os
# import csv
# import json
# import base64
# import difflib
# from collections import Counter
# from pathlib import Path

# from PIL import Image
# import pytesseract
# from dotenv import load_dotenv

# from ibm_watsonx_ai import Credentials
# from ibm_watsonx_ai.foundation_models import ModelInference

# # ---------------------------------------------------------------------------
# # CONFIG
# # ---------------------------------------------------------------------------
# INPUT_DIR = r"D:\Extra\TVS\Crop_field\input_files"
# OUTPUT_DIR = r"D:\Extra\TVS\Crop_field\output_files"

# VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# # Pixels of padding to add around the located value before cropping.
# PAD_X = 30
# PAD_Y = 15

# # Minimum fuzzy-match ratio (0-1) to accept an OCR match for the value.
# MATCH_THRESHOLD = 0.75

# # --- Accuracy escalation ---------------------------------------------------
# # The pipeline is CHEAP when the read is clean and THOROUGH only when it isn't.
# #   Pass 1: one greedy read (temperature 0) with constrained chain-of-thought.
# #   Pass 2: if an IMEI fails the Luhn checksum, zoom into its region on the
# #           page and re-read the enlarged crop (biggest accuracy win).
# #   Pass 3: if it still fails, take SAMPLED reads (temperature > 0 + top_p)
# #           and majority-vote. Sampling is ONLY used here, because voting is
# #           pointless unless the reads can actually differ.
# ZOOM_FACTOR = 3.0          # how much to enlarge the cropped region for re-reading
# SAMPLED_READS = 4          # number of sampled reads in the pass-3 vote (0 disables)

# # Greedy params for passes 1 & 2 (deterministic, most accurate single answer).
# _GREEDY_PARAMS = {"temperature": 0, "max_tokens": 400}
# # Sampled params for pass 3 only (diversity so the vote is meaningful).
# _SAMPLED_PARAMS = {"temperature": 0.5, "top_p": 0.95, "top_k": 40, "max_tokens": 400}

# # --- Resolution / DPI ------------------------------------------------------
# # Tesseract and the vision model both read digits far more accurately on
# # high-resolution input. Small scans are upscaled once at load so every stage
# # (model read, OCR locate, fallback scan, crop) sees the sharper image.
# OCR_DPI = 300              # DPI hint passed to Tesseract
# UPSCALE_MIN_WIDTH = 2200   # if an image is narrower than this, enlarge it
# UPSCALE_MAX_FACTOR = 4.0   # never enlarge more than this (avoids huge uploads)

# # If Tesseract is not on PATH (common on Windows), point to the exe:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # ---------------------------------------------------------------------------
# # watsonx model setup
# # ---------------------------------------------------------------------------
# load_dotenv()  # reads the .env file in the same folder

# API_KEY = os.getenv("IBM_API_KEY")
# SERVICE_URL = os.getenv("IBM_SERVICE_URL")
# PROJECT_ID = os.getenv("IBM_PROJECT_ID")
# MODEL_ID = os.getenv("IBM_MODEL_ID", "meta-llama/llama-4-maverick-17b-128e-instruct-fp8")

# if not all([API_KEY, SERVICE_URL, PROJECT_ID]):
#     raise SystemExit("Missing IBM_API_KEY / IBM_SERVICE_URL / IBM_PROJECT_ID in .env")

# _model = ModelInference(
#     model_id=MODEL_ID,
#     credentials=Credentials(url=SERVICE_URL, api_key=API_KEY),
#     project_id=PROJECT_ID,
# )

# PROMPT = (
#     "You are reading a mobile-shop TAX INVOICE image from India. "
#     "Find the phone's IMEI number(s). The label may be 'IMEI', 'IMEI No', "
#     "'IMEI1 / IMEI2', 'IMEI 1', 'IMEI 2', 'I.M.E.I', 'MEID', 'S/N', or the "
#     "digits may sit directly under a barcode. "
#     "An IMEI is EXACTLY 15 digits (digits only, no letters). Dual-SIM phones "
#     "print TWO IMEIs; single-SIM phones print one. There may also be a longer "
#     "16-digit IMEISV - if so, ignore the extra 2 digits and keep the 15-digit "
#     "IMEI, or return the 16 digits in 'raw' and the 15-digit IMEI in imei1.\n"
#     "Read the digits EXACTLY as printed. Do NOT normalize, reorder, or 'fix' "
#     "them to look nicer. Do NOT invent digits that are not clearly visible.\n"
#     "Pay special attention to these visually similar digit pairs - inspect "
#     "each digit individually and do NOT confuse them:\n"
#     "1 vs 7\n"
#     "2 vs 8\n"
#     "2 vs 9\n"
#     "3 vs 8\n"
#     "7 vs 8\n"
#     "0 vs 8\n"
#     "6 vs 8\n"
#     "Also watch: 5 vs 6, 5 vs 8, 4 vs 9, 0 vs 6, 0 vs O(letter), 1 vs I(letter).\n"
#     "For each digit, decide it on its own shape - do NOT let a neighbouring "
#     "digit or the desire for a 'nice' number influence your reading.\n"
#     "If a digit is genuinely unreadable, do NOT guess it - return NONE for "
#     "that IMEI instead of a half-guessed value.\n"
#     "IMPORTANT about the second IMEI: only return imei2 if a SECOND IMEI "
#     "VALUE is actually printed on the invoice. If the image shows an 'IMEI2' "
#     "or 'IMEI 2' LABEL but there is NO digits next to it (blank, empty, or a "
#     "dash), you MUST return \"imei2\": \"NONE\". Never copy imei1 into imei2 "
#     "and never invent a second IMEI just because a label exists.\n"
#     "THINK STEP BY STEP, but ONLY by transcribing - never by correcting:\n"
#     "In the 'reasoning' field, write out each IMEI you see digit by digit, "
#     "left to right, describing any digit you are unsure about and the shape "
#     "you actually see. This is transcription, NOT correction: you must copy "
#     "what is printed even if the result looks odd. Do NOT change a digit to "
#     "make the number look nicer or to pass any checksum.\n"
#     "IMPORTANT - the label may be MISSING entirely. Many invoices print the "
#     "IMEI as a bare 15-digit number with no 'IMEI' word next to it - often "
#     "directly under a barcode, or on the same line as the phone's model name "
#     "or the item description. If you see ANY standalone 15-digit number near "
#     "the item/model, treat it as an IMEI and return it, even when there is no "
#     "label at all. Do NOT skip a 15-digit number just because it is unlabeled.\n"
#     "Return ONLY a JSON object, no markdown, no extra text, in exactly this shape:\n"
#     '{"reasoning": "<your digit-by-digit transcription notes>", '
#     '"imei1": "<15 digits or NONE>", "imei2": "<15 digits or NONE>", '
#     '"raw": "<the IMEI text exactly as printed, including any label>"}\n'
#     'If you cannot find any IMEI, return the same shape with '
#     '"imei1": "NONE", "imei2": "NONE", "raw": "NONE".'
# )


# # ---------------------------------------------------------------------------
# # IMEI validation (Luhn checksum) - this is what makes IMEI verifiable
# # ---------------------------------------------------------------------------
# def only_digits(s: str) -> str:
#     return "".join(ch for ch in str(s) if ch.isdigit())


# def luhn_ok(number: str) -> bool:
#     """True if `number` (a digit string) passes the Luhn checksum."""
#     digits = [int(c) for c in number]
#     if not digits:
#         return False
#     checksum = 0
#     # Double every second digit from the right.
#     for i, d in enumerate(reversed(digits)):
#         if i % 2 == 1:
#             d *= 2
#             if d > 9:
#                 d -= 9
#         checksum += d
#     return checksum % 10 == 0


# def classify_imei(raw_value: str):
#     """
#     Return (clean_imei, status) where status is:
#       'verified'   -> 15 digits and Luhn passes
#       'unverified' -> 15 digits but Luhn fails (report but don't trust)
#       'invalid'    -> not usable (wrong length / empty / NONE)
#     """
#     if not raw_value or str(raw_value).strip().upper() == "NONE":
#         return "", "invalid"
#     d = only_digits(raw_value)
#     # Tolerate a 16-digit IMEISV by keeping the first 15 (the IMEI part).
#     if len(d) == 16:
#         d = d[:15]
#     if len(d) != 15:
#         return d, "invalid"
#     return (d, "verified") if luhn_ok(d) else (d, "unverified")


# def find_imeis_by_ocr(image: Image.Image):
#     """
#     Label-free fallback: scan the page for ANY 15-digit run and keep only the
#     ones that pass the Luhn checksum. The checksum is what makes this safe -
#     a random 15-digit number passes Luhn only ~10% of the time, so a Luhn-
#     valid run is almost certainly a real IMEI, label or not.

#     Returns a list of unique verified 15-digit strings, in reading order.
#     """
#     data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=f"--dpi {OCR_DPI}")
#     n = len(data["text"])

#     # Concatenate the digits of tokens that sit on the same physical line, so
#     # an IMEI printed with spaces/groups still forms one run - but numbers on
#     # different lines never merge into a false 15-digit run.
#     lines = {}
#     for i in range(n):
#         if not data["text"][i].strip():
#             continue
#         key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
#         lines.setdefault(key, []).append(only_digits(data["text"][i]))

#     found = []
#     for parts in lines.values():
#         run = "".join(parts)
#         # Slide a 15-digit window across the line's digits.
#         for start in range(0, max(0, len(run) - 14)):
#             window = run[start:start + 15]
#             if len(window) == 15 and luhn_ok(window) and window not in found:
#                 found.append(window)
#     return found


# # ---------------------------------------------------------------------------
# # STEP 1 - read the IMEI(s) (structured) with Llama 4 Maverick
# # ---------------------------------------------------------------------------
# def prepare_image(image: Image.Image) -> Image.Image:
#     """
#     Raise effective resolution: upscale small scans so the shorter details
#     (thin digit strokes) survive OCR and the model read. Done once at load so
#     all downstream pixel coordinates stay consistent.
#     """
#     image = image.convert("RGB")
#     w, h = image.size
#     if w < UPSCALE_MIN_WIDTH:
#         factor = min(UPSCALE_MAX_FACTOR, UPSCALE_MIN_WIDTH / float(w))
#         if factor > 1.0:
#             image = image.resize((int(w * factor), int(h * factor)), Image.LANCZOS)
#     return image


# def _encode_pil(image: Image.Image):
#     """Return (b64, mime) for a PIL image, always as PNG (lossless)."""
#     import io
#     buf = io.BytesIO()
#     image.convert("RGB").save(buf, format="PNG")
#     return base64.b64encode(buf.getvalue()).decode("utf-8"), "png"


# def _read_once(image: Image.Image, params: dict) -> dict:
#     b64, mime = _encode_pil(image)
#     messages = [{
#         "role": "user",
#         "content": [
#             {"type": "text", "text": PROMPT},
#             {"type": "image_url",
#              "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
#         ],
#     }]
#     resp = _model.chat(messages=messages, params=params)
#     text = resp["choices"][0]["message"]["content"].strip()

#     # be tolerant of stray markdown fences or text around the JSON
#     if "```" in text:
#         text = text.split("```")[1].replace("json", "", 1).strip()
#     start, end = text.find("{"), text.rfind("}")
#     if start != -1 and end != -1:
#         text = text[start:end + 1]
#     try:
#         return json.loads(text)
#     except Exception:
#         return {"imei1": "NONE", "imei2": "NONE", "raw": text}


# def _zoom_reread(image: Image.Image, value: str):
#     """
#     Pass 2: locate `value` on the page, crop it with generous padding, enlarge
#     it ZOOM_FACTOR x, and re-read just that region. Returns a cleaned IMEI
#     string if the zoomed read now passes Luhn, else None.
#     """
#     box = locate_value_box(image, value)
#     if not box:
#         return None
#     w, h = image.size
#     px, py = 40, 25
#     crop = image.crop((max(0, box[0] - px), max(0, box[1] - py),
#                        min(w, box[2] + px), min(h, box[3] + py)))
#     if ZOOM_FACTOR and ZOOM_FACTOR != 1.0:
#         crop = crop.resize((int(crop.width * ZOOM_FACTOR),
#                             int(crop.height * ZOOM_FACTOR)), Image.LANCZOS)
#     obj = _read_once(crop, _GREEDY_PARAMS)
#     for slot in ("imei1", "imei2"):
#         clean, status = classify_imei(obj.get(slot, ""))
#         if status == "verified":
#             return clean
#     return None


# def _sampled_vote(image: Image.Image, slot: str):
#     """
#     Pass 3: take SAMPLED_READS diverse reads and return the most common
#     checksum-VERIFIED value for `slot`, or None. Sampling (temp>0 + top_p/
#     top_k) is what makes the reads differ, so the vote is meaningful.
#     """
#     if SAMPLED_READS <= 0:
#         return None
#     votes = Counter()
#     for _ in range(SAMPLED_READS):
#         obj = _read_once(image, _SAMPLED_PARAMS)
#         clean, status = classify_imei(obj.get(slot, ""))
#         if status == "verified":
#             votes[clean] += 1
#     return votes.most_common(1)[0][0] if votes else None


# def read_imeis(image: Image.Image) -> dict:
#     """
#     Escalation ladder. Returns {"imei1": str, "imei2": str, "raw": str}.
#     Pass 1 greedy read -> pass 2 zoom re-read -> pass 3 sampled vote, each
#     triggered only for a slot that has not yet passed the Luhn checksum.
#     """
#     base = _read_once(image, _GREEDY_PARAMS)
#     raw = str(base.get("raw", "")).strip() or "NONE"
#     resolved = {}

#     for slot in ("imei1", "imei2"):
#         clean, status = classify_imei(base.get(slot, ""))
#         if status == "invalid" and not clean:
#             resolved[slot] = "NONE"          # genuinely nothing in this slot
#             continue
#         if status == "verified":
#             resolved[slot] = clean            # pass 1 already trustworthy
#             continue
#         # Not verified -> escalate. Try zoom, then sampled vote.
#         better = _zoom_reread(image, clean or raw)
#         if not better:
#             better = _sampled_vote(image, slot)
#         resolved[slot] = better if better else (clean or "NONE")

#     # Guard: a real second IMEI must be a distinct value, never a copy of #1.
#     if resolved.get("imei2") == resolved.get("imei1"):
#         resolved["imei2"] = "NONE"

#     # Pass 4 (label-free fallback): if a slot is still empty or unverified,
#     # try Luhn-valid 15-digit numbers found anywhere on the page by OCR. This
#     # rescues IMEIs printed with no label that the model returned as NONE.
#     def _is_verified(v):
#         return bool(v) and v != "NONE" and classify_imei(v)[1] == "verified"

#     if not _is_verified(resolved.get("imei1")) or not _is_verified(resolved.get("imei2")):
#         used = {v for v in resolved.values() if _is_verified(v)}
#         candidates = [c for c in find_imeis_by_ocr(image) if c not in used]
#         ci = iter(candidates)
#         for slot in ("imei1", "imei2"):
#             if not _is_verified(resolved.get(slot)):
#                 nxt = next(ci, None)
#                 if nxt:
#                     resolved[slot] = nxt

#     if resolved.get("imei2") == resolved.get("imei1"):
#         resolved["imei2"] = "NONE"

#     return {"imei1": resolved.get("imei1", "NONE"),
#             "imei2": resolved.get("imei2", "NONE"), "raw": raw}


# # ---------------------------------------------------------------------------
# # STEP 2/3 - locate the value's pixels via OCR
# # ---------------------------------------------------------------------------
# def _clean(s: str) -> str:
#     return "".join(ch for ch in s if ch.isalnum()).lower()


# def locate_value_box(image: Image.Image, value: str):
#     """Return (left, top, right, bottom) of the value on the page, or None."""
#     target = _clean(value)
#     if not target:
#         return None

#     data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=f"--dpi {OCR_DPI}")
#     n = len(data["text"])

#     # Group word indices by (block, paragraph, line) so we only join words
#     # that sit on the same physical line.
#     lines = {}
#     for i in range(n):
#         if not data["text"][i].strip():
#             continue
#         try:
#             if float(data["conf"][i]) < 0:
#                 continue
#         except (ValueError, TypeError):
#             continue
#         key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
#         lines.setdefault(key, []).append(i)

#     best = None  # (ratio, box)
#     for idxs in lines.values():
#         # An IMEI can be split across several OCR tokens; allow a wide window.
#         for start in range(len(idxs)):
#             for length in range(1, 7):
#                 window = idxs[start:start + length]
#                 if not window:
#                     continue
#                 joined = _clean("".join(data["text"][j] for j in window))
#                 if not joined:
#                     continue
#                 ratio = difflib.SequenceMatcher(None, joined, target).ratio()
#                 if target in joined or joined in target:
#                     ratio = max(ratio, 0.9)
#                 if best is None or ratio > best[0]:
#                     lefts = [data["left"][j] for j in window]
#                     tops = [data["top"][j] for j in window]
#                     rights = [data["left"][j] + data["width"][j] for j in window]
#                     bottoms = [data["top"][j] + data["height"][j] for j in window]
#                     box = (min(lefts), min(tops), max(rights), max(bottoms))
#                     best = (ratio, box)

#     if best and best[0] >= MATCH_THRESHOLD:
#         return best[1]
#     return None


# # ---------------------------------------------------------------------------
# # STEP 4 - crop and save
# # ---------------------------------------------------------------------------
# def crop_and_save(image: Image.Image, box, out_path: Path):
#     w, h = image.size
#     left = max(0, box[0] - PAD_X)
#     top = max(0, box[1] - PAD_Y)
#     right = min(w, box[2] + PAD_X)
#     bottom = min(h, box[3] + PAD_Y)
#     image.crop((left, top, right, bottom)).save(out_path)


# # ---------------------------------------------------------------------------
# # MAIN
# # ---------------------------------------------------------------------------
# def main():
#     in_dir, out_dir = Path(INPUT_DIR), Path(OUTPUT_DIR)
#     if not in_dir.exists():
#         raise FileNotFoundError(f"Input folder not found: {in_dir}")
#     out_dir.mkdir(parents=True, exist_ok=True)

#     images = [p for p in in_dir.iterdir()
#               if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
#     if not images:
#         print(f"No images found in {in_dir}")
#         return

#     results = []
#     print(f"Processing {len(images)} image(s)  "
#           f"(zoom x{ZOOM_FACTOR}, sampled_reads={SAMPLED_READS})...\n")

#     for img_path in images:
#         print(f"- {img_path.name}")
#         try:
#             image = prepare_image(Image.open(img_path))
#             obj = read_imeis(image)
#             logged_any = False

#             for slot in ("imei1", "imei2"):
#                 clean, status = classify_imei(obj.get(slot, ""))
#                 if status == "invalid" and not clean:
#                     continue  # this slot genuinely had no IMEI

#                 print(f"    {slot}: {clean or '(unreadable)'}  [{status}]")
#                 logged_any = True

#                 crop_status = "not_attempted"
#                 if clean:
#                     box = locate_value_box(image, clean)
#                     if box:
#                         out_path = out_dir / f"{img_path.stem}_{slot}.png"
#                         crop_and_save(image, box, out_path)
#                         crop_status = "cropped"
#                         print(f"        cropped -> {out_path.name}")
#                     else:
#                         crop_status = "value_read_but_not_located"

#                 results.append({
#                     "file": img_path.name,
#                     "slot": slot,
#                     "imei": clean,
#                     "verification": status,   # verified / unverified / invalid
#                     "raw": obj.get("raw", ""),
#                     "crop_status": crop_status,
#                 })

#             # If neither slot produced anything, still log the miss.
#             if not logged_any:
#                 results.append({
#                     "file": img_path.name, "slot": "-", "imei": "",
#                     "verification": "not_read_by_model",
#                     "raw": obj.get("raw", ""), "crop_status": "not_attempted",
#                 })

#         except Exception as e:
#             print(f"    ERROR: {e}")
#             results.append({
#                 "file": img_path.name, "slot": "-", "imei": "",
#                 "verification": f"error: {e}", "raw": "", "crop_status": "-",
#             })

#     csv_path = out_dir / "results_imei.csv"
#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=[
#             "file", "slot", "imei", "verification", "raw", "crop_status"])
#         writer.writeheader()
#         writer.writerows(results)

#     verified = sum(1 for r in results if r["verification"] == "verified")
#     print(f"\nDone. {verified} IMEI(s) checksum-verified.")
#     print(f"Log written to {csv_path}")


# if __name__ == "__main__":
#     main()

# ********************************************    

"""
Extract the IMEI number(s) from scanned mobile-shop invoices using
watsonx.ai Llama 4 Maverick (to READ the value) + Tesseract OCR
(to LOCATE the value's pixels) + Pillow (to CROP).

Why this can be trusted:
  An IMEI is 15 digits and its LAST digit is a Luhn check digit over the
  first 14. So unlike a plain invoice number or a date, an IMEI can be
  MATHEMATICALLY VERIFIED. Any single-digit misread almost always breaks
  the checksum, so we can detect it instead of returning a wrong value.

Accuracy strategy (highest-confidence first):
  1. Maverick returns STRUCTURED JSON: imei1 / imei2 (dual-SIM phones
     print two) plus the raw text, with explicit digit-confusion warnings.
  2. We read the image up to N_READS times and take the majority vote per
     IMEI (consensus). Temperature 0 keeps each read deterministic; the
     repeat is a guard against occasional flakiness.
  3. Every candidate is validated: exactly 15 digits AND passes Luhn.
     - VERIFIED  -> checksum passed (trustworthy)
     - UNVERIFIED-> 15 digits but checksum failed (flagged, not trusted)
     - INVALID   -> not 15 digits (flagged)
  4. Each IMEI is located on the page via OCR and cropped (best-effort);
     the value + verification status are ALWAYS written to results.csv.

Nothing is silently guessed: an IMEI that fails the checksum is reported
as UNVERIFIED so a human can glance at it, rather than being trusted.
"""

import os
import csv
import json
import base64
import difflib
from collections import Counter
from pathlib import Path

from PIL import Image
import pytesseract
from dotenv import load_dotenv

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_DIR = r"D:\Extra\TVS\Crop_field\input_files"
OUTPUT_DIR = r"D:\Extra\TVS\Crop_field\output_files"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Pixels of padding to add around the located value before cropping.
PAD_X = 30
PAD_Y = 15

# Minimum fuzzy-match ratio (0-1) to accept an OCR match for the value.
MATCH_THRESHOLD = 0.75

# --- Accuracy escalation ---------------------------------------------------
# The pipeline is CHEAP when the read is clean and THOROUGH only when it isn't.
#   Pass 1: one greedy read (temperature 0) with constrained chain-of-thought.
#   Pass 2: if an IMEI fails the Luhn checksum, zoom into its region on the
#           page and re-read the enlarged crop (biggest accuracy win).
#   Pass 3: if it still fails, take SAMPLED reads (temperature > 0 + top_p)
#           and majority-vote. Sampling is ONLY used here, because voting is
#           pointless unless the reads can actually differ.
ZOOM_FACTOR = 3.0          # how much to enlarge the cropped region for re-reading
SAMPLED_READS = 4          # number of sampled reads in the pass-3 vote (0 disables)

# Greedy params for passes 1 & 2 (deterministic, most accurate single answer).
_GREEDY_PARAMS = {"temperature": 0, "max_tokens": 400}
# Sampled params for pass 3 only (diversity so the vote is meaningful).
_SAMPLED_PARAMS = {"temperature": 0.5, "top_p": 0.95, "top_k": 40, "max_tokens": 400}

# --- Resolution / DPI ------------------------------------------------------
# Tesseract and the vision model both read digits far more accurately on
# high-resolution input. Small scans are upscaled once at load so every stage
# (model read, OCR locate, fallback scan, crop) sees the sharper image.
OCR_DPI = 300              # DPI hint passed to Tesseract
UPSCALE_MIN_WIDTH = 2200   # if an image is narrower than this, enlarge it
UPSCALE_MAX_FACTOR = 4.0   # never enlarge more than this (avoids huge uploads)

# If Tesseract is not on PATH (common on Windows), point to the exe:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------------------------------------------------------
# watsonx model setup
# ---------------------------------------------------------------------------
load_dotenv()  # reads the .env file in the same folder

API_KEY = os.getenv("IBM_API_KEY")
SERVICE_URL = os.getenv("IBM_SERVICE_URL")
PROJECT_ID = os.getenv("IBM_PROJECT_ID")
MODEL_ID = os.getenv("IBM_MODEL_ID", "meta-llama/llama-4-maverick-17b-128e-instruct-fp8")

if not all([API_KEY, SERVICE_URL, PROJECT_ID]):
    raise SystemExit("Missing IBM_API_KEY / IBM_SERVICE_URL / IBM_PROJECT_ID in .env")

_model = ModelInference(
    model_id=MODEL_ID,
    credentials=Credentials(url=SERVICE_URL, api_key=API_KEY),
    project_id=PROJECT_ID,
)

PROMPT = (
    "You are reading a mobile-shop TAX INVOICE image from India. "
    "Find the phone's IMEI number(s). The label may be 'IMEI', 'IMEI No', "
    "'IMEI1 / IMEI2', 'IMEI 1', 'IMEI 2', 'I.M.E.I', 'MEID', 'S/N', or the "
    "digits may sit directly under a barcode. "
    "An IMEI is EXACTLY 15 digits (digits only, no letters). Dual-SIM phones "
    "print TWO IMEIs; single-SIM phones print one. There may also be a longer "
    "16-digit IMEISV - if so, ignore the extra 2 digits and keep the 15-digit "
    "IMEI, or return the 16 digits in 'raw' and the 15-digit IMEI in imei1.\n"
    "Read the digits EXACTLY as printed. Do NOT normalize, reorder, or 'fix' "
    "them to look nicer. Do NOT invent digits that are not clearly visible.\n"
    "Pay special attention to these visually similar digit pairs - inspect "
    "each digit individually and do NOT confuse them:\n"
    "1 vs 7\n"
    "2 vs 8\n"
    "2 vs 9\n"
    "3 vs 8\n"
    "7 vs 8\n"
    "0 vs 8\n"
    "6 vs 8\n"
    "Also watch: 5 vs 6, 5 vs 8, 4 vs 9, 0 vs 6, 0 vs O(letter), 1 vs I(letter).\n"
    "For each digit, decide it on its own shape - do NOT let a neighbouring "
    "digit or the desire for a 'nice' number influence your reading.\n"
    "If a digit is genuinely unreadable, do NOT guess it - return NONE for "
    "that IMEI instead of a half-guessed value.\n"
    "IMPORTANT about the second IMEI: only return imei2 if a SECOND IMEI "
    "VALUE is actually printed on the invoice. If the image shows an 'IMEI2' "
    "or 'IMEI 2' LABEL but there is NO digits next to it (blank, empty, or a "
    "dash), you MUST return \"imei2\": \"NONE\". Never copy imei1 into imei2 "
    "and never invent a second IMEI just because a label exists.\n"
    "Not every invoice will contain two IMEI numbers. If IMEI 2 is not present "
    "on the invoice, leave the imei2 field as \"NONE\". Do not assume that "
    "every phone has a second IMEI. Only populate imei2 when a second IMEI "
    "number is clearly printed and visible on the invoice. Never copy imei1 "
    "into imei2 or generate a second IMEI when one is not present. The same "
    "rule applies to imei1: only populate imei1 when a first IMEI number is "
    "clearly printed and visible; if no IMEI is present, return "
    "\"imei1\": \"NONE\" and never invent or guess one.\n"
    "THINK STEP BY STEP, but ONLY by transcribing - never by correcting:\n"
    "In the 'reasoning' field, write out each IMEI you see digit by digit, "
    "left to right, describing any digit you are unsure about and the shape "
    "you actually see. This is transcription, NOT correction: you must copy "
    "what is printed even if the result looks odd. Do NOT change a digit to "
    "make the number look nicer or to pass any checksum.\n"
    "IMPORTANT - the label may be MISSING entirely. Many invoices print the "
    "IMEI as a bare 15-digit number with no 'IMEI' word next to it - often "
    "directly under a barcode, or on the same line as the phone's model name "
    "or the item description. If you see ANY standalone 15-digit number near "
    "the item/model, treat it as an IMEI and return it, even when there is no "
    "label at all. Do NOT skip a 15-digit number just because it is unlabeled.\n"
    "Return ONLY a JSON object, no markdown, no extra text, in exactly this shape:\n"
    '{"reasoning": "<your digit-by-digit transcription notes>", '
    '"imei1": "<15 digits or NONE>", "imei2": "<15 digits or NONE>", '
    '"raw": "<the IMEI text exactly as printed, including any label>"}\n'
    'If you cannot find any IMEI, return the same shape with '
    '"imei1": "NONE", "imei2": "NONE", "raw": "NONE".'
)


# ---------------------------------------------------------------------------
# IMEI validation (Luhn checksum) - this is what makes IMEI verifiable
# ---------------------------------------------------------------------------
def only_digits(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())


def luhn_ok(number: str) -> bool:
    """True if `number` (a digit string) passes the Luhn checksum."""
    digits = [int(c) for c in number]
    if not digits:
        return False
    checksum = 0
    # Double every second digit from the right.
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def classify_imei(raw_value: str):
    """
    Return (clean_imei, status) where status is:
      'verified'   -> 15 digits and Luhn passes
      'unverified' -> 15 digits but Luhn fails (report but don't trust)
      'invalid'    -> not usable (wrong length / empty / NONE)
    """
    if not raw_value or str(raw_value).strip().upper() == "NONE":
        return "", "invalid"
    d = only_digits(raw_value)
    # Tolerate a 16-digit IMEISV by keeping the first 15 (the IMEI part).
    if len(d) == 16:
        d = d[:15]
    if len(d) != 15:
        return d, "invalid"
    return (d, "verified") if luhn_ok(d) else (d, "unverified")


def find_imeis_by_ocr(image: Image.Image):
    """
    Label-free fallback: scan the page for ANY 15-digit run and keep only the
    ones that pass the Luhn checksum. The checksum is what makes this safe -
    a random 15-digit number passes Luhn only ~10% of the time, so a Luhn-
    valid run is almost certainly a real IMEI, label or not.

    Returns a list of unique verified 15-digit strings, in reading order.
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=f"--dpi {OCR_DPI}")
    n = len(data["text"])

    # Concatenate the digits of tokens that sit on the same physical line, so
    # an IMEI printed with spaces/groups still forms one run - but numbers on
    # different lines never merge into a false 15-digit run.
    lines = {}
    for i in range(n):
        if not data["text"][i].strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(only_digits(data["text"][i]))

    found = []
    for parts in lines.values():
        run = "".join(parts)
        # Slide a 15-digit window across the line's digits.
        for start in range(0, max(0, len(run) - 14)):
            window = run[start:start + 15]
            if len(window) == 15 and luhn_ok(window) and window not in found:
                found.append(window)
    return found


# ---------------------------------------------------------------------------
# STEP 1 - read the IMEI(s) (structured) with Llama 4 Maverick
# ---------------------------------------------------------------------------
def prepare_image(image: Image.Image) -> Image.Image:
    """
    Raise effective resolution: upscale small scans so the shorter details
    (thin digit strokes) survive OCR and the model read. Done once at load so
    all downstream pixel coordinates stay consistent.
    """
    image = image.convert("RGB")
    w, h = image.size
    if w < UPSCALE_MIN_WIDTH:
        factor = min(UPSCALE_MAX_FACTOR, UPSCALE_MIN_WIDTH / float(w))
        if factor > 1.0:
            image = image.resize((int(w * factor), int(h * factor)), Image.LANCZOS)
    return image


def _encode_pil(image: Image.Image):
    """Return (b64, mime) for a PIL image, always as PNG (lossless)."""
    import io
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "png"


def _read_once(image: Image.Image, params: dict) -> dict:
    b64, mime = _encode_pil(image)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
        ],
    }]
    resp = _model.chat(messages=messages, params=params)
    text = resp["choices"][0]["message"]["content"].strip()

    # be tolerant of stray markdown fences or text around the JSON
    if "```" in text:
        text = text.split("```")[1].replace("json", "", 1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        return {"imei1": "NONE", "imei2": "NONE", "raw": text}


def _zoom_reread(image: Image.Image, value: str):
    """
    Pass 2: locate `value` on the page, crop it with generous padding, enlarge
    it ZOOM_FACTOR x, and re-read just that region. Returns a cleaned IMEI
    string if the zoomed read now passes Luhn, else None.
    """
    box = locate_value_box(image, value)
    if not box:
        return None
    w, h = image.size
    px, py = 40, 25
    crop = image.crop((max(0, box[0] - px), max(0, box[1] - py),
                       min(w, box[2] + px), min(h, box[3] + py)))
    if ZOOM_FACTOR and ZOOM_FACTOR != 1.0:
        crop = crop.resize((int(crop.width * ZOOM_FACTOR),
                            int(crop.height * ZOOM_FACTOR)), Image.LANCZOS)
    obj = _read_once(crop, _GREEDY_PARAMS)
    for slot in ("imei1", "imei2"):
        clean, status = classify_imei(obj.get(slot, ""))
        if status == "verified":
            return clean
    return None


def _sampled_vote(image: Image.Image, slot: str):
    """
    Pass 3: take SAMPLED_READS diverse reads and return the most common
    checksum-VERIFIED value for `slot`, or None. Sampling (temp>0 + top_p/
    top_k) is what makes the reads differ, so the vote is meaningful.
    """
    if SAMPLED_READS <= 0:
        return None
    votes = Counter()
    for _ in range(SAMPLED_READS):
        obj = _read_once(image, _SAMPLED_PARAMS)
        clean, status = classify_imei(obj.get(slot, ""))
        if status == "verified":
            votes[clean] += 1
    return votes.most_common(1)[0][0] if votes else None


def read_imeis(image: Image.Image) -> dict:
    """
    Escalation ladder. Returns {"imei1": str, "imei2": str, "raw": str}.
    Pass 1 greedy read -> pass 2 zoom re-read -> pass 3 sampled vote, each
    triggered only for a slot that has not yet passed the Luhn checksum.
    """
    base = _read_once(image, _GREEDY_PARAMS)
    raw = str(base.get("raw", "")).strip() or "NONE"
    resolved = {}

    for slot in ("imei1", "imei2"):
        clean, status = classify_imei(base.get(slot, ""))
        if status == "invalid" and not clean:
            resolved[slot] = "NONE"          # genuinely nothing in this slot
            continue
        if status == "verified":
            resolved[slot] = clean            # pass 1 already trustworthy
            continue
        # Not verified -> escalate. Try zoom, then sampled vote.
        better = _zoom_reread(image, clean or raw)
        if not better:
            better = _sampled_vote(image, slot)
        resolved[slot] = better if better else (clean or "NONE")

    # Guard: a real second IMEI must be a distinct value, never a copy of #1.
    if resolved.get("imei2") == resolved.get("imei1"):
        resolved["imei2"] = "NONE"

    # Pass 4 (label-free fallback): if a slot is still empty or unverified,
    # try Luhn-valid 15-digit numbers found anywhere on the page by OCR. This
    # rescues IMEIs printed with no label that the model returned as NONE.
    def _is_verified(v):
        return bool(v) and v != "NONE" and classify_imei(v)[1] == "verified"

    if not _is_verified(resolved.get("imei1")) or not _is_verified(resolved.get("imei2")):
        used = {v for v in resolved.values() if _is_verified(v)}
        candidates = [c for c in find_imeis_by_ocr(image) if c not in used]
        ci = iter(candidates)
        for slot in ("imei1", "imei2"):
            if not _is_verified(resolved.get(slot)):
                nxt = next(ci, None)
                if nxt:
                    resolved[slot] = nxt

    if resolved.get("imei2") == resolved.get("imei1"):
        resolved["imei2"] = "NONE"

    return {"imei1": resolved.get("imei1", "NONE"),
            "imei2": resolved.get("imei2", "NONE"), "raw": raw}


# ---------------------------------------------------------------------------
# STEP 2/3 - locate the value's pixels via OCR
# ---------------------------------------------------------------------------
def _clean(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum()).lower()


def locate_value_box(image: Image.Image, value: str):
    """Return (left, top, right, bottom) of the value on the page, or None."""
    target = _clean(value)
    if not target:
        return None

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=f"--dpi {OCR_DPI}")
    n = len(data["text"])

    # Group word indices by (block, paragraph, line) so we only join words
    # that sit on the same physical line.
    lines = {}
    for i in range(n):
        if not data["text"][i].strip():
            continue
        try:
            if float(data["conf"][i]) < 0:
                continue
        except (ValueError, TypeError):
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)

    best = None  # (ratio, box)
    for idxs in lines.values():
        # An IMEI can be split across several OCR tokens; allow a wide window.
        for start in range(len(idxs)):
            for length in range(1, 7):
                window = idxs[start:start + length]
                if not window:
                    continue
                joined = _clean("".join(data["text"][j] for j in window))
                if not joined:
                    continue
                ratio = difflib.SequenceMatcher(None, joined, target).ratio()
                if target in joined or joined in target:
                    ratio = max(ratio, 0.9)
                if best is None or ratio > best[0]:
                    lefts = [data["left"][j] for j in window]
                    tops = [data["top"][j] for j in window]
                    rights = [data["left"][j] + data["width"][j] for j in window]
                    bottoms = [data["top"][j] + data["height"][j] for j in window]
                    box = (min(lefts), min(tops), max(rights), max(bottoms))
                    best = (ratio, box)

    if best and best[0] >= MATCH_THRESHOLD:
        return best[1]
    return None


# ---------------------------------------------------------------------------
# STEP 4 - crop and save
# ---------------------------------------------------------------------------
def crop_and_save(image: Image.Image, box, out_path: Path):
    w, h = image.size
    left = max(0, box[0] - PAD_X)
    top = max(0, box[1] - PAD_Y)
    right = min(w, box[2] + PAD_X)
    bottom = min(h, box[3] + PAD_Y)
    image.crop((left, top, right, bottom)).save(out_path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    in_dir, out_dir = Path(INPUT_DIR), Path(OUTPUT_DIR)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in in_dir.iterdir()
              if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    if not images:
        print(f"No images found in {in_dir}")
        return

    results = []
    print(f"Processing {len(images)} image(s)  "
          f"(zoom x{ZOOM_FACTOR}, sampled_reads={SAMPLED_READS})...\n")

    for img_path in images:
        print(f"- {img_path.name}")
        try:
            image = prepare_image(Image.open(img_path))
            obj = read_imeis(image)
            logged_any = False

            for slot in ("imei1", "imei2"):
                clean, status = classify_imei(obj.get(slot, ""))
                if status == "invalid" and not clean:
                    continue  # this slot genuinely had no IMEI

                print(f"    {slot}: {clean or '(unreadable)'}  [{status}]")
                logged_any = True

                crop_status = "not_attempted"
                if clean:
                    box = locate_value_box(image, clean)
                    if box:
                        out_path = out_dir / f"{img_path.stem}_{slot}.png"
                        crop_and_save(image, box, out_path)
                        crop_status = "cropped"
                        print(f"        cropped -> {out_path.name}")
                    else:
                        crop_status = "value_read_but_not_located"

                results.append({
                    "file": img_path.name,
                    "slot": slot,
                    "imei": clean,
                    "verification": status,   # verified / unverified / invalid
                    "raw": obj.get("raw", ""),
                    "crop_status": crop_status,
                })

            # If neither slot produced anything, still log the miss.
            if not logged_any:
                results.append({
                    "file": img_path.name, "slot": "-", "imei": "",
                    "verification": "not_read_by_model",
                    "raw": obj.get("raw", ""), "crop_status": "not_attempted",
                })

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "file": img_path.name, "slot": "-", "imei": "",
                "verification": f"error: {e}", "raw": "", "crop_status": "-",
            })

    csv_path = out_dir / "results_imei.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "slot", "imei", "verification", "raw", "crop_status"])
        writer.writeheader()
        writer.writerows(results)

    verified = sum(1 for r in results if r["verification"] == "verified")
    print(f"\nDone. {verified} IMEI(s) checksum-verified.")
    print(f"Log written to {csv_path}")


if __name__ == "__main__":
    main()