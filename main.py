import os
import io
import json
import csv
import re
import traceback
from pathlib import Path

import pdfplumber
import anthropic
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

app = FastAPI()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "")
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")

STATIC_DIR = Path(__file__).parent / "static"

SYSTEM_PROMPT = """You are a financial assistant helping a prop stylist categorize credit card expenses.

You will receive raw text from a credit card statement. Extract all individual purchase transactions and classify each one.

Categories (use EXACTLY these strings):
- Food — restaurants, cafes, grocery stores, food delivery, bars
- Transportation — Lyft, Uber, airlines, taxis, trains, transit
- Supplies — any store purchase, Amazon, hardware, props, clothing, cleaners, subscriptions for tools/software
- Not Relevant — bank fees, annual membership fees, payments, interest, credits, personal services (therapy, massage), entertainment tickets, streaming subscriptions for personal use (Netflix, Apple), taxes

Return ONLY a JSON array. No markdown, no explanation. Each item:
{"date":"MM/DD","merchant":"Cleaned merchant name","amount":12.34,"category":"Food"}

Exclude payment credits (negative amounts from "Payment Thank You").
Round amounts to 2 decimal places."""


def extract_text_from_pdf(data: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def extract_text_from_csv(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = [",".join(row) for row in reader]
    return "\n".join(rows)


def categorize_with_claude(text: str) -> list:
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured on server.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Here is the credit card statement text:\n\n{text}"}],
    )

    raw = "".join(block.text for block in message.content if hasattr(block, "text"))
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)


def get_sheets_service():
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise HTTPException(status_code=500, detail="Google service account not configured on server.")
    try:
        creds_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Service account JSON is malformed: {e}")
    try:
        creds = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        return build("sheets", "v4", credentials=creds, cache_discovery=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not initialize Google Sheets client: {e}")


# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def get_config():
    sheet_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/edit" if GOOGLE_SHEET_ID else ""
    return JSONResponse(content={"sheet_url": sheet_url})


@app.post("/api/process")
async def process_statement(files: list[UploadFile] = File(...)):
    all_transactions = []

    for file in files:
        data = await file.read()
        filename = file.filename or ""

        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(data)
        elif filename.lower().endswith(".csv"):
            text = extract_text_from_csv(data)
        else:
            raise HTTPException(status_code=400, detail=f"{filename}: only PDF and CSV files are supported.")

        if not text.strip():
            raise HTTPException(status_code=422, detail=f"{filename}: could not extract any text.")

        transactions = categorize_with_claude(text)
        all_transactions.extend(transactions)

    return JSONResponse(content={"transactions": all_transactions})


@app.post("/api/export")
async def export_to_sheets(payload: dict):
    transactions = payload.get("transactions", [])
    if not transactions:
        raise HTTPException(status_code=400, detail="No transactions provided.")

    if not GOOGLE_SHEET_ID:
        raise HTTPException(status_code=500, detail="Google Sheet ID not configured on server.")

    try:
        service = get_sheets_service()
        sheet = service.spreadsheets()

        header = [["Date", "Merchant", "Amount", "Category"]]
        rows = [[t.get("date", ""), t.get("merchant", ""), str(t.get("amount", "")), t.get("category", "")] for t in transactions]
        values = header + rows

        # Clear the full A:D range so stale rows from a larger prior export don't linger
        sheet.values().clear(spreadsheetId=GOOGLE_SHEET_ID, range="A:D").execute()
        sheet.values().update(
            spreadsheetId=GOOGLE_SHEET_ID,
            range="A1",
            valueInputOption="USER_ENTERED",
            body={"values": values},
        ).execute()

        return JSONResponse(content={"success": True, "rows_written": len(rows)})

    except HTTPException:
        raise
    except HttpError as e:
        traceback.print_exc()
        status = getattr(e.resp, "status", 500)
        try:
            err_body = json.loads(e.content.decode("utf-8"))
            msg = err_body.get("error", {}).get("message") or str(e)
        except Exception:
            msg = str(e)
        if status == 403:
            msg = f"Google Sheets denied access (403). Make sure the service account email has Editor access to the sheet. Underlying error: {msg}"
        elif status == 404:
            msg = f"Sheet not found (404). Check that GOOGLE_SHEET_ID is correct. Underlying error: {msg}"
        raise HTTPException(status_code=502, detail=f"Google Sheets API error: {msg}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected export error: {type(e).__name__}: {e}")


# Serve static files (fallback)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
