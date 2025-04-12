from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
import io

# Set the Tesseract executable path (update accordingly)
pytesseract.pytesseract.tesseract_cmd = r"model\Tesseract-OCR\tesseract.exe"  # Change path if needed

app = FastAPI()

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    try:
        # Read the image contents and open it using PIL
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Extract text using pytesseract
        extracted_text = pytesseract.image_to_string(image)
        
        return {"extracted_text": extracted_text}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)