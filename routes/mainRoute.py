import base64, io, json, os
from PIL import Image
from dotenv import load_dotenv

from fastapi import APIRouter
from fastapi import Request,  Header, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ultralytics import YOLO

load_dotenv()

router = APIRouter()

model = YOLO("./pcb_defect_detect/pcb_defect_detect.pt")

class ImageUploadBody(BaseModel):
    image: str

@router.get("/test")
async def test():
    if not os.path.exists("./pcb_defect_detect/val"):
        return {"msg": "test file doesn't exist"}
    image = Image.open("./pcb_defect_detect/val/0014089.jpg")
    results = model.predict(source=image, conf=0.5)
    return {"results": json.loads(results[0].to_json())}

@router.post("/inference")
async def test_image_upload(body: ImageUploadBody):
    image = Image.open(io.BytesIO(base64.b64decode(body.image)))
    results = model.predict(source=image, conf=0.5)
    return {"results": json.loads(results[0].to_json())}