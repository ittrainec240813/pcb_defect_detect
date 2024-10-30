import base64, io, json, os
from PIL import Image
from dotenv import load_dotenv

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional

import google.generativeai as genai
from ultralytics import YOLO

from ..helpers.RAGHelper import RagModel

load_dotenv()

router = APIRouter()

model = YOLO("./pcb_defect_detect/pcb_defect_detect.pt")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
chat_model = genai.GenerativeModel('gemini-1.5-flash')
generation_config = genai.types.GenerationConfig(temperature=0.2, top_p=0.5, top_k=16)

rag_model = RagModel('gemini-1.5-flash', os.environ["GOOGLE_API_KEY"])

class ImageUploadBody(BaseModel):
    image: str

class AskRagBody(BaseModel):
    query: str
    session_id: Optional[str] = "foobar-default"

@router.get("/test")
async def test():
    if not os.path.exists("./pcb_defect_detect/val"):
        return {"msg": "test file doesn't exist"}
    image = Image.open("./pcb_defect_detect/val/0014089.jpg")
    results = model.predict(source=image)
    return {"results": json.loads(results[0].to_json())}

@router.post("/inference")
async def test_image_upload(body: ImageUploadBody):
    image = Image.open(io.BytesIO(base64.b64decode(body.image)))
    results = model.predict(source=image)
    return {"results": json.loads(results[0].to_json())}

@router.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    chat = chat_model.start_chat()
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                completion = chat.send_message(data, generation_config=generation_config)
                if (completion.parts[0].text != None):
                    # 取得生成結果
                    out = completion.parts[0].text
                else:
                    # 回覆 "Gemini沒答案!請換個說法！"
                    out = "Gemini沒答案!請換個說法！"
            except Exception as e:
                print(e)
                out = "Gemini執行出錯!請換個說法！" 

            await websocket.send_text(out)
    except WebSocketDisconnect:
        print("socket disconnected")

@router.post("/rag")
async def ask_rag(body: AskRagBody):
    answer = rag_model.query(body.query, body.session_id)
    return {"answer": answer}