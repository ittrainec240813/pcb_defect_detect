import os, io
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

from fastapi import APIRouter
from fastapi import Request,  Header, BackgroundTasks, HTTPException, status

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage

load_dotenv()

router = APIRouter()

# 設定 Google AI API 金鑰
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 設定生成文字的參數
generation_config = genai.types.GenerationConfig(max_output_tokens=2048, temperature=0.2, top_p=0.5, top_k=16)

# 使用 Gemini-1.5-flash 模型
model = genai.GenerativeModel('gemini-1.5-flash')
chat = model.start_chat()

# 設定 Line Bot 的 API 金鑰和秘密金鑰
line_bot_api = LineBotApi(os.environ["CHANNEL_ACCESS_TOKEN"])
line_handler = WebhookHandler(os.environ["CHANNEL_SECRET"])

# 設定是否正在與使用者交談
working_status = os.getenv("DEFALUT_TALKING", default = "true").lower() == "true"

# 暫存相片
temp_image = ""

# 處理 Line Webhook 請求
@router.post("/line/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_line_signature=Header(None),
):
    # 取得請求內容
    body = await request.body()
    try:
        # 將處理 Line 事件的任務加入背景工作
        background_tasks.add_task(
            line_handler.handle, body.decode("utf-8"), x_line_signature
        )
    except InvalidSignatureError:
        # 處理無效的簽章錯誤
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "ok"

# 處理文字訊息事件
@line_handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    global working_status        
    global temp_image

    # 檢查事件類型和訊息類型
    if event.type != "message" or event.message.type != "text":
        # 回覆錯誤訊息
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="Event type error:[No message or the message does not contain text]")
        )
        
    # 檢查使用者是否輸入 "再見"
    elif event.message.text == "再見":
        # 回覆 "Bye!"
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="Bye!")
        )
        return
       
    # 檢查是否正在與使用者交談
    elif working_status:
        try: 
            # 取得使用者輸入的文字
            text_prompt = event.message.text
            prompt = [text_prompt]
            if temp_image != "":
                prompt.append(Image.open(io.BytesIO(temp_image)))
                temp_image = ""
            # 使用 Gemini 模型生成文字
            # completion = model.generate_content(prompt, generation_config=generation_config)
            completion = chat.send_message(prompt, generation_config=generation_config)
            # 檢查生成結果是否為空
            if (completion.parts[0].text != None):
                # 取得生成結果
                out = completion.parts[0].text
            else:
                # 回覆 "Gemini沒答案!請換個說法！"
                out = "Gemini沒答案!請換個說法！"
        except Exception as e:
            print(e)
            # 處理錯誤
            out = "Gemini執行出錯!請換個說法！" 
  
        # 回覆生成結果
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=out))
        
@line_handler.add(MessageEvent, message=ImageMessage)
def handel_image_message(event):
    global temp_image

    temp_image = line_bot_api.get_message_content(event.message.id).content