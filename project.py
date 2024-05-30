from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
from diffusers import AutoPipelineForText2Image
pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
from fastapi import FastAPI, Response, Request, Query
app = FastAPI()
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from fastapi.responses import StreamingResponse, HTMLResponse
import io
from pydantic import BaseModel
class Item(BaseModel):
    inp: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <body>
            <form action="/predict/" method="get">
                <input type="text" name="inp" placeholder="Введите текст">
                <input type="submit" value="Подтвердить">
            </form>
        </body>
    </html>
    """

@app.get("/predict/")
def predict(inp: str = Query(...)):
    input_ids = tokenizer(inp, return_tensors='pt').input_ids
    outputs = model.generate(input_ids=input_ids)
    a = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    prompt = a
    negative_prompt = "low quality, bad quality"
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, prior_guidance_scale =1.0, height=768, width=768).images[0]
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return StreamingResponse(img_io, media_type="image/jpeg")

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)