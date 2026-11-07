import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from llama_cpp import Llama
import uvicorn

app = FastAPI()

MODEL_NAME = "model.gguf"
MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

if not os.path.exists(MODEL_NAME) or os.path.getsize(MODEL_NAME) < 1000000:
    print("SK Rajaa का दिमाग डाउनलोड हो रहा है (380 MB)...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(MODEL_URL, stream=True, allow_redirects=True, headers=headers)
    if response.status_code == 200:
        with open(MODEL_NAME, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("✅ डाउनलोड पूरा हुआ!")
    else:
        print("❌ डाउनलोड फेल!")

# 8GB RAM है, इसलिए हम 4 थ्रेड्स इस्तेमाल करेंगे ताकि स्पीड दोगुनी हो जाए
llm = Llama(model_path=MODEL_NAME, n_ctx=2048, n_threads=4)

@app.get("/")
async def home():
    return {"status": "SK Rajaa का दिमाग 100% एक्टिव है!"}

@app.post("/ask")
async def ask_brain(request: Request):
    try:
        data = await request.json()
        user_message = data.get("question", "")
        
        if not user_message:
            return JSONResponse(content={"error": "सवाल खाली है!"}, status_code=400)

        system_prompt = "तुम एक बहुत ही होशियार और मददगार AI असिस्टेंट हो। तुम्हारा नाम 'SK Rajaa' है। तुम हिंदी और इंग्लिश दोनों समझते हो और सटीक जवाब देते हो।"
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        output = llm(prompt, max_tokens=300, stop=["<|im_end|>"], echo=False)
        answer = output['choices'][0]['text'].strip()
        return {"reply": answer}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
        
