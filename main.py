import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from llama_cpp import Llama
import uvicorn

app = FastAPI()

MODEL_NAME = "model.gguf"
# लिंक के आखिर में ?download=true लगा दिया है ताकि असली फाइल ही डाउनलोड हो
MODEL_URL = "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q4_k_m.gguf?download=true"

# सर्वर चालू होते ही असली मॉडल डाउनलोड करेगा
if not os.path.exists(MODEL_NAME) or os.path.getsize(MODEL_NAME) < 1000000: # 1MB से छोटी फाइल को नकली मानेगा
    print("असली मॉडल डाउनलोड हो रहा है, कृपया 1-2 मिनट प्रतीक्षा करें...")
    headers = {'User-Agent': 'Mozilla/5.0'} # HuggingFace को लगेगा कोई असली ब्राउज़र है
    response = requests.get(MODEL_URL, stream=True, allow_redirects=True, headers=headers)
    
    if response.status_code == 200:
        with open(MODEL_NAME, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"डाउनलोड पूरा हुआ! फाइल का साइज: {os.path.getsize(MODEL_NAME) / (1024*1024):.2f} MB")
    else:
        print(f"डाउनलोड फेल! Status code: {response.status_code}")

# रेंडर की 512MB RAM के लिए सेटिंग्स
llm = Llama(model_path=MODEL_NAME, n_ctx=512, n_threads=2)

@app.post("/ask")
async def ask_brain(request: Request):
    try:
        data = await request.json()
        user_message = data.get("question", "")
        
        if not user_message:
            return JSONResponse(content={"error": "सवाल खाली है!"}, status_code=400)

        prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        output = llm(
            prompt,
            max_tokens=150,
            stop=["<|im_end|>"],
            echo=False
        )
        
        answer = output['choices'][0]['text'].strip()
        return {"reply": answer}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
