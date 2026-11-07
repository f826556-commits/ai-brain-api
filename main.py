import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from llama_cpp import Llama
import uvicorn

app = FastAPI()

# मॉडल का नाम और डाउनलोड लिंक (HuggingFace से)
MODEL_NAME = "model.gguf"
MODEL_URL = "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q4_k_m.gguf"

# सर्वर चालू होते ही यह चेक करेगा कि मॉडल है या नहीं, नहीं है तो डाउनलोड करेगा
if not os.path.exists(MODEL_NAME):
    print("मॉडल डाउनलोड हो रहा है, कृपया प्रतीक्षा करें...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_NAME, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("डाउनलोड पूरा हुआ!")

# रेंडर की 512MB RAM के लिए सेटिंग्स (n_ctx कम रखा है ताकि क्रैश न हो)
llm = Llama(model_path=MODEL_NAME, n_ctx=512, n_threads=2)

@app.post("/ask")
async def ask_brain(request: Request):
    try:
        data = await request.json()
        user_message = data.get("question", "")
        
        if not user_message:
            return JSONResponse(content={"error": "सवाल खाली है!"}, status_code=400)

        # AI के लिए प्रॉम्प्ट 
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

# Render के लिए पोर्ट
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
  
