from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session

app = FastAPI()

# Carrega o modelo de IA no servidor da nuvem
try:
    session = new_session("birefnet-portrait")
except Exception:
    session = new_session("u2net")

@app.post("/processar-foto")
async def processar_foto(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_original is None:
            raise HTTPException(status_code=400, detail="Arquivo de imagem inválido.")
            
        img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Remove o fundo na nuvem com alta precisão
        result_no_bg = remove(img_pil, session=session, alpha_matting=False)
        
        # Suaviza e alinha os contornos do cabelo
        img_arr = np.array(result_no_bg)
        rgb = img_arr[:, :, :3]
        alpha = img_arr[:, :, 3]
        
        base_size = int(max(alpha.shape) * 0.015)
        if base_size % 2 == 0:
            base_size += 1
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_size, base_size))
        alpha_eroded = cv2.erode(alpha, kernel, iterations=2)
        alpha_dilated = cv2.dilate(alpha_eroded, kernel, iterations=2)
        
        blur_size = base_size * 2 + 1
        alpha_blurred = cv2.GaussianBlur(alpha_dilated, (blur_size, blur_size), 0)
        _, alpha_smooth = cv2.threshold(alpha_blurred, 127, 255, cv2.THRESH_BINARY)
        alpha_final = cv2.GaussianBlur(alpha_smooth, (5, 5), 0)
        
        result_arr = np.dstack((rgb, alpha_final))
        final_pil = Image.fromarray(result_arr)
        
        import io
        byte_arr = io.BytesIO()
        final_pil.save(byte_arr, format='PNG')
        byte_arr.seek(0)
        
        return Response(content=byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
