
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from rembg import remove, new_session
import numpy as np
import cv2
from PIL import Image
import io
import threading

app = FastAPI()

session = None
session_lock = threading.Lock()


def get_session():
    global session

    if session is None:
        with session_lock:
            if session is None:
                try:
                    session = new_session("birefnet-portrait")
                except Exception as e:
                    print(
                        f"Falha ao carregar birefnet-portrait: {e}",
                        flush=True
                    )
                    session = new_session("u2net")

    return session


@app.get("/")
def health():
    return {"status": "online"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/processar-foto")
async def processar_foto(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_original is None:
            raise HTTPException(
                status_code=400,
                detail="Arquivo de imagem inválido."
            )

        img_rgb = cv2.cvtColor(
            img_original,
            cv2.COLOR_BGR2RGB
        )
        img_pil = Image.fromarray(img_rgb)

        modelo = get_session()

        result_no_bg = remove(
            img_pil,
            session=modelo,
            alpha_matting=False
        )

        img_arr = np.array(result_no_bg)

        if img_arr.ndim != 3 or img_arr.shape[2] < 4:
            raise RuntimeError(
                "O modelo não retornou uma imagem com canal alfa."
            )

        rgb = img_arr[:, :, :3]
        alpha = img_arr[:, :, 3]

        base_size = max(3, int(max(alpha.shape) * 0.015))

        if base_size % 2 == 0:
            base_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (base_size, base_size)
        )

        alpha_eroded = cv2.erode(
            alpha, kernel, iterations=2
        )
        alpha_dilated = cv2.dilate(
            alpha_eroded, kernel, iterations=2
        )

        blur_size = base_size * 2 + 1

        alpha_blurred = cv2.GaussianBlur(
            alpha_dilated,
            (blur_size, blur_size),
            0
        )

        _, alpha_smooth = cv2.threshold(
            alpha_blurred,
            127,
            255,
            cv2.THRESH_BINARY
        )

        alpha_final = cv2.GaussianBlur(
            alpha_smooth, (5, 5), 0
        )

        result_arr = np.dstack((rgb, alpha_final))
        final_pil = Image.fromarray(result_arr)

        byte_arr = io.BytesIO()
        final_pil.save(byte_arr, format="PNG")

        return Response(
            content=byte_arr.getvalue(),
            media_type="image/png"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Erro ao processar foto: {e}", flush=True)
        raise HTTPException(
            status_code=500,
            detail="Erro ao processar a imagem."
        )
