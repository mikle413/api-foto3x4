import os
import threading
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import tkinter as tk
from tkinter import messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from rembg import remove, new_session

# ==========================================================
# 1. MÓDULO DE CORREÇÃO DE ILUMINAÇÃO NATURAL (SRP)
# ==========================================================
class ImageEnhancer:
    """Ajusta suavemente brilho e contraste mantendo o aspecto 100% natural."""
    
    @staticmethod
    def adjust_lighting(image_pil: Image.Image) -> Image.Image:
        enhancer_b = ImageEnhance.Brightness(image_pil)
        img_bright = enhancer_b.enhance(1.06)
        
        enhancer_c = ImageEnhance.Contrast(img_bright)
        img_natural = enhancer_c.enhance(1.04)
        
        return img_natural

# ==========================================================
# 2. MÓDULO DE REMOÇÃO DE FUNDO ACELERADA (SRP)
# ==========================================================
class BackgroundRemover:
    """Usa o motor de alta precisão com redimensionamento inteligente para velocidade relâmpago."""
    
    def __init__(self):
        try:
            self.session = new_session("birefnet-portrait")
        except Exception:
            try:
                self.session = new_session("isnet-general-use")
            except Exception:
                self.session = new_session("u2net")
            
    def remove_bg(self, image_pil: Image.Image) -> Image.Image:
        orig_width, orig_height = image_pil.size
        
        # Redimensiona temporariamente para acelerar a IA (mantém proporção)
        max_dimension = 1000
        if max(orig_width, orig_height) > max_dimension:
            image_pil.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            
        # Executa o recorte rápido na imagem menor
        result_small = remove(
            image_pil,
            session=self.session,
            alpha_matting=False
        )
        
        # Redimensiona a máscara gerada de volta para o tamanho original exato da foto do cliente
        if result_small.size != (orig_width, orig_height):
            result_small = result_small.resize((orig_width, orig_height), Image.Resampling.LANCZOS)
            
        return result_small

# ==========================================================
# 3. MÓDULO DE POLIMENTO DE CONTORNO "A BORRACHA" (SRP)
# ==========================================================
class ContourSmoother:
    """Elimina o frizz e cria uma silhueta lisa, simétrica e profissional."""
    
    @staticmethod
    def smooth_edges(image_pil: Image.Image) -> Image.Image:
        img_arr = np.array(image_pil)
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
        return Image.fromarray(result_arr)

# ==========================================================
# 4. MÓDULO GERENCIADOR DE SALVAMENTO (SRP)
# ==========================================================
class OutputManager:
    """Salva o resultado na pasta de destino final."""
    
    TARGET_DIR = r"C:\Users\cecopias\Documents\Fotos_Prontas_3x4"
    
    @classmethod
    def ensure_directory(cls):
        os.makedirs(cls.TARGET_DIR, exist_ok=True)
        
    @classmethod
    def save_photo(cls, image_pil: Image.Image, original_filepath: str) -> str:
        cls.ensure_directory()
        filename = os.path.basename(original_filepath)
        name, _ = os.path.splitext(filename)
        output_filename = f"{name}_3x4_sem_fundo.png"
        output_path = os.path.join(cls.TARGET_DIR, output_filename)
        
        image_pil.save(output_path, format="PNG")
        return output_path

# ==========================================================
# 5. MÓDULO PIPELINE COM CALLBACKS DE PROGRESSO (SRP)
# ==========================================================
class PhotoProcessingService:
    """Orquestra o fluxo informando cada etapa para a barra de progresso."""
    
    def __init__(self):
        self.enhancer = ImageEnhancer()
        self.bg_remover = BackgroundRemover()
        self.smoother = ContourSmoother()
        self.output_mgr = OutputManager()
        
    def process_file(self, filepath: str, progress_callback) -> str:
        progress_callback(20, "Ajustando iluminação natural...")
        img_original = Image.open(filepath).convert("RGB")
        img_enhanced = self.enhancer.adjust_lighting(img_original)
        
        progress_callback(50, "Removendo fundo (Modo Acelerado)...")
        # Aplica a imagem original completa combinada com o motor de alta precisão otimizado
        img_no_bg = self.bg_remover.remove_bg(img_enhanced)
        
        progress_callback(80, "Alinhando cabelo e contornos...")
        img_polished = self.smoother.smooth_edges(img_no_bg)
        
        progress_callback(95, "Salvando arquivo final...")
        output_path = self.output_mgr.save_photo(img_polished, filepath)
        
        progress_callback(100, "Concluído com sucesso!")
        return output_path

# ==========================================================
# 6. INTERFACE GRÁFICA COM BARRA DE PROGRESSO (SRP)
# ==========================================================
class AppGUI(TkinterDnD.Tk):
    """Responsável pela interface visual, drag-and-drop e barra de progresso."""
    
    def __init__(self, service: PhotoProcessingService):
        super().__init__()
        self.service = service
        
        self.title("Gráfica Rápida - Recorte Inteligente 3x4")
        self.geometry("520x440")
        self.resizable(False, False)
        self.configure(bg="#222831")
        
        self._setup_ui()
        
    def _setup_ui(self):
        lbl_title = tk.Label(
            self, 
            text="Recorte e Tratamento de Fotos 3x4", 
            font=("Segoe UI", 16, "bold"),
            bg="#222831", 
            fg="#EEEEEE"
        )
        lbl_title.pack(pady=15)
        
        self.drop_frame = tk.Frame(
            self, 
            bg="#393E46", 
            bd=2, 
            relief="ridge",
            width=460, 
            height=180
        )
        self.drop_frame.pack_propagate(False)
        self.drop_frame.pack(pady=5)
        
        self.lbl_drop = tk.Label(
            self.drop_frame,
            text="ARRASTE E SOLTE A FOTO AQUI\n\n(Ou clique para selecionar)",
            font=("Segoe UI", 12, "bold"),
            bg="#393E46",
            fg="#00ADB5",
            cursor="hand2"
        )
        self.lbl_drop.pack(expand=True, fill="both")
        
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self._on_file_drop)
        self.lbl_drop.bind("<Button-1>", lambda e: self._open_file_dialog())
        
        self.lbl_status = tk.Label(
            self, 
            text="Aguardando arquivo...", 
            font=("Segoe UI", 10),
            bg="#222831", 
            fg="#999999"
        )
        self.lbl_status.pack(pady=8)
        
        self.progress = ttk.Progressbar(
            self, 
            orient="horizontal", 
            length=460, 
            mode="determinate"
        )
        self.progress.pack(pady=5)
        
    def _on_file_drop(self, event):
        files = self.tk.splitlist(event.data)
        if files:
            filepath = files[0].strip("{}")
            self._start_processing(filepath)
            
    def _open_file_dialog(self):
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.webp")]
        )
        if filepath:
            self._start_processing(filepath)
            
    def _start_processing(self, filepath: str):
        if not filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            messagebox.showerror("Erro", "Formato de arquivo não suportado!")
            return
            
        self.lbl_drop.config(state="disabled")
        self.progress["value"] = 0
        
        threading.Thread(
            target=self._async_process, 
            args=(filepath,), 
            daemon=True
        ).start()
        
    def update_progress(self, val: int, msg: str):
        self.after(0, lambda: self.lbl_status.config(text=msg, fg="#FFD369"))
        self.after(0, lambda: self.progress.config(value=val))
        
    def _async_process(self, filepath: str):
        try:
            output_path = self.service.process_file(filepath, self.update_progress)
            self.after(0, self._on_success, output_path)
        except Exception as e:
            self.after(0, self._on_error, str(e))
            
    def _on_success(self, output_path: str):
        self.lbl_status.config(
            text="Concluído! Salvo em Fotos_Prontas_3x4", 
            fg="#00ADB5"
        )
        self.lbl_drop.config(state="normal")
        messagebox.showinfo(
            "Sucesso", 
            f"Foto processada rapidamente com alta precisão!\n\nSalva em:\n{output_path}"
        )
        
    def _on_error(self, err_msg: str):
        self.lbl_status.config(text="Erro ao processar imagem.", fg="#FF2E63")
        self.progress["value"] = 0
        self.lbl_drop.config(state="normal")
        messagebox.showerror("Erro no Processamento", f"Ocorreu um erro:\n{err_msg}")

if __name__ == "__main__":
    service = PhotoProcessingService()
    app = AppGUI(service)
    app.mainloop()