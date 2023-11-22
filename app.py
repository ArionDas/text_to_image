import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Creating the app
app = tk.Tk()
app.geometry("540x630")
app.title("Text To Image")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(master=app, height=40, width=512, text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, torch_dtype=torch.float16, use_auth_token=auth_token)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

def generate():
    with autocast(device):
        image = pipe(prompt).images[0]
    
    image.save('generatedImage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)

trigger = ctk.CTkButton(master=app, height=40, width=120, text_color="white", fg_color="blue", text="Generate")
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()