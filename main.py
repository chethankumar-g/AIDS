from fastapi import FastAPI, Request, Form, UploadFile, File, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from model_loading import resnet_model,xception_model,vggnet_model
from preprocess import preprocess_image
import shutil
import numpy as np
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs("static/uploads", exist_ok=True)

model_dict = {
    "resnet50": {
        "name": "ResNet50",
        "model": resnet_model,
    },
    "xception": {
        "name": "Xception",
        "model": xception_model,
    },
    "vggnet": {
        "name": "VGG NET 19",
        "model": vggnet_model,
    }
}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    print(request)
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/{model_name}", response_class=HTMLResponse)
async def show_model_page(model_name: str, request: Request):
    
    if model_name not in model_dict:
        return HTMLResponse(content="Model not found", status_code = status.HTTP_404_NOT_FOUND)
    return templates.TemplateResponse("model.html", {
        "request": request,
        "model_name": model_name,
        "image": None,
        "result": np.array([-1])
    })

@app.post("/{model_name}", response_class=HTMLResponse)
async def predict(model_name: str, request: Request, xray: UploadFile = File(...)):

    if model_name not in model_dict:
        return HTMLResponse(content="Model not found", status_code = status.HTTP_404_NOT_FOUND)
    
    upload_path = f"static/uploads/{xray.filename}"
    try:
        with open(upload_path, "x") as f:
            pass
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(xray.file, buffer)
    except Exception as e:
        print("File Error: ",e)

    processed_image = preprocess_image(upload_path)
    _result = model_dict[model_name]["model"].predict(processed_image)[0]
    result = list(map(lambda x: round(x,2), _result))
    
    return templates.TemplateResponse("model.html", {
                                        "request": request,
                                        "model_name": model_name,
                                        "image": upload_path,
                                        "result": result
                                        }
                                      )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app",port=50000,reload=True)