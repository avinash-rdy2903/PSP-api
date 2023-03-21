import numpy as np
import cv2
import tensorflow as tf
from src.utils.get_psp import get_psp_model
import uvicorn
from fastapi import FastAPI,File, UploadFile
from fastapi.responses import Response
from starlette.responses import StreamingResponse
from PIL import Image
from io import BytesIO
from src.utils.inference import inference
import matplotlib.pyplot as plt

s1 = get_psp_model(5,False)
s2 = get_psp_model(6,False)
s1_path = r"C:\Users\Avinash\Desktop\New folder\Project\models\psp stage1 720X720.h5"
s2_path = r"C:\Users\Avinash\Desktop\New folder\Project\models\psp stage2 720X720.h5"

s1.load_weights(s1_path)
s2.load_weights(s2_path)

app = FastAPI()
@app.get("/help")
def help():
    return {"framework": "Developed using FastApi, for serving tensorflow models",
            "routes":{'stage 1':"use route '/detect/s1' for predicting stage 1 of segmentation",
                    'stage 2':"use route '/detect/s2' for predicting stage 2 of segmentation"}}


@app.post("/detect/{stage}",responses = {
        200: {
            "content": {"image/png": {}}
        }
    },

    # Prevent FastAPI from adding "application/json" as an additional
    # response media type in the autogenerated OpenAPI specification.
    # https://github.com/tiangolo/fastapi/issues/3258
    response_class=Response)
async def detect(stage:str,img:UploadFile=File(...)):
    try:
        extension = img.filename.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
            return "Image must be jpg or png format!"
        if stage not in ["s2", "s1"]:
            return "Invalid stage"
        image = (Image.open(BytesIO(await img.read())))
        # image.show()
        image = np.asarray(image)       
        
        pred = inference(s1 if stage=="s1" else s2,image)
        # print(np.unique(pred))
        bytes_img = cv2.imencode('.png',pred)[1].tobytes()
        
        return Response(content=bytes_img, media_type="image/png")
    finally:
        img.file.close()

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8080,debug=True)
