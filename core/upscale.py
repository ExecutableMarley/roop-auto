import shutil
from tqdm import tqdm
from codeformer.app import inference_app

#Uses codeformer-pip https://github.com/kadirnar/codeformer-pip

def upscaleImage(sourcePath, outputPath, upscale_factor = 2, fidelity = 0.5):
    result = inference_app(
        image=sourcePath, 
        upscale=upscale_factor, 
        codeformer_fidelity=fidelity,
        background_enhance=True,
        face_upsample=True,
        )
    shutil.move(result, outputPath)

def upscaleReplaceImages(sourcePaths, upscale_factor = 2, fidelity = 0.5):
    for sourcePath in sourcePaths:
        result = inference_app(
            image=sourcePath, 
            upscale=upscale_factor, 
            codeformer_fidelity=fidelity,
            background_enhance=True,
            face_upsample=True,
            )
        shutil.move(result, sourcePath)

def upscaleVideo(sourcePath, outputPath, upscale_factor = 2, fidelity = 0.5):
    result = inference_app(
        video=sourcePath, 
        upscale=upscale_factor, 
        codeformer_fidelity=fidelity,
        background_enhance=True,
        face_upsample=True,
        )
    shutil.move(result, outputPath)