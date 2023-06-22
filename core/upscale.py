import shutil
from tqdm import tqdm
from codeformer.app import inference_app
from core.hidePrint import HiddenPrints
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

#Uses codeformer-pip https://github.com/kadirnar/codeformer-pip

def upscaleImage(sourcePath, outputPath, upscale_factor = 1, fidelity = 0.5):
    result = inference_app(
        image=sourcePath, 
        upscale=upscale_factor, 
        codeformer_fidelity=fidelity,
        background_enhance=True,
        face_upsample=True,
        )
    shutil.move(result, outputPath)

def upscaleReplaceImages(sourcePaths, upscale_factor = 1, fidelity = 0.5):
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    with tqdm(total=len(sourcePaths), desc="Processing", unit="frame", dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        for sourcePath in sourcePaths:
            with HiddenPrints():
                result = inference_app(
                    image=sourcePath, 
                    upscale=upscale_factor, 
                    codeformer_fidelity=fidelity,
                    background_enhance=True,
                    face_upsample=True,
                    )
            shutil.move(result, sourcePath)
            progress.update(1)

def multiUpscaleImages(sourceDir, upscale_factor = 1, fidelity = 0.5, threadCount: int = 1, progress: Any = None):
    with ThreadPoolExecutor(max_workers=threadCount) as executor:
        futures = []
        for path in sourceDir.iterdir():
            if path.is_file():
                future = executor.submit(upscaleImage, path, path, upscale_factor, fidelity)
                futures.append(future)
        executor.shutdown(wait=True)
        for future in futures:
            future.result()
            if progress:
                progress.update(1)