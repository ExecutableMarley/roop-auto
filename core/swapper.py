import os
from tqdm import tqdm
import cv2
from cv2 import Mat
import insightface
from insightface.model_zoo.inswapper import INSwapper
from insightface.app.common import Face
import core.globals
from core.analyser import get_face_single, get_face_many, drawOnFace
from core.hidePrint import HiddenPrints
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

FACE_SWAPPER = None


def get_face_swapper() -> INSwapper:
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../inswapper_128.onnx')
        FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=core.globals.providers)
    return FACE_SWAPPER


def swap_face_in_frame(source_face: Face, target_face: Face, frame:Mat):
    if target_face:
        return get_face_swapper().get(frame, target_face, source_face, paste_back=True)
    return frame


def process_faces(source_face: Face, frame: Mat, all_faces=False):
    if all_faces:
        many_faces = get_face_many(frame)
        if many_faces:
            for face in many_faces:
                frame = swap_face_in_frame(source_face, face, frame)
    else:
        face = get_face_single(frame)
        if face:
            frame = swap_face_in_frame(source_face, face, frame)
    return frame

def processFramesMany(sourceFace: Face, frame_paths: list):
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'

    with tqdm(total=len(frame_paths), desc="Processing", unit="frame", dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        for frame_path in frame_paths:
            frame: Mat = cv2.imread(frame_path)
            try:
                with HiddenPrints():
                    result = process_faces(sourceFace, frame, core.globals.all_faces)
                cv2.imwrite(frame_path, result)
            except Exception:
                progress.set_postfix(status='E', refresh=True)
                pass
            progress.update(1)
        progress.set_postfix(desc="Done", refresh=True)

def processFrame(sourceFace: Face, frame_path: str, output_file: str):
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    with tqdm(total=1, desc="Processing", unit="frame", dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        frame: Mat = cv2.imread(frame_path)
        with HiddenPrints():
            result = process_faces(sourceFace, frame, core.globals.all_faces)
        cv2.imwrite(output_file, result)
        progress.update(1)

def processFrame2(sourceFace: Face, frame_path: str, output_file: str, progress: Any = None):
    frame: Mat = cv2.imread(frame_path)
    with HiddenPrints():
        result = process_faces(sourceFace, frame, core.globals.all_faces)
    cv2.imwrite(output_file, result)
    if progress:
        progress.update(1)

def multiProcessFramesMany(sourceFace: Face, temp_frame_paths: List[str], threadCount: int = 1, progress: Any = None) -> None:
    with ThreadPoolExecutor(max_workers=threadCount) as executor:
        futures = []
        for path in temp_frame_paths:
            future = executor.submit(processFrame2, sourceFace, path, path, progress)
            futures.append(future)
        executor.shutdown(wait=True)
        for future in futures:
            future.result()