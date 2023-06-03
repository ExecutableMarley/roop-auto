import os
from tqdm import tqdm
import cv2
from cv2 import Mat
import insightface
from insightface.model_zoo.inswapper import INSwapper
from insightface.app.common import Face
import core.globals
from core.analyser import get_face_single, get_face_many
from core.hidePrint import HiddenPrints



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
    print("\n\nImage saved as:", output_file, "\n\n")


from threading import Thread

#creates a thread and returns value when joined
class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
    
def process_video_gpu(sourceFace: Face, source_video:str, outputfile:str, fps, gpu_threads, all_faces: bool=False):
    #opening input video for read
    cap = cv2.VideoCapture(source_video)
    #opening output video for writing
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter( os.path.join(outputfile, "output.mp4"), fourcc, fps, (width, height))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temp = []
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    with tqdm(total=frame_count, desc='Processing', unit="frame", dynamic_ncols=True, bar_format=bar_format) as progress:
        while True:
            #getting frame
            ret, frame = cap.read()
            if not ret:
                break
            #we are having an array of length %gpu_threads%, running in parallel
            #so if array is equal or longer than gpu threads, waiting 
            while len(temp) >= gpu_threads:
                #we are order dependent, so we are forced to wait for first element to finish. When finished removing thread from the list
                swappedFrame = temp.pop(0).join()
                output_video.write(swappedFrame)
                progress.update(1)
            #adding new frame to the list and starting it 
            temp.append(ThreadWithReturnValue(target=process_faces, args=(sourceFace,frame, all_faces)))
            temp[-1].start()
        progress.set_postfix(desc="Done", refresh=True)