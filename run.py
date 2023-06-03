#!/usr/bin/env python3

import shlex
import os
import logging
import sys

commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
sys.argv += shlex.split(commandline_args)

import installer
installer.ensure_base_requirements()
installer.add_args()
installer.parse_args()
installer.setup_logging(False)
if __name__ == "__main__":
    installer.run_setup()
    installer.log.debug('Starting')

log = logging.getLogger("roop")

import platform
import signal
import shutil
import glob
import argparse
import multiprocessing as mp
import torch
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import asksaveasfilename
import webbrowser
import psutil
import cv2
import threading
import numpy as np
from PIL import Image, ImageTk
import core.globals
from core.swapper import processFrame, processFramesMany, process_video_gpu
from core.utils import is_img, is_video, detect_fps, set_fps, create_video, add_audio, extract_frames, rreplace
from core.analyser import get_face_single
from core.upscale import upscaleReplaceImages, upscaleImage
import insightface

if 'ROCMExecutionProvider' in core.globals.providers:
    del torch
 
pool = None
args = {}

signal.signal(signal.SIGINT, lambda signal_number, frame: quit())
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--face', help='Source Face path', dest='source_img')
parser.add_argument('-t', '--target', help='Target image/video Path', dest='target_path')
parser.add_argument('-o', '--output', help='Output File Path', dest='output_file')
parser.add_argument('--gpu', help='Force use gpu', dest='gpu', action='store_true', default=False)
parser.add_argument('--cpu', help='Force use cpu', dest='cpu', action='store_true', default=False)
parser.add_argument('--keep-fps', help='maintain original fps', dest='keep_fps', action='store_true', default=False)
parser.add_argument('--keep-frames', help='keep frames directory', dest='keep_frames', action='store_true', default=False)
parser.add_argument('--max-memory', help='maximum amount of RAM in GB to be used', type=int)
parser.add_argument('--max-cores', help='number of cores to be use for CPU mode', dest='cores_count', type=int, default=max(psutil.cpu_count() - 2, 2))
parser.add_argument('--all-faces', help='swap all faces in frame', dest='all_faces', action='store_true', default=False)
parser.add_argument('--debug', help='enable debug mode', dest='debug', action='store_true', default=False)
parser.add_argument('--upscale', help='upscale images', dest='upscale', action='store_true', default=False)
parser.add_argument('--gpu-threads', help='number of threads for gpu to run in parallel', dest='gpu_threads', type=int, default=4)

for name, value in vars(parser.parse_args()).items():
    args[name] = value

logging.disable(logging.NOTSET if args['debug'] else logging.DEBUG)

if not args['debug']:
    np.warnings.filterwarnings('ignore')

sep = "/"
if os.name == "nt":
    sep = "\\"


def limit_resources():
    if args['max_memory']:
        memory = args['max_memory'] * 1024 * 1024 * 1024
        memory = 4 * 1024 * 1024 * 1024
        if str(platform.system()).lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check():
    if sys.version_info < (3, 9):
        quit('Python version is not supported - please upgrade to 3.9 or higher')
    if not shutil.which('ffmpeg'):
        quit('ffmpeg is not installed!')
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'inswapper_128.onnx')
    if not os.path.isfile(model_path):
        quit('File "inswapper_128.onnx" does not exist!')
    
    if args['cpu']:
        log.info('Forcing CPU mode')
        args['gpu'] = False
    #Check if GPU is available
    elif torch.cuda.is_available() and core.globals.getOnnxruntimeDevice() == 'GPU':
        log.info('GPU available. Using GPU mode')
        args['gpu'] = True
    else:
        log.info('GPU not available. Defaulting to CPU mode')
        args['gpu'] = False

def preview_image(image_path):
    img = Image.open(image_path)
    img = img.resize((180, 180), Image.LANCZOS)
    photo_img = ImageTk.PhotoImage(img)
    left_frame = tk.Frame(window)
    left_frame.place(x=60, y=100)
    img_label = tk.Label(left_frame, image=photo_img)
    img_label.image = photo_img
    img_label.pack()


def preview_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((180, 180), Image.LANCZOS)
        photo_img = ImageTk.PhotoImage(img)
        right_frame = tk.Frame(window)
        right_frame.place(x=360, y=100)
        img_label = tk.Label(right_frame, image=photo_img)
        img_label.image = photo_img
        img_label.pack()

    cap.release()


def select_face():
    args['source_img'] = filedialog.askopenfilename(title="Select a face")
    if args['source_img'] == None or args['source_img'] == '':
        return
    preview_image(args['source_img'])

def select_target():
    args['target_path'] = filedialog.askopenfilename(title="Select a target")
    if args['target_path'] == None or args['target_path'] == '':
        return
    threading.Thread(target=preview_video, args=(args['target_path'],)).start()

def selectDirectory():
    args['target_path'] = filedialog.askdirectory(title="Select a Directory")
    if args['target_path'] == None or args['target_path'] == '':
        return
    try:
        first_file = os.listdir(args['target_path'])[0]
        if is_img(first_file) and False:
            threading.Thread(target=preview_image, args=(args['target_path'] + "/" +  first_file,)).start()
        elif is_video(first_file):
            threading.Thread(target=preview_video, args=(args['target_path'] + "/" +  first_file,)).start()
    except:
        print("No files in directory")

def toggle_fps_limit():
    args['keep_fps'] = int(limit_fps.get() != True)


def toggle_all_faces():
    core.globals.all_faces = True if all_faces.get() == 1 else False


def toggle_keep_frames():
    args['keep_frames'] = int(keep_frames.get())

def toogle_upscale():
    args['upscale'] = int(upscale_check.get())

def save_file():
    filename, ext = 'output.mp4', '.mp4'
    if is_img(args['target_path']):
        filename, ext = 'output.png', '.png'
    if os.path.isdir(args['target_path']):
        
        newDir = args['target_path'] + "/swapped"
        Path(newDir).mkdir(exist_ok=True)
        
        return
    args['output_file'] = asksaveasfilename(initialfile=filename, defaultextension=ext, filetypes=[("All Files","*.*"),("Videos","*.mp4")])


def status(string):
    if 'cli_mode' in args:
        print("Status: " + string)
    else:
        status_label["text"] = "Status: " + string
        window.update()

def start_processing(sourceFace, frame_paths, fps):
    n = len(frame_paths)//(args['cores_count'])
    
    #if args['gpu']:
    #    process_video_gpu(sourceFace, args['target_path'], args['output_file'], fps, int(args['gpu_threads']), core.globals.all_faces)
    #    return
        
    #Single Threaded
    if len(frame_paths) < args['cores_count'] or args['gpu'] or is_img(args['target_path']):
        processFramesMany(sourceFace, args["frame_paths"])
        return
    #Multi Threading for CPU only
    else:
        processes = []
        for i in range(0, len(frame_paths), n):
            p = pool.apply_async(processFramesMany, args=(sourceFace, frame_paths[i:i+n],))
            processes.append(p)
        for p in processes:
            p.get()
        pool.close()
        pool.join()

def sanity_check() -> bool:
    if not args['source_img'] or not os.path.isfile(args['source_img']):
        print("\n[WARNING] Please select an image containing a face.")
        status("No Source Image Selected")
        return False
    elif not args['target_path'] or not (os.path.isfile(args['target_path']) or os.path.isdir(args['target_path']) ):
        print("\n[WARNING] Please select a video/image to swap face in.")
        status("No Target File Selected")
        return False
    if not args['output_file']:
        target_path = args['target_path']
        args['output_file'] = rreplace(target_path, "/", "/swapped-", 1) if "/" in target_path else "swapped-" + target_path
    test_face = get_face_single(cv2.imread(args['source_img']))
    if not test_face:
        print("\n[WARNING] No face detected in source image. Please try with another one.\n")
        return False
    return True

def processImage(sourceFace, target_path, output_file=None):
    if not output_file:
        output_file = target_path + "/swapped-" + target_path.split("/")[-1]
    processFrame(sourceFace, target_path, output_file)
    if args['upscale']:
        upscaleImage(output_file, output_file, 1, 0.5)

def processVideo(sourceFace, target_path, output_file=None):
    if not output_file:
        output_file = target_path + "/swapped-" + target_path.split("/")[-1]
    video_name_full = target_path.split("/")[-1]
    video_name = os.path.splitext(video_name_full)[0]
    output_dir = os.path.dirname(target_path) + "/" + video_name if os.path.dirname(target_path) else video_name
    if not output_file:
        output_file = output_dir + "/swapped-" + video_name + ".mp4"
    
    Path(output_dir).mkdir(exist_ok=True)
    status("detecting video's FPS...")
    fps, exact_fps = detect_fps(target_path)
    if not args['keep_fps'] and fps > 30:
        this_path = output_dir + "/" + video_name + ".mp4"
        set_fps(target_path, this_path, 30)
        target_path, exact_fps = this_path, 30
    else:
        shutil.copy(target_path, output_dir)
    status("extracting frames...")
    extract_frames(target_path, output_dir)
    args['frame_paths'] = tuple(sorted(
        glob.glob(output_dir + "/*.png"),
        key=lambda x: int(x.split(sep)[-1].replace(".png", ""))
    ))
    status("swapping in progress...")
    start_processing(sourceFace, args["frame_paths"], fps)
    status("creating video...")
    
    if args['upscale']:
        upscaleReplaceImages(args['frame_paths'], 1, 0.5)
    
    create_video(video_name, exact_fps, output_dir)
    status("adding audio...")
    add_audio(output_dir, target_path, video_name_full, args['keep_frames'], output_file)
    #save_path = output_dir + "/" + video_name + ".mp4"
    print("\n\nVideo saved as:", output_file, "\n\n")
    status("swap successful!")


currently_processing = False
def processData():
    global currently_processing
    if currently_processing:
        return
    currently_processing = True
    
    if not sanity_check():
        currently_processing = False
        return
    
    sourceFace = get_face_single(cv2.imread(args['source_img']))
    
    target_path = args['target_path']
    if is_img(target_path):
        processFrame(sourceFace, target_path, args['output_file'])
        if args['upscale']:
            upscaleImage(args['output_file'], args['output_file'])
        status("swap successful!")
    elif is_video(target_path):
        processVideo(sourceFace, target_path, args['output_file'])
    elif os.path.isdir(target_path):
        #Create output directory
        Path(args['output_file']).mkdir(exist_ok=True)
        #Get all files in directory
        files = os.listdir(target_path)
        for file in files:
            if is_img(file):
                processFrame(sourceFace, target_path + "/" + file, args['output_file'] + "/" + file)
            elif is_video(file):
                processVideo(sourceFace, target_path + "/" + file)
            else:
                pass
        status("swap successful!")
    else:
        print("\n[WARNING] Please select a video/image/Directory to swap face in.")
        status("Unknown File Type")
    currently_processing = False

if __name__ == "__main__":
    global status_label, window

    log.info('roop using device: ' + core.globals.getOnnxruntimeDevice())

    pre_check()
    limit_resources()

    if args['source_img']:
        args['cli_mode'] = True
        processData()
        quit()
    window = tk.Tk()
    window.geometry("600x700")
    window.title("roop-auto")
    window.configure(bg="#2d3436")
    window.resizable(width=False, height=False)

    # Contact information
    support_link = tk.Label(window, text="Donate to project <3", fg="#fd79a8", bg="#2d3436", cursor="hand2", font=("Arial", 8))
    support_link.place(x=180,y=20,width=250,height=30)
    support_link.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/sponsors/s0md3v"))

    # Select a face button
    face_button = tk.Button(window, text="Select a face", command=select_face, bg="#2d3436", fg="#74b9ff", highlightthickness=4, relief="flat", highlightbackground="#74b9ff", activebackground="#74b9ff", borderwidth=4)
    face_button.place(x=60,y=320,width=180,height=80)

    # Select a target button
    target_button = tk.Button(window, text="Select a target", command=select_target, bg="#2d3436", fg="#74b9ff", highlightthickness=4, relief="flat", highlightbackground="#74b9ff", activebackground="#74b9ff", borderwidth=4)
    target_button.place(x=360,y=320,width=180,height=80)

    # Select a directory button
    directory_button = tk.Button(window, text="Select a directory", command=selectDirectory, bg="#2d3436", fg="#74b9ff", highlightthickness=4, relief="flat", highlightbackground="#74b9ff", activebackground="#74b9ff", borderwidth=4)
    directory_button.place(x=360,y=420,width=180,height=80)

    # All faces checkbox
    all_faces = tk.IntVar()
    all_faces_checkbox = tk.Checkbutton(window, anchor="w", relief="groove", activebackground="#2d3436", activeforeground="#74b9ff", selectcolor="black", text="Process all faces in frame", fg="#dfe6e9", borderwidth=0, highlightthickness=0, bg="#2d3436", variable=all_faces, command=toggle_all_faces)
    all_faces_checkbox.place(x=60,y=500,width=240,height=31)

    # FPS limit checkbox
    limit_fps = tk.IntVar(None, not args['keep_fps'])
    fps_checkbox = tk.Checkbutton(window, anchor="w", relief="groove", activebackground="#2d3436", activeforeground="#74b9ff", selectcolor="black", text="Limit FPS to 30", fg="#dfe6e9", borderwidth=0, highlightthickness=0, bg="#2d3436", variable=limit_fps, command=toggle_fps_limit)
    fps_checkbox.place(x=60,y=475,width=240,height=31)

    upscale_check = tk.IntVar(None, args['upscale'])
    upscale_checkbox = tk.Checkbutton(window, anchor="w", relief="groove", activebackground="#2d3436", activeforeground="#74b9ff", selectcolor="black", text="Upscale", fg="#dfe6e9", borderwidth=0, highlightthickness=0, bg="#2d3436", variable=upscale_check, command=toogle_upscale)
    upscale_checkbox.place(x=360,y=500,width=240,height=31)

    # Keep frames checkbox
    keep_frames = tk.IntVar(None, args['keep_frames'])
    frames_checkbox = tk.Checkbutton(window, anchor="w", relief="groove", activebackground="#2d3436", activeforeground="#74b9ff", selectcolor="black", text="Keep frames dir", fg="#dfe6e9", borderwidth=0, highlightthickness=0, bg="#2d3436", variable=keep_frames, command=toggle_keep_frames)
    frames_checkbox.place(x=60,y=450,width=240,height=31)

    # Start button
    start_button = tk.Button(window, text="Start", bg="#f1c40f", relief="flat", borderwidth=0, highlightthickness=0, command=lambda: [save_file(), threading.Thread(target=processData).start()])
    start_button.place(x=240,y=560,width=120,height=49)
    
    # Status label
    status_label = tk.Label(window, width=580, justify="center", text="Status: waiting for input...", fg="#2ecc71", bg="#2d3436")
    status_label.place(x=10,y=640,width=580,height=30)

    window.mainloop()
