import insightface
from insightface.app.common import Face
import core.globals
from core.hidePrint import HiddenPrints
from cv2 import Mat

FACE_ANALYSER = None

def get_face_analyser():
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=core.globals.providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_face_single(img_data: Mat) -> insightface.app.common.Face or None:
    with HiddenPrints():
        face = get_face_analyser().get(img_data)
        try:
            return sorted(face, key=lambda x: x.bbox[0])[0]
        except IndexError:
            return None


def get_face_many(img_data: Mat) -> list or None:
    with HiddenPrints():
        try:
            return get_face_analyser().get(img_data)
        except IndexError:
            return None

def drawOnFaces(img_data: Mat, faces: list[insightface.app.common.Face]) -> Mat:
    with HiddenPrints():
        return get_face_analyser().draw_on(img_data, faces)

def drawOnFace(img_data: Mat, face: insightface.app.common.Face) -> Mat:
    with HiddenPrints():
        return get_face_analyser().draw_on(img_data, [face])

def compareFaces(face1: Face, face2: Face, featureCount: int = 10) -> float:
    if featureCount < 1:
        featureCount = 1
    elif featureCount > 100:
        featureCount = 100 
    #Compare faces using embeddings
    sumDif: float = 0.0
    for i in featureCount:
        sumDif += abs(face1.embedding_norm[i] - face2.embedding_norm[i])
    return sumDif / featureCount