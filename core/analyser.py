import insightface
import core.globals
from core.hidePrint import HiddenPrints

FACE_ANALYSER = None

def get_face_analyser():
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=core.globals.providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_face_single(img_data) -> insightface.app.common.Face or None:
    with HiddenPrints():
        face = get_face_analyser().get(img_data)
        try:
            return sorted(face, key=lambda x: x.bbox[0])[0]
        except IndexError:
            return None


def get_face_many(img_data) -> list or None:
    with HiddenPrints():
        try:
            return get_face_analyser().get(img_data)
        except IndexError:
            return None
