"""
Taka Scanner - Streamlit web app (live webcam + photo mode) for Bangladeshi banknotes.

GitHub repo layout (Streamlit Community Cloud):
    app.py                          <- this file
    requirements.txt                <- libraries (see the version that goes with this file)
    best_banknote_model.onnx        <- your model (recommended: light + fast on CPU)
    best_banknote_model.meta.json   <- optional (created by taka_realtime.py); defaults are built in
    best_banknote_model.pth         <- optional fallback if there is no .onnx (needs torch + timm)

Run locally:   pip install -r requirements.txt   then   streamlit run app.py
Deploy:        share.streamlit.io -> New app -> pick repo/branch -> Main file path: app.py
"""
import json
import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

# ------------------------------ CONFIG ------------------------------
ONNX_FILE = "best_banknote_model.onnx"
PTH_FILE = "best_banknote_model.pth"
META_FILE = "best_banknote_model.meta.json"
DEFAULT_META = {
    "img_size": 224,
    "class_names": ["2", "5", "10", "20", "50", "100", "200", "500", "1000"],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
BN = {"2": "দুই", "5": "পাঁচ", "10": "দশ", "20": "বিশ", "50": "পঞ্চাশ",
      "100": "একশো", "200": "দুইশো", "500": "পাঁচশো", "1000": "এক হাজার"}
# --------------------------------------------------------------------

HERE = Path(__file__).resolve().parent


# ------------------------------ model helpers ------------------------------
def locate(name):
    for sub in ("", "models", "model", "public"):
        p = HERE / sub / name
        if p.is_file():
            return p
    for p in HERE.rglob(name):
        if p.is_file():
            return p
    return None


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class OrtEngine:
    def __init__(self, path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def predict(self, x):
        return softmax(self.sess.run(None, {self.inp: x})[0][0])


class TorchEngine:
    def __init__(self, path, meta):
        import timm
        import torch

        self.torch = torch
        try:
            ck = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception:
            ck = torch.load(str(path), map_location="cpu", weights_only=False)
        if "class_names" in ck:
            meta["class_names"] = [str(c) for c in ck["class_names"]]
        self.model = timm.create_model(ck.get("model_name", "mobilenetv3_large_100"),
                                       pretrained=False, num_classes=len(meta["class_names"]))
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()
        torch.set_num_threads(2)

    def predict(self, x):
        with self.torch.inference_mode():
            return softmax(self.model(self.torch.from_numpy(x))[0].numpy())


@st.cache_resource(show_spinner="Loading model...")
def get_engine():
    meta = json.loads(json.dumps(DEFAULT_META))
    mp = locate(META_FILE)
    if mp:
        meta.update(json.loads(mp.read_text(encoding="utf-8")))
    onnx = locate(ONNX_FILE)
    if onnx:
        return OrtEngine(onnx), meta, "ONNX Runtime (CPU)"
    pth = locate(PTH_FILE)
    if pth:
        return TorchEngine(pth, meta), meta, "PyTorch (CPU)"
    raise FileNotFoundError(
        f"Model not found. Put {ONNX_FILE} (recommended) or {PTH_FILE} in the repo.")


def preprocess(rgb, meta):
    s = meta["img_size"]
    x = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = (x - np.array(meta["mean"], np.float32)) / np.array(meta["std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]


def roi_box(w, h, full, rw, rh):
    if full:
        return 0, 0, w, h
    bw, bh = int(w * rw), int(h * rh)
    x1, y1 = (w - bw) // 2, (h - bh) // 2
    return x1, y1, x1 + bw, y1 + bh


def put_text(img, text, org, scale=0.9, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ------------------------------ live video ------------------------------
class TakaProcessor(VideoProcessorBase):
    """Runs on a background thread for every webcam frame."""

    def __init__(self, engine, meta):
        self.engine, self.meta = engine, meta
        self.names = meta["class_names"]
        self.conf, self.ema, self.full, self.rw, self.rh, self.skip = 0.6, 0.5, False, 0.8, 0.6, 2
        self.smooth = np.zeros(len(self.names), np.float32)
        self.n, self.ms = 0, 0.0
        self.lock = threading.Lock()
        self.state = (None, 0.0, [], 0.0)      # (label or None, confidence, top3, ms)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        x1, y1, x2, y2 = roi_box(w, h, self.full, self.rw, self.rh)
        self.n += 1
        if self.n % max(int(self.skip), 1) == 0:
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            p = self.engine.predict(preprocess(rgb, self.meta))
            self.smooth = self.ema * self.smooth + (1.0 - self.ema) * p
            self.ms = 0.8 * self.ms + 0.2 * (time.perf_counter() - t0) * 1000

        k = int(self.smooth.argmax())
        c = float(self.smooth[k])
        sure = c >= self.conf
        top = [(self.names[i], float(self.smooth[i])) for i in np.argsort(self.smooth)[::-1][:3]]
        with self.lock:
            self.state = (self.names[k] if sure else None, c, top, self.ms)

        color = (0, 220, 0) if sure else (0, 200, 255)
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), color, 3)
        label = f"{self.names[k]} Taka  {c * 100:.0f}%" if sure else "Show a note..."
        put_text(img, label, (x1 + 8, max(y1 - 12, 32)), 1.0, color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def rtc_config():
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:  # optional TURN server (add [turn] url/username/credential in Streamlit secrets)
        t = st.secrets["turn"]
        servers.append({"urls": [t["url"]], "username": t["username"], "credential": t["credential"]})
    except Exception:
        pass
    return RTCConfiguration({"iceServers": servers})


# ------------------------------ photo mode ------------------------------
def photo_tab(engine, meta, conf):
    import pandas as pd

    st.write("Take a photo or upload one. Fill the frame with **one** note.")
    shot = st.camera_input("Camera")
    up = st.file_uploader("...or upload an image", type=["jpg", "jpeg", "png", "webp"])
    src = shot or up
    if src is None:
        return
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    probs = engine.predict(preprocess(np.array(img), meta))
    names = meta["class_names"]
    k = int(probs.argmax())
    if probs[k] >= conf:
        st.success(f"### ৳ {names[k]}  ·  {BN.get(names[k], '')} টাকা\nConfidence: {probs[k]:.0%}")
    else:
        st.warning(f"Not sure (best guess ৳ {names[k]}, {probs[k]:.0%}). Try a closer, brighter photo.")
    order = np.argsort(probs)[::-1][:5]
    st.dataframe(pd.DataFrame({"Taka": [names[i] for i in order],
                               "Probability": [f"{probs[i]:.1%}" for i in order]}),
                 hide_index=True, use_container_width=True)


# ------------------------------ page ------------------------------
def main():
    st.set_page_config(page_title="Taka Scanner", page_icon="৳", layout="centered")
    st.title("৳ Taka Scanner")
    st.caption("Bangladeshi banknote recognition. Hold one note inside the box.")

    try:
        engine, meta, backend = get_engine()
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.sidebar:
        st.header("Settings")
        conf = st.slider("Minimum confidence", 0.30, 0.95, 0.60, 0.05)
        ema = st.slider("Smoothing", 0.0, 0.9, 0.5, 0.1)
        skip = st.slider("Run model every N frames", 1, 6, 2)
        full = st.checkbox("Use the whole frame (no box)", value=False)
        rw = st.slider("Box width", 0.4, 1.0, 0.8, 0.05, disabled=full)
        rh = st.slider("Box height", 0.3, 1.0, 0.6, 0.05, disabled=full)
        st.caption(f"Backend: {backend}")

    tab_live, tab_photo = st.tabs(["Live camera", "Photo / upload"])

    with tab_photo:                      # build this first: the live tab below runs a loop
        photo_tab(engine, meta, conf)

    with tab_live:
        ctx = webrtc_streamer(
            key="taka",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config(),
            video_processor_factory=lambda: TakaProcessor(engine, meta),
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                          "frameRate": {"ideal": 24}, "facingMode": {"ideal": "environment"}},
                "audio": False,
            },
            async_processing=True,
        )
        out = st.empty()
        if not ctx.state.playing:
            out.info("Press START, allow the camera, then show a note.")
        while ctx.state.playing:
            vp = ctx.video_processor
            if vp is not None:
                vp.conf, vp.ema, vp.skip, vp.full, vp.rw, vp.rh = conf, ema, skip, full, rw, rh
                with vp.lock:
                    label, c, top, ms = vp.state
                if label is None:
                    out.info("Show a note inside the box...")
                else:
                    out.success(f"### ৳ {label}  ·  {BN.get(label, '')} টাকা\n"
                                f"Confidence {c:.0%}  ·  {ms:.0f} ms per frame")
            time.sleep(0.15)


if __name__ == "__main__":
    main()