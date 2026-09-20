"""Spike: quick functional checks on the synthetic WAV (run chord_test.py first)."""
import os
import sys
import time
import traceback

WAV = os.path.abspath("test_C_Am_F_G.wav")


def step(name, fn):
    t0 = time.time()
    try:
        print(f"[{name}] OK ->", fn(), "(%.1fs)" % (time.time() - t0))
    except Exception as e:
        print(f"[{name}] FAIL -> {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


def bp():
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    onnx = str(ICASSP_2022_MODEL_PATH) + ".onnx"
    model = Model(onnx if os.path.exists(onnx) else ICASSP_2022_MODEL_PATH)
    _, midi, notes = predict(WAV, model)
    pitches = sorted({n[2] for n in notes})
    return f"model_type={model.model_type} notes={len(notes)} pitches={pitches}"


def beats():
    from beat_this.inference import File2Beats

    b, d = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)(WAV)
    return f"beats={len(b)} downbeats={len(d)}"


def whisper_cuda():
    which = sys.argv[1] if len(sys.argv) > 1 else "plain"
    if which == "torchdll":
        import torch  # noqa: F401  (puts torch/lib cuDNN+cuBLAS DLLs on the search path)

        os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
    from faster_whisper import WhisperModel

    m = WhisperModel("tiny", device="cuda", compute_type="float16")
    segs, info = m.transcribe(WAV)
    return f"mode={which} lang={info.language} segs={len(list(segs))}"


step("basic-pitch", bp)
step("beat_this", beats)
step("faster-whisper cuda", whisper_cuda)
