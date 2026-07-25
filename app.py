import gc

import gradio as gr
import torch
import torchaudio

from resemble_enhance.enhancer.inference import load_enhancer
from resemble_enhance.inference import inference

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def _fn(path, solver, nfe, tau, denoising, overlap_seconds, use_fp16):
    if path is None:
        return None, None

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)

    enhancer = load_enhancer(None, device)
    enhancer.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    if device == "cuda" and use_fp16:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

    with autocast_ctx:
        # denoiser only
        wav1, new_sr = inference(
            model=enhancer.denoiser, dwav=dwav, sr=sr, device=device,
            chunk_seconds=30.0, overlap_seconds=overlap_seconds,
        )
        # denoise + enhance
        wav2, new_sr = inference(
            model=enhancer, dwav=dwav, sr=sr, device=device,
            chunk_seconds=30.0, overlap_seconds=overlap_seconds,
        )

    wav1 = wav1.float().cpu().numpy()
    wav2 = wav2.float().cpu().numpy()

    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return (new_sr, wav1), (new_sr, wav2)


def main():
    inputs: list = [
        gr.Audio(type="filepath", label="Input Audio"),
        gr.Dropdown(choices=["Midpoint", "RK4", "Euler"], value="Midpoint", label="CFM ODE Solver"),
        gr.Slider(minimum=1, maximum=128, value=64, step=1, label="CFM Number of Function Evaluations"),
        gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="CFM Prior Temperature"),
        gr.Checkbox(value=False, label="Denoise Before Enhancement"),
        gr.Slider(minimum=1, maximum=10, value=4, step=0.5, label="Chunk Overlap (seconds) - higher = smoother seams, no crackle"),
        gr.Checkbox(value=True, label="Use FP16 (lower VRAM, ~2x less memory)"),
    ]

    outputs: list = [
        gr.Audio(label="Output Denoised Audio"),
        gr.Audio(label="Output Enhanced Audio"),
    ]

    interface = gr.Interface(
        fn=_fn,
        title="Resemble Enhance",
        description="AI-driven audio enhancement for your audio files, powered by Resemble AI.",
        inputs=inputs,
        outputs=outputs,
    )

    interface.launch(share=True)


if __name__ == "__main__":
    main()
