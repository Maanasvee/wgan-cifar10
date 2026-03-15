import gradio as gr
import os
import json
from PIL import Image

samples_dir = "samples"

def get_all_images():
    if not os.path.exists(samples_dir):
        return []
    files = sorted([f for f in os.listdir(samples_dir) if f.endswith('.png')])
    images = []
    for f in files:
        path = os.path.join(samples_dir, f)
        images.append((path, f))
    return images

def get_loss_plot():
    import matplotlib.pyplot as plt
    path = "logs/history.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        history = json.load(f)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["critic_loss"], label="Critic Loss", color="#60a5fa")
    ax.plot(history["gen_loss"],    label="Generator Loss", color="#f472b6")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("WGAN Training Loss Curves")
    ax.legend()
    ax.set_facecolor("#0f0f1a")
    fig.patch.set_facecolor("#0a0a0f")
    ax.tick_params(colors="white")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2035")
    return fig

with gr.Blocks(title="WGAN CIFAR-10 Dashboard") as demo:
    gr.Markdown("# Wasserstein GAN — CIFAR-10")
    gr.Markdown("Generated images from WGAN trained on CIFAR-10 dataset for 50 epochs")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## GAN vs WGAN")
            gr.Dataframe(
                headers=["Feature", "GAN", "WGAN"],
                value=[
                    ["Loss Function", "Binary Cross-Entropy", "Wasserstein Distance"],
                    ["Output Layer", "Sigmoid → [0,1]", "Raw score (no sigmoid)"],
                    ["Constraint", "None", "Weight Clipping ±0.01"],
                    ["Optimizer", "Adam", "RMSProp"],
                    ["Mode Collapse", "Common", "Rare"],
                    ["Loss Meaning", "Uninformative", "Correlates with quality"],
                ]
            )

    with gr.Row():
        gr.Markdown("## Loss Curves")
    with gr.Row():
        loss_plot = gr.Plot()

    with gr.Row():
        gr.Markdown("## Generated Images — Epoch by Epoch")

    gallery = gr.Gallery(
        label="Generated Images",
        columns=5,
        height="auto"
    )

    def load_all():
        images = get_all_images()
        image_list = [path for path, _ in images]
        fig = get_loss_plot()
        return image_list, fig

    demo.load(load_all, outputs=[gallery, loss_plot])

demo.launch()
