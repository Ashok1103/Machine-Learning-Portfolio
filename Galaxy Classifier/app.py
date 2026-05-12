import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import gradio as gr

# ── setup ─────────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Smooth / Elliptical", "Spiral", "Edge-on / Disturbed"]
CLASS_COLORS = ["#5DCAA5", "#AFA9EC", "#F0997B"]

DESCRIPTIONS = {
    "Smooth / Elliptical": (
        "Round, featureless galaxies with a smooth brightness gradient. "
        "Typically older stellar populations with little active star formation. "
        "Includes elliptical and lenticular (S0) galaxies."
    ),
    "Spiral": (
        "Galaxies with disk structure and spiral arms seen roughly face-on. "
        "Arms trace regions of active star formation. "
        "Includes barred and unbarred spirals."
    ),
    "Edge-on / Disturbed": (
        "Disk galaxies seen edge-on (showing a thin elongated profile), "
        "merging systems with dual nuclei, or tidally disturbed galaxies. "
        "These can be ambiguous with smooth ellipticals due to projection effects."
    ),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── load model ────────────────────────────────────────────────────────────────
def load_model():
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, 3)
    net.load_state_dict(
        torch.load("resnet18_finetuned.pth", map_location=device)
    )
    net.eval()
    return net.to(device)

model = load_model()

# ── grad-cam ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, net):
        self.net   = net
        self.feats = None
        self.grads = None
        net.layer4.register_forward_hook(
            lambda m, i, o: setattr(self, "feats", o.detach())
        )
        net.layer4.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grads", go[0].detach())
        )

    def __call__(self, tensor, cls=None):
        self.net.zero_grad()
        out  = self.net(tensor)
        cls  = cls if cls is not None else out.argmax(1).item()
        out[0, cls].backward()

        weights = self.grads[0].mean(dim=(1, 2))
        cam     = (weights[:, None, None] * self.feats[0]).sum(0)
        cam     = torch.relu(cam).cpu().numpy()
        cam    -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, cls, out.softmax(1)[0].detach().cpu().numpy()

gradcam = GradCAM(model)

# ── preprocessing ─────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── inference ─────────────────────────────────────────────────────────────────
def overlay_cam(pil_img, cam, alpha=0.45):
    img  = np.array(pil_img.resize((224, 224))) / 255.0
    heat = np.array(
        Image.fromarray(np.uint8(255 * cam)).resize((224, 224), Image.BILINEAR)
    ) / 255.0
    rgb  = colormap.jet(heat)[..., :3]
    out  = np.clip((1 - alpha) * img + alpha * rgb, 0, 1)
    return (out * 255).astype(np.uint8)


def predict(image):
    if image is None:
        return None, None, "Upload a galaxy image to classify."

    pil  = Image.fromarray(image).convert("RGB")
    t    = preprocess(pil).unsqueeze(0).to(device)

    cam, pred_cls, probs = gradcam(t)

    # ── grad-cam figure ───────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.patch.set_facecolor("#0f0f0f")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f0f0f")
        ax.axis("off")

    ax1.imshow(pil.resize((224, 224)))
    ax1.set_title("Original", color="white", fontsize=11)

    ax2.imshow(overlay_cam(pil, cam))
    ax2.set_title("Grad-CAM", color="white", fontsize=11)

    plt.tight_layout(pad=0.5)

    # ── confidence dict for gr.Label ─────────────────────────────────────────
    confidences = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}

    # ── text description ──────────────────────────────────────────────────────
    pred_name = CLASS_NAMES[pred_cls]
    info = (
        f"**{pred_name}** — {probs[pred_cls]*100:.1f}% confidence\n\n"
        f"{DESCRIPTIONS[pred_name]}"
    )

    return fig, confidences, info


# ── interface ──────────────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload a galaxy image"),
    outputs=[
        gr.Plot(label="Grad-CAM — where the model looks"),
        gr.Label(label="Confidence", num_top_classes=3),
        gr.Markdown(label="What this means"),
    ],
    title="Galaxy Morphology Classifier",
    description=(
        "Classifies galaxy morphology from DECam Legacy Survey imaging into three "
        "categories using a fine-tuned ResNet-18. "
        "Grad-CAM shows which region of the image drove the prediction.\n\n"
        "**Built by Ashok Kumar Ramu** · University of Waterloo Physics & Astronomy"
    ),
    examples=[],   # add example images here once uploaded
)

if __name__ == "__main__":
    demo.launch()
