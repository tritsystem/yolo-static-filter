"""CLIP appearance verifier -- the pattern-based (motion-independent) stage.

Given a cropped detection, CLIP scores whether it looks like a PERSON/worker or a
STRUCTURE/background. This is how a worker who has been stationary since the camera turned on
is kept (identity by appearance, not by motion history), and how a structure that YOLO
*confidently* misreads as a worker is still caught.

Plug it into StaticDetectionFilter(verifier=ClipAppearanceVerifier()). Any object with a
`.verify(crop_bgr) -> ("person"|"structure"|"unsure", margin)` method works, so the core
filter never imports torch/clip.

Run `python clip_verifier.py` to smoke-test it (downloads ViT-B-32 weights once, a few hundred MB).
"""
import numpy as np
import cv2


class ClipAppearanceVerifier:
    PERSON = ["a photo of a construction worker", "a person", "a worker in a hi-vis vest and hard hat",
              "a human standing", "a construction worker on a building site"]
    STRUCTURE = ["metal scaffolding", "a steel beam", "construction equipment", "a concrete wall",
                 "an empty background", "a pile of building materials", "a crane or girder"]

    def __init__(self, model_name="ViT-B-32-quickgelu", pretrained="openai", device=None, margin=0.15):
        import torch
        import open_clip
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device)
        self.model.eval()
        tok = open_clip.get_tokenizer(model_name)
        self.n_person = len(self.PERSON)
        prompts = self.PERSON + self.STRUCTURE
        with torch.no_grad():
            t = self.model.encode_text(tok(prompts).to(self.device))
            self.text = (t / t.norm(dim=-1, keepdim=True))
        self.margin = float(margin)

    def verify(self, crop_bgr):
        """-> ('person'|'structure'|'unsure', margin). margin = p_person - p_structure."""
        if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 4:
            return "unsure", 0.0
        from PIL import Image
        torch = self.torch
        img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            e = self.model.encode_image(x)
            e = e / e.norm(dim=-1, keepdim=True)
            logits = self.model.logit_scale.exp() * (e @ self.text.T)
            probs = logits.softmax(dim=-1)[0]
        p_person = float(probs[:self.n_person].sum())
        p_struct = float(probs[self.n_person:].sum())
        m = p_person - p_struct
        if m > self.margin:
            return "person", m
        if -m > self.margin:
            return "structure", -m
        return "unsure", m


def _smoketest():
    """Run real CLIP on two synthetic crops: a greyscale rigid-looking block vs. a colourful
    blob. Not a correctness proof (synthetic), just confirms the pipeline loads and scores."""
    v = ClipAppearanceVerifier()
    struct = np.full((80, 60, 3), 120, np.uint8)
    cv2.line(struct, (0, 0), (60, 80), (90, 90, 90), 4); cv2.line(struct, (60, 0), (0, 80), (90, 90, 90), 4)
    person = np.random.default_rng(0).integers(60, 200, (80, 40, 3), dtype=np.uint8)
    print("[clip] grey girder-like crop ->", v.verify(struct))
    print("[clip] colourful crop        ->", v.verify(person))
    print("[clip] pipeline loaded OK on", v.device)


if __name__ == "__main__":
    _smoketest()
