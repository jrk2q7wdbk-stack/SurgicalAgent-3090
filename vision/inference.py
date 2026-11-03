import os
import json
import torch

from PIL import Image

from .network import Rendezvous
from .preprocessing import get_test_transform
from .postprocessing import postprocess


class SurgicalTripletRecognizer:

    def __init__(
        self,
        checkpoint_path,
        device="cuda",
        instrument_names=None,
        verb_names=None,
        target_names=None,
        triplet_classes=None,
        instrument_names_path=None,
        verb_names_path=None,
        target_names_path=None,
        triplet_classes_path=None,
        threshold=0.5,
        triplet_threshold=0.3,
        top_k_triplets=5
    ):
        # ============================
        # 1. device
        # ============================
        self.device = torch.device(
            device if torch.cuda.is_available()
            else "cpu"
        )

        print(f"[Vision] Using device: {self.device}")

        # ============================
        # 2. project path
        # ============================
        current_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        project_root = os.path.dirname(
            current_dir
        )

        knowledge_dir = os.path.join(
            project_root,
            "knowledge"
        )

        # ============================
        # 3. load label names
        # ============================
        if instrument_names is None:
            with open(
                instrument_names_path or os.path.join(
                    knowledge_dir, "instrument_names.json"
                ),
                "r",
                encoding="utf-8"
            ) as f:
                instrument_names = json.load(f)

        if verb_names is None:
            with open(
                verb_names_path or os.path.join(
                    knowledge_dir, "verb_names.json"
                ),
                "r",
                encoding="utf-8"
            ) as f:
                verb_names = json.load(f)

        if target_names is None:
            with open(
                target_names_path or os.path.join(
                    knowledge_dir, "target_names.json"
                ),
                "r",
                encoding="utf-8"
            ) as f:
                target_names = json.load(f)

        if triplet_classes is None:
            with open(
                triplet_classes_path or os.path.join(
                    knowledge_dir, "triplet_classes.json"
                ),
                "r",
                encoding="utf-8"
            ) as f:
                triplet_classes = json.load(f)

        self.instrument_names = instrument_names
        self.verb_names = verb_names
        self.target_names = target_names
        self.triplet_classes = triplet_classes

        # ============================
        # 4. thresholds
        # ============================
        self.threshold = threshold
        self.triplet_threshold = triplet_threshold
        self.top_k_triplets = top_k_triplets

        # ============================
        # 5. create model
        # ============================
        self.model = Rendezvous(
            basename="resnet18",
            num_tool=6,
            num_verb=10,
            num_target=15,
            num_triplet=100,
            layer_size=8,
            num_heads=4,
            hr_output=False,
            use_ln=False
        ).to(self.device)

        # ============================
        # 6. load checkpoint
        # ============================
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        # 兼容常见 checkpoint 格式
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 兼容 DataParallel 保存出来的 module.xxx
        clean_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]

            clean_state_dict[key] = value

        self.model.load_state_dict(
            clean_state_dict,
            strict=True
        )

        self.model.eval()

        # ============================
        # 7. preprocessing
        # ============================
        self.transform = get_test_transform()

        print("[Vision] Model loaded successfully.")

    def predict(self, image_path):

        # ============================
        # 1. check image
        # ============================
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"找不到图像：{image_path}"
            )

        # ============================
        # 2. load image
        # ============================
        image = Image.open(
            image_path
        ).convert("RGB")

        # ============================
        # 3. preprocessing
        # ============================
        tensor = (
            self.transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        # ============================
        # 4. inference
        # ============================
        with torch.inference_mode():

            (
                (cam_i, logit_i),
                (cam_v, logit_v),
                (cam_t, logit_t),
                logit_ivt
            ) = self.model(tensor)

        # ============================
        # 5. postprocessing
        # ============================
        result = postprocess(
            logit_i=logit_i,
            logit_v=logit_v,
            logit_t=logit_t,
            logit_ivt=logit_ivt,

            instrument_names=self.instrument_names,
            verb_names=self.verb_names,
            target_names=self.target_names,
            triplet_classes=self.triplet_classes,

            threshold=self.threshold,
            triplet_threshold=self.triplet_threshold,
            top_k_triplets=self.top_k_triplets
        )

        result["image"] = os.path.basename(
            image_path
        )

        return result
