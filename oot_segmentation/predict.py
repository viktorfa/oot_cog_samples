from cog import BasePredictor, Input, Path, BaseModel
import tempfile

from oot_diffusion.inference_segmentation import ClothesMaskModel


class Output(BaseModel):
    model_mask: Path
    mask: Path
    original_image: Path
    model_parse: Path
    face_mask: Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.segmentation_model = ClothesMaskModel()

        return self.segmentation_model

    # The arguments and types the model takes as input
    def predict(
        self,
        model_image: Path = Input(
            description="Clear picture of the model",
            default="https://raw.githubusercontent.com/viktorfa/oot_diffusion/main/oot_diffusion/assets/model_1.png",
        ),
    ) -> Output:
        """Run a single prediction on the model"""

        (
            model_mask,
            mask,
            original_image,
            model_parse,
            face_mask,
        ) = self.segmentation_model.generate(model_image)

        model_mask_path = Path(tempfile.mktemp(suffix=".png"))
        mask_path = Path(tempfile.mktemp(suffix=".png"))
        original_image_path = Path(tempfile.mktemp(suffix=".png"))
        model_parse_path = Path(tempfile.mktemp(suffix=".png"))
        face_mask_path = Path(tempfile.mktemp(suffix=".png"))

        model_mask.save(model_mask_path, "PNG")
        mask.save(mask_path, "PNG")
        original_image.save(original_image_path, "PNG")
        model_parse.save(model_parse_path, "PNG")
        face_mask.save(face_mask_path, "PNG")

        return Output(
            model_mask=model_mask_path,
            mask=mask_path,
            original_image=original_image_path,
            model_parse=model_parse_path,
            face_mask=face_mask_path,
        )
