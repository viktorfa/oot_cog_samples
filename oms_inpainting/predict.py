from cog import BasePredictor, Input, Path, BaseModel

from oot_diffusion.inference_segmentation import ClothesMaskModel
from oms_diffusion.inference_inpainting import InpaintingModel


class Output(BaseModel):
    generated_images: list[Path]
    superposed_images: list[Path]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = InpaintingModel()
        self.model.load_pipe()
        self.segmentation_model = ClothesMaskModel()

        return self.model

    # The arguments and types the model takes as input
    def predict(
        self,
        model_image: Path = Input(
            description="Clear picture of the model",
            default="https://raw.githubusercontent.com/viktorfa/oot_diffusion/main/oot_diffusion/assets/model_1.png",
        ),
        garment_image: Path = Input(
            description="Clear picture of upper body garment",
            default="https://raw.githubusercontent.com/viktorfa/oot_diffusion/main/oot_diffusion/assets/cloth_1.jpg",
        ),
        keep_original_face: bool = Input(
            default=True, description="Keep original face"
        ),
        only_inpaint: bool = Input(
            default=False, description="Generate new image based on original"
        ),
        steps: int = Input(default=20, description="Inference steps", ge=1, le=40),
        guidance_scale: float = Input(
            default=2.0, description="Guidance scale", ge=1.0, le=5.0
        ),
        seed: int = Input(default=0, description="Seed", ge=0, le=0xFFFFFFFFFFFFFFFF),
    ) -> Output:
        """Run a single prediction on the model"""

        (
            model_mask,
            mask,
            original_image,
            model_parse,
            face_mask,
        ) = self.segmentation_model.generate(model_image)

        generated_images, garment_mask_image, superposed_images = self.model.generate(
            cloth_image=garment_image,
            person_image=original_image,
            person_mask_image=mask,
            face_mask_image=face_mask,
            num_images_per_prompt=4,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            use_face_mask=keep_original_face,
            no_superpose=only_inpaint,
        )

        generated_paths: list[Path] = []
        superposed_paths: list[Path] = []
        for i, img in enumerate(generated_images):
            result_path = Path(f"result_{i}.png")
            img.save(result_path, "PNG")
            generated_paths.append(result_path)
        for i, img in enumerate(superposed_images):
            result_path = Path(f"superposed_{i}.png")
            img.save(result_path, "PNG")
            superposed_paths.append(result_path)

        return Output(
            generated_images=generated_paths,
            superposed_images=superposed_paths,
        )
