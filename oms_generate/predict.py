from cog import BasePredictor, Input, Path

from oms_diffusion.inference_generate import GenerateModel


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = GenerateModel()
        self.model.load_pipe()

        return self.model

    # The arguments and types the model takes as input
    def predict(
        self,
        garment_image: Path = Input(
            description="Clear picture of upper body garment",
            default="https://raw.githubusercontent.com/viktorfa/oot_diffusion/main/oot_diffusion/assets/cloth_1.jpg",
        ),
        prompt: str = Input(
            default="Realistic photo of a beautiful model", description="Prompt"
        ),
        steps: int = Input(default=20, description="Inference steps", ge=1, le=40),
        guidance_scale: float = Input(
            default=2.0, description="Guidance scale", ge=1.0, le=5.0
        ),
        seed: int = Input(default=0, description="Seed", ge=0, le=0xFFFFFFFFFFFFFFFF),
    ) -> list[Path]:
        """Run a single prediction on the model"""

        generated_images, garment_mask_image = self.model.generate(
            cloth_image=garment_image,
            num_images_per_prompt=4,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            prompt=prompt,
        )

        output_paths: list[Path] = []
        for i, img in enumerate(generated_images):
            result_path = Path(f"result_{i}.png")
            img.save(result_path, "PNG")
            output_paths.append(result_path)

        return output_paths
