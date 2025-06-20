import os
import requests
import json
from PIL import Image
import io
import base64
from utils import image_to_base64, base64_to_image


class HuggingFaceAPI:
    """Service to interact with Hugging Face Inference API"""

    def __init__(self, model_id="CompVis/stable-diffusion-v1-4", api_token=None):
        self.api_token = api_token or os.environ.get("HF_API_TOKEN")

        if not self.api_token:
            raise ValueError("Missing Hugging Face API token. Set HF_API_TOKEN as environment variable.")

        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        print("✅ Loaded Hugging Face token from environment.")
        print("Model:", self.model_id)
        print("API URL:", self.api_url)

    def generate_image(
        self,
        prompt,
        negative_prompt="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=None,
    ):
        """Generate an image from a text prompt using HF API"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }
        }

        if seed is not None:
            payload["parameters"]["seed"] = seed

        print("🔁 Sending generation request to Hugging Face...")
        response = requests.post(self.api_url, headers=self.headers, json=payload)

        if response.status_code != 200:
            print("❌ Hugging Face API Error:", response.text)
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

        try:
            image = Image.open(io.BytesIO(response.content))
        except Exception as e:
            raise Exception(f"Failed to decode image from Hugging Face response: {e}")

        base64_image = image_to_base64(image)

        result_seed = seed
        if "x-seed" in response.headers:
            result_seed = int(response.headers.get("x-seed"))

        return {
            "image": base64_image,
            "prompt": prompt,
            "seed": result_seed,
        }

    def generate_variations(
        self,
        image,
        prompt="",
        negative_prompt="",
        strength=0.75,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_variations=4,
    ):
        """Generate variations of an input image using img2img"""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        img2img_url = f"https://api-inference.huggingface.co/models/{self.model_id}/img2img"

        variations = []

        for _ in range(num_variations):
            files = {
                'image': img_byte_arr,
            }

            data = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'strength': strength,
                'guidance_scale': guidance_scale,
                'num_inference_steps': num_inference_steps,
            }

            response = requests.post(img2img_url, headers=self.headers, files=files, data=data)

            if response.status_code != 200:
                print("❌ Variation API Error:", response.text)
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

            variation_image = Image.open(io.BytesIO(response.content))
            base64_image = image_to_base64(variation_image)

            seed = None
            if "x-seed" in response.headers:
                seed = int(response.headers.get("x-seed"))
            else:
                import random
                seed = random.randint(0, 2**32 - 1)

            variations.append({
                "image": base64_image,
                "prompt": prompt,
                "seed": seed,
            })

        return variations
