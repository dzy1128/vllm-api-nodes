import io
import time
import random
import base64
import numpy as np
from PIL import Image
from openai import OpenAI


MODEL_LIST = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-0.8B",
]

DEFAULT_BASE_URL = "http://192.168.207.229:8000/v1"


def _build_log(model, base_url, tokens_usage, elapsed, extra=None):
    lines = [
        f"Model: {model}",
        f"Endpoint: {base_url}",
    ]
    if tokens_usage:
        lines += [
            f"Prompt tokens: {tokens_usage.prompt_tokens}",
            f"Completion tokens: {tokens_usage.completion_tokens}",
            f"Total tokens: {tokens_usage.total_tokens}",
        ]
    lines.append(f"Response time: {elapsed:.2f}s")
    if extra:
        lines.extend(extra)
    return "\n".join(lines)


def _tensor_to_base64(image_tensor):
    """Convert a ComfyUI IMAGE tensor [B,H,W,C] float32 0-1 to base64 JPEG."""
    img_np = image_tensor[0].cpu().numpy()
    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class ImageUnderstanding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                }),
                "user_prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True,
                }),
                "model": (MODEL_LIST,),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "max_tokens": ("INT", {
                    "default": 4096, "min": 1, "max": 32768, "step": 1,
                }),
                "temperature": ("FLOAT", {
                    "default": 1.3, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "top_p": ("FLOAT", {
                    "default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "top_k": ("INT", {
                    "default": 80, "min": 1, "max": 200, "step": 1,
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 1.2, "min": -2.0, "max": 2.0, "step": 0.01,
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.5, "min": -2.0, "max": 2.0, "step": 0.01,
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 2**31 - 1, "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "log")
    FUNCTION = "run"
    CATEGORY = "vLLM API"

    def run(self, image, system_prompt, user_prompt, model, base_url,
            max_tokens, temperature, top_p, top_k,
            presence_penalty, frequency_penalty, enable_thinking, seed):
        actual_seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
        image_b64 = _tensor_to_base64(image)

        client = OpenAI(base_url=base_url, api_key="EMPTY")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": user_prompt},
            ]},
        ]

        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=actual_seed,
            extra_body={"top_k": top_k,
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},},
        )
        elapsed = time.time() - t0

        content = resp.choices[0].message.content
        log = _build_log(model, base_url, resp.usage, elapsed,
                         extra=[f"Seed: {actual_seed}"])
        return (content, log)


class TextGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                }),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "model": (MODEL_LIST,),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "max_tokens": ("INT", {
                    "default": 4096, "min": 1, "max": 32768, "step": 1,
                }),
                "temperature": ("FLOAT", {
                    "default": 1.3, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "top_p": ("FLOAT", {
                    "default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "top_k": ("INT", {
                    "default": 80, "min": 1, "max": 200, "step": 1,
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 1.2, "min": -2.0, "max": 2.0, "step": 0.01,
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.5, "min": -2.0, "max": 2.0, "step": 0.01,
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 2**31 - 1, "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "log")
    FUNCTION = "run"
    CATEGORY = "vLLM API"

    def run(self, system_prompt, user_prompt, model, base_url,
            max_tokens, temperature, top_p, top_k,
            presence_penalty, frequency_penalty, enable_thinking, seed):
        actual_seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)

        client = OpenAI(base_url=base_url, api_key="EMPTY")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=actual_seed,
            extra_body={"top_k": top_k,
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},},
        )
        elapsed = time.time() - t0

        content = resp.choices[0].message.content
        log = _build_log(model, base_url, resp.usage, elapsed,
                         extra=[f"Seed: {actual_seed}"])
        return (content, log)


NODE_CLASS_MAPPINGS = {
    "vLLM_ImageUnderstanding": ImageUnderstanding,
    "vLLM_TextGeneration": TextGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "vLLM_ImageUnderstanding": "vLLM Image Understanding",
    "vLLM_TextGeneration": "vLLM Text Generation",
}
