"""Figure question answering with Groq Vision or a lazily loaded local ViLT model."""

from dataclasses import dataclass
from io import BytesIO
import base64
import re
import time
from typing import Optional, Union

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
from PIL import Image

from config import (
    GROQ_BASE_URL,
    GROQ_VISION_MODEL_NAME,
    OLLAMA_REQUEST_TIMEOUT,
    VQA_LOCAL_MODEL_NAME,
)


@dataclass
class VQAResult:
    answer: str
    latency_s: float
    success: bool
    error: Optional[str]
    model_name: str
    provider: str


class VisualQAModel:
    """Answer questions about figures using Groq Vision or local ViLT."""

    def __init__(
        self,
        provider: str = "ollama",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = OLLAMA_REQUEST_TIMEOUT,
    ):
        self.provider = provider.strip().lower()
        if self.provider not in {"groq", "ollama"}:
            raise ValueError("provider must be either 'groq' or 'ollama'.")

        self.api_key = api_key
        self.model_name = model_name or (
            GROQ_VISION_MODEL_NAME if self.provider == "groq" else VQA_LOCAL_MODEL_NAME
        )
        self.base_url = base_url or GROQ_BASE_URL
        self.timeout = timeout
        self.client = None
        self.processor = None
        self.model = None

        if self.provider == "groq":
            if not api_key or not api_key.strip():
                raise ValueError("A non-empty API key is required for Groq Vision.")
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=api_key,
                timeout=float(self.timeout),
            )

    @staticmethod
    def _to_rgb_image(image: Union[Image.Image, bytes]) -> Image.Image:
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL.Image or encoded image bytes.")
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Image has invalid dimensions.")
        return image.convert("RGB")

    @classmethod
    def _image_data_url(cls, image: Union[Image.Image, bytes]) -> str:
        rgb_image = cls._to_rgb_image(image)
        buffer = BytesIO()
        rgb_image.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _failure(
        self, error: str, started_at: Optional[float] = None
    ) -> VQAResult:
        latency = time.perf_counter() - started_at if started_at is not None else 0.0
        return VQAResult(
            answer="",
            latency_s=latency,
            success=False,
            error=error,
            model_name=self.model_name,
            provider=self.provider,
        )

    def answer_figure(
        self, image: Union[Image.Image, bytes], question: str
    ) -> VQAResult:
        """Answer a question grounded only in information visible in the figure."""
        if not question or not question.strip():
            return self._failure("The figure question is empty.")

        try:
            if self.provider == "groq":
                image_url = self._image_data_url(image)
                started_at = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Answer questions using only information visible in the provided figure. "
                                "Do not infer facts that the figure does not show. "
                                "If the answer cannot be determined from the figure, say so directly. "
                                "Give a concise answer and do not reveal internal reasoning."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question.strip()},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url},
                                },
                            ],
                        },
                    ],
                    temperature=0.1,
                    max_tokens=512,
                )
                latency = time.perf_counter() - started_at
                raw_answer = response.choices[0].message.content or ""
                answer = re.sub(
                    r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL
                ).strip()
                if not answer:
                    return self._failure(
                        "No response generated by the model.", started_at
                    )
                return VQAResult(
                    answer=answer,
                    latency_s=latency,
                    success=True,
                    error=None,
                    model_name=self.model_name,
                    provider=self.provider,
                )

            rgb_image = self._to_rgb_image(image)
            started_at = time.perf_counter()
            if self.processor is None or self.model is None:
                from transformers import ViltForQuestionAnswering, ViltProcessor

                self.processor = ViltProcessor.from_pretrained(self.model_name)
                self.model = ViltForQuestionAnswering.from_pretrained(self.model_name)

            inputs = self.processor(
                images=rgb_image, text=question.strip(), return_tensors="pt"
            )
            outputs = self.model(**inputs)
            answer_index = int(outputs.logits.argmax(-1).item())
            answer = self.model.config.id2label[answer_index]
            return VQAResult(
                answer=answer,
                latency_s=time.perf_counter() - started_at,
                success=True,
                error=None,
                model_name=self.model_name,
                provider=self.provider,
            )
        except APITimeoutError:
            return self._failure(
                f"Groq Vision request timed out after {self.timeout} seconds.",
                locals().get("started_at"),
            )
        except APIConnectionError:
            return self._failure(
                "Could not connect to the Groq Vision API.",
                locals().get("started_at"),
            )
        except APIError as e:
            return self._failure(
                f"Groq Vision API error: {e}", locals().get("started_at")
            )
        except Exception as e:
            return self._failure(
                f"{self.provider.capitalize()} visual QA failed: {e}",
                locals().get("started_at"),
            )
