# ollama_vlm_report_generator.py
"""
Ollama-based VLM medical report generation using llava-llama3:8b.

Default server: http://86.50.170.53:11434

Requirements on the remote machine:
  1. Install Ollama:   curl -fsSL https://ollama.com/install.sh | sh
  2. Pull the model:   ollama pull llava-llama3:8b
  3. Start with host:  OLLAMA_HOST=0.0.0.0:11434 ollama serve

Requirements on this machine:
  pip install requests Pillow
"""
import io
import base64
import time
import json
import requests
import logging
from PIL import Image
from typing import Dict, List, Optional

from config import PipelineConfig, OllamaVLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================== #
#  Ollama Client
# ================================================================== #

class OllamaClient:
    """HTTP client for communicating with an Ollama server."""

    def __init__(self, config: OllamaVLMConfig):
        self.config = config
        self.base_url = config.host.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })

    def health_check(self) -> bool:
        try:
            resp = self.session.get(
                f"{self.base_url}/api/tags", timeout=10,
            )
            return resp.status_code == 200
        except requests.ConnectionError:
            return False
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def list_models(self) -> List[str]:
        try:
            resp = self.session.get(
                f"{self.base_url}/api/tags", timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not list Ollama models: {e}")
            return []

    def model_available(self, model_name: str) -> bool:
        models = self.list_models()
        base_name = model_name.split(":")[0]
        for m in models:
            if m == model_name:
                return True
            if m.split(":")[0] == base_name:
                return True
        return False

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        try:
            resp = self.session.post(
                f"{self.base_url}/api/show",
                json={"name": model_name}, timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Could not get model info for {model_name}: {e}")
            return None

    def pull_model(self, model_name: str) -> bool:
        try:
            logger.info(f"Pulling model {model_name} on Ollama server...")
            resp = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=1800, stream=True,
            )
            resp.raise_for_status()
            last_status = ""
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if status != last_status:
                        logger.info(f"  Pull: {status}")
                        last_status = status
            logger.info(f"Model {model_name} pulled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    def chat(
        self,
        messages: List[Dict],
        images: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str:
        if images:
            for msg in reversed(messages):
                if msg["role"] == "user":
                    msg["images"] = images
                    break

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repeat_penalty": self.config.repeat_penalty,
                "num_predict": self.config.max_new_tokens,
            },
            "keep_alive": self.config.keep_alive,
        }
        if self.config.seed is not None:
            payload["options"]["seed"] = self.config.seed

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                if stream:
                    return self._chat_stream(payload)
                else:
                    return self._chat_sync(payload)
            except requests.Timeout:
                last_error = f"Request timed out after {self.config.timeout}s."
                logger.warning(
                    f"Ollama timeout (attempt {attempt+1}/{self.config.max_retries})"
                )
            except requests.ConnectionError as e:
                last_error = f"Connection error: {e}"
                logger.warning(
                    f"Ollama connection error (attempt {attempt+1}/{self.config.max_retries})"
                )
            except requests.HTTPError as e:
                last_error = f"HTTP error: {e}"
                if hasattr(e, "response") and e.response is not None:
                    if e.response.status_code == 404:
                        raise RuntimeError(
                            f"Model '{self.config.model_name}' not found. "
                            f"Run: ollama pull {self.config.model_name}"
                        ) from e
                logger.warning(
                    f"Ollama HTTP error (attempt {attempt+1}/{self.config.max_retries})"
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Ollama error (attempt {attempt+1}/{self.config.max_retries}): {e}"
                )

            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)

        raise RuntimeError(
            f"Ollama generation failed after {self.config.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _chat_sync(self, payload: dict) -> str:
        resp = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload, timeout=self.config.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    def _chat_stream(self, payload: dict) -> str:
        payload["stream"] = True
        resp = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload, timeout=self.config.timeout, stream=True,
        )
        resp.raise_for_status()
        full_response = []
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                token = msg.get("content", "")
                full_response.append(token)
                if chunk.get("done", False):
                    eval_count = chunk.get("eval_count", 0)
                    eval_duration = chunk.get("eval_duration", 0)
                    if eval_duration > 0:
                        tps = eval_count / (eval_duration / 1e9)
                        logger.info(f"Generation speed: {tps:.1f} tokens/s")
                    break
        return "".join(full_response)

    def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repeat_penalty": self.config.repeat_penalty,
                "num_predict": self.config.max_new_tokens,
            },
            "keep_alive": self.config.keep_alive,
        }
        if images:
            payload["images"] = images
        if system:
            payload["system"] = system
        if self.config.seed is not None:
            payload["options"]["seed"] = self.config.seed

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                resp = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload, timeout=self.config.timeout,
                )
                resp.raise_for_status()
                if stream:
                    full = []
                    for line in resp.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            full.append(chunk.get("response", ""))
                            if chunk.get("done", False):
                                break
                    return "".join(full)
                else:
                    return resp.json().get("response", "")
            except Exception as e:
                last_error = str(e)
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))

        raise RuntimeError(f"Ollama generate failed: {last_error}")


# ================================================================== #
#  Image Encoder (PIL -> base64)
# ================================================================== #

class ImageEncoder:
    @staticmethod
    def encode(
        image: Image.Image,
        max_size: int = 1024,
        quality: int = 90,
    ) -> str:
        image = image.convert("RGB")
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            logger.debug(f"Image resized from {w}x{h} to {new_w}x{new_h}")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        b64_str = base64.b64encode(buffer.read()).decode("utf-8")
        size_kb = len(b64_str) * 3 / 4 / 1024
        logger.debug(f"Encoded image: {size_kb:.1f} KB (base64)")
        return b64_str


# ================================================================== #
#  Prompt Builder for llava-llama3:8b via Ollama
# ================================================================== #

class OllamaPromptBuilder:

    @staticmethod
    def build_system_prompt(
        modality_context: str = "chest X-ray radiology",
    ) -> str:
        return (
            f"You are an expert radiologist specializing in "
            f"{modality_context}. Your task is to analyze medical "
            f"images and generate accurate, detailed, and clinically "
            f"relevant radiology reports.\n\n"
            f"Guidelines:\n"
            f"- Be precise and specific in your observations\n"
            f"- Do NOT hallucinate findings not visible in the image\n"
            f"- Use standard radiology terminology\n"
            f"- Describe findings systematically (heart, lungs, "
            f"mediastinum, pleura, bones, soft tissues)\n"
            f"- Always structure your response with FINDINGS and "
            f"IMPRESSION sections\n"
            f"- If uncertain about a finding, indicate the level "
            f"of confidence"
        )

    @staticmethod
    def build_few_shot_prompt(
        example_captions: List[str],
        example_scores: Optional[List[float]] = None,
        modality_context: str = "chest X-ray radiology",
        include_scores: bool = True,
        detected_conditions: Optional[Dict] = None,
    ) -> str:
        parts = []

        if detected_conditions:
            pathological = {
                k: v for k, v in detected_conditions.items()
                if k not in ("normal", "support_devices")
            }
            if pathological:
                cond_strs = []
                for k, v in sorted(
                    pathological.items(),
                    key=lambda x: x[1].get("avg_score", 0),
                    reverse=True,
                ):
                    freq = v.get("frequency", v.get("weighted_probability", 0))
                    cond_strs.append(
                        f"{k.replace('_', ' ')} ({freq*100:.0f}% of similar cases)"
                    )
                parts.append(
                    "**Database Analysis Context:**\n"
                    "The following conditions were frequently found in "
                    "visually similar cases: " + ", ".join(cond_strs) + ".\n"
                    "Use this as supplementary context but rely primarily "
                    "on what you directly observe in the provided image."
                )
                parts.append("")

        if example_captions:
            parts.append(
                "**Reference Reports from Similar Cases:**\n"
                "Below are radiology reports from the most visually "
                "similar cases found in the medical database. Use "
                "these as reference for style, terminology, and the "
                "types of findings to look for."
            )
            parts.append("")

            for i, caption in enumerate(example_captions, 1):
                score_str = ""
                if include_scores and example_scores and i - 1 < len(example_scores):
                    score_str = f" [similarity: {example_scores[i-1]:.3f}]"
                parts.append(f"--- Similar Case {i}{score_str} ---")
                parts.append(caption.strip())
                parts.append("")

            parts.append("--- End of Reference Cases ---")
            parts.append("")

        parts.append(
            f"**Your Task:**\n"
            f"Analyze the provided {modality_context} image carefully. "
            f"Generate a comprehensive radiology report structured as "
            f"follows:\n\n"
            f"FINDINGS:\n"
            f"(Describe all observations in detail. Include: heart size "
            f"and contour, mediastinal width, lung fields, pleural "
            f"spaces, osseous structures, any lines/tubes, and any "
            f"abnormalities.)\n\n"
            f"IMPRESSION:\n"
            f"(Summarize the key diagnoses and provide clinical "
            f"recommendations if appropriate.)"
        )

        return "\n".join(parts)

    @staticmethod
    def build_zero_shot_prompt(
        modality_context: str = "chest X-ray radiology",
        detected_conditions: Optional[Dict] = None,
    ) -> str:
        parts = []

        if detected_conditions:
            pathological = {
                k: v for k, v in detected_conditions.items()
                if k not in ("normal", "support_devices")
            }
            if pathological:
                cond_strs = [
                    k.replace('_', ' ')
                    for k in sorted(
                        pathological.keys(),
                        key=lambda x: pathological[x].get(
                            "avg_score",
                            pathological[x].get("weighted_probability", 0),
                        ),
                        reverse=True,
                    )
                ]
                parts.append(
                    "Automated pre-screening suggests possible: "
                    + ", ".join(cond_strs) + ". "
                    "Verify these findings based on your own analysis."
                )
                parts.append("")

        parts.append(
            f"Analyze the provided {modality_context} image carefully "
            f"and generate a detailed radiology report."
        )
        parts.append("")
        parts.append(
            "Structure your response as follows:\n\n"
            "FINDINGS:\n"
            "(Provide detailed observations of all anatomical "
            "structures visible in the image.)\n\n"
            "IMPRESSION:\n"
            "(Summarize key diagnoses and recommendations.)"
        )

        return "\n".join(parts)

    @staticmethod
    def build_chat_messages(
        user_prompt: str,
        system_prompt: str,
    ) -> List[Dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


# ================================================================== #
#  Ollama VLM Report Generator
# ================================================================== #

class OllamaVLMReportGenerator:
    """
    Generates medical reports using llava-llama3:8b
    hosted on an Ollama server at http://86.50.170.53:11434.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.ollama_config = config.ollama_vlm
        self.client = OllamaClient(self.ollama_config)
        self.prompt_builder = OllamaPromptBuilder()
        self.image_encoder = ImageEncoder()
        self._connection_verified = False

    # ---------------------------------------------------------- #
    #  Connection management
    # ---------------------------------------------------------- #

    def verify_connection(self) -> bool:
        if self._connection_verified:
            return True

        logger.info(
            f"Verifying Ollama connection at "
            f"{self.ollama_config.host} for "
            f"{self.ollama_config.model_name}..."
        )

        if not self.client.health_check():
            logger.error(
                f"Cannot reach Ollama server at {self.ollama_config.host}.\n"
                f"Start Ollama: OLLAMA_HOST=0.0.0.0:11434 ollama serve"
            )
            return False

        logger.info("Ollama server is reachable")

        if not self.client.model_available(self.ollama_config.model_name):
            available = self.client.list_models()
            logger.warning(
                f"Model '{self.ollama_config.model_name}' not found. "
                f"Available: {available}"
            )
            logger.info(f"Attempting to pull {self.ollama_config.model_name}...")
            if not self.client.pull_model(self.ollama_config.model_name):
                logger.error(
                    f"Failed to pull. Run on server: "
                    f"ollama pull {self.ollama_config.model_name}"
                )
                return False

        info = self.client.get_model_info(self.ollama_config.model_name)
        if info:
            params = info.get("details", {}).get("parameter_size", "unknown")
            family = info.get("details", {}).get("family", "unknown")
            logger.info(
                f"Model ready: {self.ollama_config.model_name} "
                f"(family={family}, params={params})"
            )

        self._connection_verified = True
        return True

    def reset_connection(self):
        self._connection_verified = False

    @property
    def is_available(self) -> bool:
        return self.ollama_config.enabled and self.verify_connection()

    # ---------------------------------------------------------- #
    #  Core generation
    # ---------------------------------------------------------- #

    def _generate(
        self,
        prompt: str,
        image: Image.Image,
        system_prompt: Optional[str] = None,
    ) -> str:
        if not self.verify_connection():
            raise RuntimeError(
                f"Ollama server not available at {self.ollama_config.host}"
            )

        b64_image = self.image_encoder.encode(
            image,
            max_size=self.ollama_config.image_max_size,
            quality=self.ollama_config.image_quality,
        )

        if system_prompt is None:
            system_prompt = (
                self.ollama_config.system_prompt
                or self.prompt_builder.build_system_prompt(
                    self.ollama_config.modality_context
                )
            )

        messages = self.prompt_builder.build_chat_messages(
            user_prompt=prompt,
            system_prompt=system_prompt,
        )

        logger.info(
            f"Sending request to Ollama "
            f"({self.ollama_config.model_name} @ {self.ollama_config.host})..."
        )
        start_time = time.time()

        result = self.client.chat(
            messages=messages,
            images=[b64_image],
            stream=self.ollama_config.stream,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"llava-llama3:8b generation completed in "
            f"{elapsed:.1f}s ({len(result)} chars)"
        )

        return result.strip()

    # ---------------------------------------------------------- #
    #  Public API — few-shot
    # ---------------------------------------------------------- #

    def generate_report_few_shot(
        self,
        query_image: Image.Image,
        retrieval_results: List[Dict],
        detected_conditions: Optional[Dict] = None,
        num_examples: Optional[int] = None,
    ) -> Dict:
        if not self.is_available:
            return {
                "report": (
                    f"Ollama VLM not available. Check connection to "
                    f"{self.ollama_config.host}.\n"
                    f"Ensure model is pulled: ollama pull "
                    f"{self.ollama_config.model_name}"
                ),
                "method": "ollama_few_shot",
                "success": False,
                "error": "Ollama not reachable",
            }

        if num_examples is None:
            num_examples = self.ollama_config.num_examples

        examples = retrieval_results[:num_examples]

        try:
            example_captions = [r["caption"] for r in examples]
            example_scores = [r.get("score", 0.0) for r in examples]

            logger.info(
                f"llava-llama3:8b few-shot generation with "
                f"{len(example_captions)} examples..."
            )

            prompt = self.prompt_builder.build_few_shot_prompt(
                example_captions=example_captions,
                example_scores=example_scores,
                modality_context=self.ollama_config.modality_context,
                include_scores=self.ollama_config.include_scores_in_prompt,
                detected_conditions=(
                    detected_conditions
                    if self.ollama_config.include_conditions_context
                    else None
                ),
            )

            report_text = self._generate(prompt, query_image)

            formatted = self._format_report(
                report_text,
                retrieval_results=examples,
                mode="few_shot",
            )

            return {
                "report": formatted,
                "raw_vlm_output": report_text,
                "method": "ollama_few_shot",
                "num_examples": len(example_captions),
                "success": True,
                "model": self.ollama_config.model_name,
                "backend": "ollama",
                "host": self.ollama_config.host,
            }

        except Exception as e:
            logger.error(
                f"llava-llama3:8b generation failed: {e}", exc_info=True,
            )
            return {
                "report": f"Ollama VLM generation failed: {e}",
                "method": "ollama_few_shot",
                "success": False,
                "error": str(e),
            }

    # ---------------------------------------------------------- #
    #  Public API — zero-shot
    # ---------------------------------------------------------- #

    def generate_report_zero_shot(
        self,
        query_image: Image.Image,
        detected_conditions: Optional[Dict] = None,
    ) -> Dict:
        if not self.is_available:
            return {
                "report": (
                    f"Ollama VLM not available. Check connection to "
                    f"{self.ollama_config.host}."
                ),
                "method": "ollama_zero_shot",
                "success": False,
                "error": "Ollama not reachable",
            }

        try:
            logger.info("llava-llama3:8b zero-shot generation...")

            prompt = self.prompt_builder.build_zero_shot_prompt(
                modality_context=self.ollama_config.modality_context,
                detected_conditions=(
                    detected_conditions
                    if self.ollama_config.include_conditions_context
                    else None
                ),
            )

            report_text = self._generate(prompt, query_image)

            formatted = self._format_report(
                report_text, mode="zero_shot",
            )

            return {
                "report": formatted,
                "raw_vlm_output": report_text,
                "method": "ollama_zero_shot",
                "num_examples": 0,
                "success": True,
                "model": self.ollama_config.model_name,
                "backend": "ollama",
                "host": self.ollama_config.host,
            }

        except Exception as e:
            logger.error(
                f"llava-llama3:8b zero-shot failed: {e}", exc_info=True,
            )
            return {
                "report": f"Ollama VLM generation failed: {e}",
                "method": "ollama_zero_shot",
                "success": False,
                "error": str(e),
            }

    # ---------------------------------------------------------- #
    #  Report formatting
    # ---------------------------------------------------------- #

    def _format_report(
        self,
        raw_text: str,
        retrieval_results: Optional[List[Dict]] = None,
        mode: str = "few_shot",
    ) -> str:
        lines = []
        lines.append("=" * 65)

        if mode == "few_shot":
            lines.append(
                "  AI-GENERATED RADIOLOGY REPORT  "
                "(llava-llama3:8b Few-Shot)"
            )
        else:
            lines.append(
                "  AI-GENERATED RADIOLOGY REPORT  "
                "(llava-llama3:8b Zero-Shot)"
            )

        lines.append(
            f"  Model: {self.ollama_config.model_name} "
            f"@ {self.ollama_config.host}"
        )
        lines.append("=" * 65)
        lines.append("")

        cleaned = self._clean_vlm_output(raw_text)
        lines.append(cleaned)

        lines.append("")
        lines.append("-" * 65)

        if mode == "few_shot" and retrieval_results:
            lines.append(
                f"  Generated using {len(retrieval_results)} "
                f"similar case captions as context"
            )
            scores = [r.get("score", 0) for r in retrieval_results]
            if scores:
                lines.append(
                    f"  Example similarity range: "
                    f"{min(scores):.3f} - {max(scores):.3f}"
                )
        else:
            lines.append("  Generated without example context")

        lines.append(
            f"  Temperature: {self.ollama_config.temperature} | "
            f"Max tokens: {self.ollama_config.max_new_tokens}"
        )
        lines.append("=" * 65)

        return "\n".join(lines)

    @staticmethod
    def _clean_vlm_output(text: str) -> str:
        text = text.strip()
        for artifact in [
            "<|eot_id|>", "<|begin_of_text|>",
            "<|end_header_id|>", "<|start_header_id|>",
        ]:
            text = text.replace(artifact, "")

        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("FINDINGS") or upper.startswith("**FINDINGS"):
                line = "FINDINGS:" + (line.split(":", 1)[-1] if ":" in line else "")
            elif upper.startswith("IMPRESSION") or upper.startswith("**IMPRESSION"):
                line = "IMPRESSION:" + (line.split(":", 1)[-1] if ":" in line else "")
            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines).strip()

        has_findings = "FINDINGS" in text.upper()
        has_impression = "IMPRESSION" in text.upper()

        if not has_findings and not has_impression:
            text = f"FINDINGS:\n{text}\n\nIMPRESSION:\nSee findings above."

        return text