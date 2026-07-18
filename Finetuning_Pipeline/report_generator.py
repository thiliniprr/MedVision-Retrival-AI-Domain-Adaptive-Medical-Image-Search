# report_generator.py
"""
Report generation — dispatches to retrieval-based methods
or VLM-based methods (local TinyLLaVA or Ollama llava-llama3:8b).

Supported output types:
  1. "vlm_few_shot"  — Local TinyLLaVA few-shot with query image +
                       k-retrieved images/captions (score >= 0.3)
  2. "ollama_few_shot" — Ollama llava-llama3 few-shot with query
                         image + k-retrieved images/captions
  3. "template"      — Template report using top 1-3 images with
                       weighted probability scores (score >= 0.3)
  4. "visual"        — Side-by-side gallery of top 3-5 similar
                       images with captions, saved to output folder
"""
import os
import logging
from typing import Dict, List, Optional
from PIL import Image

from config import PipelineConfig

logger = logging.getLogger(__name__)

# Minimum score threshold for all methods
SCORE_THRESHOLD = 0.3


class ReportGenerator:
    """
    Unified report generator that dispatches to the appropriate
    backend based on the requested method.

    Supported methods:
      1. "vlm_few_shot"     — local TinyLLaVA few-shot
      2. "vlm_zero_shot"    — local TinyLLaVA zero-shot
      3. "ollama_few_shot"  — Ollama llava-llama3 few-shot
      4. "ollama_zero_shot" — Ollama llava-llama3 zero-shot
      5. "template"         — Template-based with weighted probabilities
      6. "visual"           — Side-by-side visual gallery
      7. "weighted"         — Weighted average retrieval report
      8. "majority"         — Majority vote retrieval report
      9. "concat"           — Concatenated captions
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # VLM generators (lazy loaded)
        self._vlm_generator = None
        self._vlm_dataset = None

        # Condition detection keywords
        self._condition_keywords = {
            "cardiomegaly": [
                "cardiomegaly", "enlarged heart",
                "cardiac enlargement", "heart is enlarged",
            ],
            "pleural_effusion": [
                "pleural effusion", "effusion",
                "fluid in the pleural", "blunting",
            ],
            "pneumonia": [
                "pneumonia", "consolidation", "infiltrate",
                "airspace opacity", "airspace disease",
            ],
            "pneumothorax": [
                "pneumothorax", "collapsed lung",
            ],
            "atelectasis": [
                "atelectasis", "volume loss",
                "subsegmental", "bibasilar",
            ],
            "edema": [
                "edema", "pulmonary edema", "vascular congestion",
                "cephalization", "interstitial edema",
            ],
            "fracture": [
                "fracture", "broken",
            ],
            "nodule": [
                "nodule", "mass", "lesion", "opacity",
            ],
            "normal": [
                "normal", "unremarkable", "no acute",
                "clear lungs", "no focal", "no active disease",
            ],
            "support_devices": [
                "pacemaker", "catheter", "tube", "line",
                "device", "hardware", "wire", "port",
            ],
        }

    def set_vlm_dataset(self, dataset):
        """Set dataset for VLM image loading."""
        self._vlm_dataset = dataset

    # ---------------------------------------------------------- #
    #  VLM generator (lazy init)
    # ---------------------------------------------------------- #

    def _get_vlm_generator(self):
        """Get or create the VLM report generator."""
        if self._vlm_generator is None:
            from vlm_report_generator import VLMReportGenerator
            self._vlm_generator = VLMReportGenerator(
                self.config, self._vlm_dataset
            )
        return self._vlm_generator

    # ---------------------------------------------------------- #
    #  Filter results by score threshold
    # ---------------------------------------------------------- #

    @staticmethod
    def _filter_by_score(
        results: List[Dict],
        min_score: float = SCORE_THRESHOLD,
    ) -> List[Dict]:
        """Keep only results with score >= min_score."""
        filtered = [r for r in results if r.get("score", 0) >= min_score]
        if not filtered and results:
            # Always keep at least the best result
            filtered = [results[0]]
        return filtered

    # ---------------------------------------------------------- #
    #  Condition detection from captions
    # ---------------------------------------------------------- #

    def _detect_conditions(
        self, retrieval_results: List[Dict]
    ) -> Dict:
        """
        Analyze retrieved captions to detect common conditions.
        Used as additional context for VLM prompts and template reports.
        """
        if not retrieval_results:
            return {}

        condition_hits = {}

        for result in retrieval_results:
            caption = result.get("caption", "").lower()
            score = result.get("score", 0.0)

            for condition, keywords in (
                self._condition_keywords.items()
            ):
                for keyword in keywords:
                    if keyword in caption:
                        if condition not in condition_hits:
                            condition_hits[condition] = {
                                "count": 0,
                                "total_score": 0.0,
                                "captions": [],
                            }
                        condition_hits[condition]["count"] += 1
                        condition_hits[condition][
                            "total_score"
                        ] += score
                        condition_hits[condition][
                            "captions"
                        ].append(caption[:100])
                        break

        n = len(retrieval_results)
        detected = {}
        for condition, data in condition_hits.items():
            detected[condition] = {
                "frequency": data["count"] / n,
                "avg_score": (
                    data["total_score"] / data["count"]
                ),
                "count": data["count"],
                "weighted_probability": min(
                    1.0,
                    (data["total_score"] / data["count"])
                    * (data["count"] / n),
                ),
            }

        return detected

    # ---------------------------------------------------------- #
    #  Main dispatch
    # ---------------------------------------------------------- #

    def generate_report(
        self,
        retrieval_results: List[Dict],
        method: Optional[str] = None,
        query_image: Optional[Image.Image] = None,
    ) -> Dict:
        """
        Generate a report using the specified method.

        Args:
            retrieval_results: List of dicts from RetrievalEngine
            method: Report generation method
            query_image: PIL Image (required for VLM and visual methods)

        Returns:
            Dict with 'report', 'method', 'success', and metadata
        """
        if method is None:
            method = self.config.retrieval.aggregation_method

        logger.info(f"Report generation method: {method}")

        # Filter results by score threshold (>= 0.3)
        filtered_results = self._filter_by_score(
            retrieval_results, SCORE_THRESHOLD
        )
        logger.info(
            f"Results after filtering (score >= {SCORE_THRESHOLD}): "
            f"{len(filtered_results)}/{len(retrieval_results)}"
        )

        # Detect conditions for context
        detected_conditions = self._detect_conditions(
            filtered_results
        )

        # ============================================================
        # Method 1: Local VLM (TinyLLaVA) — vlm_few_shot / vlm_zero_shot
        # ============================================================
        if method.startswith("vlm_"):
            mode = method.replace("vlm_", "")
            return self._dispatch_vlm(
                mode=mode,
                query_image=query_image,
                retrieval_results=filtered_results,
                detected_conditions=detected_conditions,
            )

        # ============================================================
        # Method 2: Ollama (llava-llama3) — ollama_few_shot / ollama_zero_shot
        # ============================================================
        if method.startswith("ollama_"):
            mode = method.replace("ollama_", "")
            return self._dispatch_ollama(
                mode=mode,
                query_image=query_image,
                retrieval_results=filtered_results,
                detected_conditions=detected_conditions,
            )

        # ============================================================
        # Method 3: Template report (top 1-3, weighted probabilities)
        # ============================================================
        if method == "template":
            return self._generate_template_report(
                retrieval_results=filtered_results,
                detected_conditions=detected_conditions,
            )

        # ============================================================
        # Method 4: Visual gallery (top 3-5 side-by-side)
        # ============================================================
        if method == "visual":
            return self._generate_visual_report(
                retrieval_results=filtered_results,
                query_image=query_image,
                detected_conditions=detected_conditions,
            )

        # ============================================================
        # Other retrieval-based methods
        # ============================================================
        return self._generate_retrieval_report(
            retrieval_results=filtered_results,
            method=method,
            detected_conditions=detected_conditions,
        )

    # ---------------------------------------------------------- #
    #  VLM dispatch (local TinyLLaVA)
    # ---------------------------------------------------------- #

    def _dispatch_vlm(
        self,
        mode: str,
        query_image: Optional[Image.Image],
        retrieval_results: List[Dict],
        detected_conditions: Dict,
    ) -> Dict:
        """
        Dispatch to local TinyLLaVA backend.
        Passes query image + retrieved images/captions as context.
        """
        if query_image is None:
            return {
                "report": (
                    "VLM methods require a query image. "
                    "Pass query_image parameter."
                ),
                "method": f"vlm_{mode}",
                "success": False,
                "error": "No query image provided",
            }

        gen = self._get_vlm_generator()

        if mode == "few_shot":
            return gen.generate_report_few_shot(
                query_image=query_image,
                retrieval_results=retrieval_results,
                detected_conditions=detected_conditions,
            )
        elif mode == "zero_shot":
            return gen.generate_report_zero_shot(
                query_image=query_image,
                detected_conditions=detected_conditions,
            )
        else:
            return {
                "report": f"Unknown VLM mode: {mode}",
                "method": f"vlm_{mode}",
                "success": False,
                "error": f"Invalid mode: {mode}",
            }

    # ---------------------------------------------------------- #
    #  Ollama dispatch (llava-llama3)
    # ---------------------------------------------------------- #

    def _dispatch_ollama(
        self,
        mode: str,
        query_image: Optional[Image.Image],
        retrieval_results: List[Dict],
        detected_conditions: Dict,
    ) -> Dict:
        """
        Force dispatch to Ollama backend (llava-llama3:8b).
        """
        if query_image is None:
            return {
                "report": (
                    "Ollama VLM methods require a query image."
                ),
                "method": f"ollama_{mode}",
                "success": False,
                "error": "No query image provided",
            }

        gen = self._get_vlm_generator()

        if mode == "few_shot":
            return gen.generate_report_ollama_few_shot(
                query_image=query_image,
                retrieval_results=retrieval_results,
                detected_conditions=detected_conditions,
            )
        elif mode == "zero_shot":
            return gen.generate_report_ollama_zero_shot(
                query_image=query_image,
                detected_conditions=detected_conditions,
            )
        else:
            return {
                "report": f"Unknown Ollama mode: {mode}",
                "method": f"ollama_{mode}",
                "success": False,
                "error": f"Invalid mode: {mode}",
            }

    # ---------------------------------------------------------- #
    #  Method 3: Template Report (top 1-3 with weighted probability)
    # ---------------------------------------------------------- #

    def _generate_template_report(
        self,
        retrieval_results: List[Dict],
        detected_conditions: Dict,
    ) -> Dict:
        """
        Generate a structured template report using top 1-3 retrieved
        cases with score >= 0.3. Each detected condition is assigned
        a weighted probability based on match scores.

        Weighted probability for condition C:
          P(C) = sum(score_i for image_i where C is found) / sum(all scores)
        """
        # Use top 1-3 results only
        top_results = retrieval_results[:3]

        if not top_results:
            return {
                "report": "No similar cases found above threshold.",
                "method": "template",
                "success": False,
                "error": "No results above threshold",
            }

        # Compute weighted probabilities per condition
        total_score = sum(r.get("score", 0) for r in top_results)
        condition_weighted = {}

        for result in top_results:
            caption = result.get("caption", "").lower()
            score = result.get("score", 0.0)

            for condition, keywords in self._condition_keywords.items():
                for keyword in keywords:
                    if keyword in caption:
                        if condition not in condition_weighted:
                            condition_weighted[condition] = {
                                "score_sum": 0.0,
                                "count": 0,
                                "sources": [],
                            }
                        condition_weighted[condition]["score_sum"] += score
                        condition_weighted[condition]["count"] += 1
                        condition_weighted[condition]["sources"].append({
                            "rank": result.get("rank", 0),
                            "score": score,
                            "excerpt": caption[:80],
                        })
                        break  # count once per caption per condition

        # Compute weighted probability
        for cond, data in condition_weighted.items():
            if total_score > 0:
                data["weighted_probability"] = round(
                    data["score_sum"] / total_score, 4
                )
            else:
                data["weighted_probability"] = 0.0

        # Build the report
        lines = []
        lines.append("=" * 65)
        lines.append("  RADIOLOGY REPORT (Template-Based)")
        lines.append(
            f"  Based on {len(top_results)} similar cases "
            f"(score >= {SCORE_THRESHOLD})"
        )
        lines.append("=" * 65)
        lines.append("")

        # ── REFERENCE CASES ──
        lines.append("REFERENCE CASES:")
        for r in top_results:
            rank = r.get("rank", "?")
            score = r.get("score", 0)
            caption = r.get("caption", "N/A").strip()
            lines.append(
                f"  [{rank}] Score: {score:.4f}"
            )
            lines.append(f"      {caption[:200]}")
            lines.append("")

        # ── FINDINGS with weighted probabilities ──
        lines.append("FINDINGS (with weighted probability):")

        # Heart
        cardio = condition_weighted.get("cardiomegaly", {})
        if cardio.get("weighted_probability", 0) > 0:
            wp = cardio["weighted_probability"]
            lines.append(
                f"  Heart: Enlarged cardiac silhouette, "
                f"suggesting cardiomegaly. "
                f"[P = {wp:.1%}]"
            )
        else:
            lines.append(
                "  Heart: Normal cardiac silhouette. [P(cardiomegaly) ≈ 0%]"
            )

        # Lungs
        lung_conditions = [
            ("pneumonia", "Airspace opacity suggesting pneumonia"),
            ("atelectasis", "Atelectatic changes"),
            ("edema", "Pulmonary edema"),
            ("nodule", "Pulmonary nodule/opacity"),
        ]
        lung_found = False
        for cond_key, cond_desc in lung_conditions:
            data = condition_weighted.get(cond_key, {})
            wp = data.get("weighted_probability", 0)
            if wp > 0:
                lines.append(
                    f"  Lungs: {cond_desc}. [P = {wp:.1%}]"
                )
                lung_found = True
        if not lung_found:
            lines.append(
                "  Lungs: Clear. No focal consolidation or mass."
            )

        # Pleura
        effusion = condition_weighted.get("pleural_effusion", {})
        pneumo = condition_weighted.get("pneumothorax", {})
        if effusion.get("weighted_probability", 0) > 0:
            wp = effusion["weighted_probability"]
            lines.append(
                f"  Pleura: Pleural effusion noted. [P = {wp:.1%}]"
            )
        elif pneumo.get("weighted_probability", 0) > 0:
            wp = pneumo["weighted_probability"]
            lines.append(
                f"  Pleura: Pneumothorax identified. [P = {wp:.1%}]"
            )
        else:
            lines.append(
                "  Pleura: No pleural effusion or pneumothorax."
            )

        # Bones
        fracture = condition_weighted.get("fracture", {})
        if fracture.get("weighted_probability", 0) > 0:
            wp = fracture["weighted_probability"]
            lines.append(
                f"  Bones: Fracture suspected. [P = {wp:.1%}]"
            )

        # Support devices
        devices = condition_weighted.get("support_devices", {})
        if devices.get("weighted_probability", 0) > 0:
            wp = devices["weighted_probability"]
            lines.append(
                f"  Devices: Support devices/lines present. "
                f"[P = {wp:.1%}]"
            )

        lines.append("")

        # ── IMPRESSION ──
        lines.append("IMPRESSION:")
        pathological = {
            k: v for k, v in condition_weighted.items()
            if k not in ("normal", "support_devices")
            and v.get("weighted_probability", 0) > 0
        }

        if pathological:
            ranked_conditions = sorted(
                pathological.items(),
                key=lambda x: x[1]["weighted_probability"],
                reverse=True,
            )
            for i, (cond, data) in enumerate(ranked_conditions, 1):
                wp = data["weighted_probability"]
                label = cond.replace("_", " ").title()
                lines.append(
                    f"  {i}. {label} — weighted probability: "
                    f"{wp:.1%}"
                )
        else:
            normal_data = condition_weighted.get("normal", {})
            if normal_data.get("weighted_probability", 0) > 0:
                wp = normal_data["weighted_probability"]
                lines.append(
                    f"  No acute cardiopulmonary abnormality. "
                    f"[P(normal) = {wp:.1%}]"
                )
            else:
                lines.append(
                    "  No definitive findings based on similar cases."
                )

        lines.append("")
        lines.append("-" * 65)

        # Score summary
        scores = [r.get("score", 0) for r in top_results]
        lines.append(
            f"  Based on {len(top_results)} similar cases "
            f"(scores: {', '.join(f'{s:.4f}' for s in scores)})"
        )
        lines.append(
            f"  Score threshold: >= {SCORE_THRESHOLD}"
        )
        lines.append("=" * 65)

        return {
            "report": "\n".join(lines),
            "method": "template",
            "success": True,
            "num_results": len(top_results),
            "detected_conditions": condition_weighted,
            "similar_cases": top_results,
        }

    # ---------------------------------------------------------- #
    #  Method 4: Visual Gallery Report (top 3-5 side-by-side)
    # ---------------------------------------------------------- #

    def _generate_visual_report(
        self,
        retrieval_results: List[Dict],
        query_image: Optional[Image.Image],
        detected_conditions: Dict,
    ) -> Dict:
        """
        Generate a visual side-by-side gallery of top 3-5 similar
        images with captions. The gallery is saved to the output folder.

        Returns the text summary AND the path to the saved gallery.
        """
        # Use top 3-5 results
        top_results = retrieval_results[:5]
        if len(top_results) < 3:
            top_results = retrieval_results[:max(len(retrieval_results), 1)]

        # Text summary
        lines = []
        lines.append("=" * 65)
        lines.append("  VISUAL RETRIEVAL REPORT")
        lines.append(
            f"  Top {len(top_results)} similar cases "
            f"(score >= {SCORE_THRESHOLD})"
        )
        lines.append("=" * 65)
        lines.append("")

        for i, r in enumerate(top_results):
            score = r.get("score", 0)
            rank = r.get("rank", i + 1)
            caption = r.get("caption", "N/A").strip()
            orig_idx = r.get("original_index", "?")

            if score >= 0.8:
                label = "🟢 Very Similar"
            elif score >= 0.6:
                label = "🟡 Similar"
            elif score >= 0.4:
                label = "🟠 Moderate"
            else:
                label = "🔴 Weak"

            lines.append(f"  Match #{rank}  |  Score: {score:.4f}  |  {label}")
            lines.append(f"  Image ID: {orig_idx}")
            lines.append(f"  Caption: {caption[:200]}")
            lines.append("")

        lines.append("=" * 65)
        lines.append(
            f"  Gallery image saved to: {self.config.output_dir}/"
        )
        lines.append("=" * 65)

        return {
            "report": "\n".join(lines),
            "method": "visual",
            "success": True,
            "num_results": len(top_results),
            "detected_conditions": detected_conditions,
            "similar_cases": top_results,
            "save_gallery": True,  # Signal to pipeline to save gallery
        }

    # ---------------------------------------------------------- #
    #  Other retrieval-based report generation
    # ---------------------------------------------------------- #

    def _generate_retrieval_report(
        self,
        retrieval_results: List[Dict],
        method: str,
        detected_conditions: Dict,
    ) -> Dict:
        """Generate report using other retrieval-based methods."""
        if not retrieval_results:
            return {
                "report": "No similar cases found.",
                "method": method,
                "success": False,
                "error": "Empty retrieval results",
            }

        if method == "weighted":
            report = self._weighted_report(
                retrieval_results, detected_conditions
            )
        elif method == "majority":
            report = self._majority_report(
                retrieval_results, detected_conditions
            )
        elif method == "concat":
            report = self._concat_report(retrieval_results)
        else:
            report = self._concat_report(retrieval_results)

        return {
            "report": report,
            "method": method,
            "success": True,
            "num_results": len(retrieval_results),
            "detected_conditions": detected_conditions,
            "similar_cases": retrieval_results,
        }

    def _weighted_report(
        self,
        results: List[Dict],
        conditions: Dict,
    ) -> str:
        """Generate weighted report from retrieved captions."""
        lines = []
        lines.append("=" * 65)
        lines.append("  RADIOLOGY REPORT (Weighted Retrieval)")
        lines.append("=" * 65)
        lines.append("")

        lines.append("FINDINGS:")
        for i, r in enumerate(results[:5], 1):
            score = r.get("score", 0.0)
            caption = r.get("caption", "N/A").strip()
            lines.append(
                f"  [{i}] (score: {score:.3f}) {caption}"
            )
        lines.append("")

        lines.append("IMPRESSION:")
        if conditions:
            pathological = {
                k: v for k, v in conditions.items()
                if k not in ("normal", "support_devices")
                and v["frequency"] >= 0.3
            }
            if pathological:
                for cond, data in sorted(
                    pathological.items(),
                    key=lambda x: x[1]["frequency"],
                    reverse=True,
                ):
                    lines.append(
                        f"  - {cond.replace('_', ' ').title()}: "
                        f"found in {data['frequency']*100:.0f}% "
                        f"of similar cases"
                    )
            else:
                lines.append(
                    "  No predominant pathology detected."
                )
        else:
            lines.append("  See findings above.")

        lines.append("")
        lines.append("=" * 65)
        return "\n".join(lines)

    def _majority_report(
        self,
        results: List[Dict],
        conditions: Dict,
    ) -> str:
        """Generate report based on majority vote conditions."""
        lines = []
        lines.append("=" * 65)
        lines.append("  RADIOLOGY REPORT (Majority Vote)")
        lines.append("=" * 65)
        lines.append("")

        lines.append("FINDINGS:")
        if conditions:
            for cond, data in sorted(
                conditions.items(),
                key=lambda x: x[1]["frequency"],
                reverse=True,
            ):
                freq_pct = data["frequency"] * 100
                if freq_pct >= 20:
                    lines.append(
                        f"  - {cond.replace('_', ' ').title()}: "
                        f"{freq_pct:.0f}% frequency "
                        f"(avg similarity: {data['avg_score']:.3f})"
                    )
        else:
            lines.append("  No conditions detected.")

        lines.append("")
        lines.append("IMPRESSION:")

        top_conditions = [
            k for k, v in conditions.items()
            if v["frequency"] >= 0.4
            and k not in ("normal", "support_devices")
        ]
        if top_conditions:
            lines.append(
                "  " + "; ".join(
                    c.replace("_", " ").title()
                    for c in top_conditions
                )
            )
        elif conditions.get("normal", {}).get(
            "frequency", 0
        ) >= 0.5:
            lines.append(
                "  No acute cardiopulmonary abnormality."
            )
        else:
            lines.append("  See findings above.")

        lines.append("")
        lines.append("=" * 65)
        return "\n".join(lines)

    def _concat_report(self, results: List[Dict]) -> str:
        """Concatenate top retrieved captions."""
        lines = []
        lines.append("=" * 65)
        lines.append("  RADIOLOGY REPORT (Retrieved Captions)")
        lines.append("=" * 65)
        lines.append("")

        for i, r in enumerate(results[:5], 1):
            score = r.get("score", 0.0)
            caption = r.get("caption", "N/A").strip()
            lines.append(f"Case {i} (similarity: {score:.3f}):")
            lines.append(f"  {caption}")
            lines.append("")

        lines.append("=" * 65)
        return "\n".join(lines)