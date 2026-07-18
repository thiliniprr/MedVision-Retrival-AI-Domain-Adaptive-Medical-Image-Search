# visual_report.py
"""
Visual report generator — creates side-by-side gallery of
query image + top 3-5 retrieved similar images with captions
and similarity scores. Saved to the output folder.

Output types:
  1. Matplotlib figure (publication quality, saved as PNG)
  2. PIL gallery (no matplotlib needed, Gradio-compatible)
  3. Text-only ASCII art (always works, no dependencies)
"""
import os
import io
import logging
import numpy as np
from typing import Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Score threshold
SCORE_THRESHOLD = 0.3


class VisualReportGenerator:
    """
    Creates visual comparison reports showing query image
    alongside top 3-5 retrieved similar cases with captions.

    The gallery is automatically saved to the output folder.
    """

    def __init__(self, dataset=None):
        self.dataset = dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    # ---------------------------------------------------------- #
    #  Load a retrieved image from the dataset
    # ---------------------------------------------------------- #

    def _load_image_by_index(self, original_index: int) -> Optional[Image.Image]:
        if self.dataset is None:
            return None
        try:
            if hasattr(self.dataset, "dataset"):
                item = self.dataset.dataset[original_index]
            else:
                item = self.dataset[original_index]

            raw = item.get("image", None)
            if raw is None:
                return None
            if isinstance(raw, Image.Image):
                return raw.convert("RGB")
            if isinstance(raw, str):
                return Image.open(raw).convert("RGB")
            if isinstance(raw, dict):
                if "bytes" in raw and raw["bytes"]:
                    return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
                if "path" in raw and raw["path"]:
                    return Image.open(raw["path"]).convert("RGB")
            arr = np.array(raw)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        except Exception as e:
            logger.warning(f"Could not load image at index {original_index}: {e}")
            return None

    # ---------------------------------------------------------- #
    #  Main gallery creation (top 3-5 side-by-side, saved to output)
    # ---------------------------------------------------------- #

    def create_visual_report(
        self,
        query_image: Image.Image,
        retrieval_results: List[Dict],
        save_path: Optional[str] = None,
        title: str = "Medical Image Retrieval — Visual Comparison",
        output_dir: str = "./output",
    ) -> Optional[str]:
        """
        Create a side-by-side gallery of query image + top 3-5
        retrieved similar images with captions and scores.

        Layout:
        ┌──────────────────────────────────────────────┐
        │              QUERY IMAGE (large)              │
        ├──────────┬──────────┬──────────┬──────────┬───┤
        │ Match #1 │ Match #2 │ Match #3 │ Match #4 │#5 │
        │ score    │ score    │ score    │ score    │   │
        │ caption  │ caption  │ caption  │ caption  │   │
        └──────────┴──────────┴──────────┴──────────┴───┘

        Returns:
            Path to the saved gallery PNG, or None if failed
        """
        if not HAS_MATPLOTLIB:
            logger.warning(
                "matplotlib not available — falling back to PIL gallery"
            )
            return self._create_pil_gallery_and_save(
                query_image, retrieval_results,
                save_path=save_path, output_dir=output_dir,
            )

        # Filter by score threshold and take top 3-5
        filtered = [
            r for r in retrieval_results
            if r.get("score", 0) >= SCORE_THRESHOLD
        ]
        if not filtered:
            filtered = retrieval_results[:1] if retrieval_results else []

        # Take top 3-5
        top_results = filtered[:5]
        if len(top_results) < 3 and len(filtered) >= 3:
            top_results = filtered[:3]

        n_results = len(top_results)
        if n_results == 0:
            logger.warning("No retrieval results to visualize")
            return None

        n_cols = min(n_results, 5)
        n_rows_results = (n_results + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(4 * n_cols, 4 + 5 * n_rows_results))
        gs = gridspec.GridSpec(
            1 + n_rows_results, n_cols,
            height_ratios=[1.2] + [1.0] * n_rows_results,
            hspace=0.4, wspace=0.3,
        )

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

        # ── Query image (top, centered) ──
        ax_query_start = max(0, (n_cols - 2) // 2)
        ax_query_end = min(n_cols, ax_query_start + 2)
        ax_query = fig.add_subplot(gs[0, ax_query_start:ax_query_end])
        ax_query.imshow(query_image)
        ax_query.set_title(
            "QUERY IMAGE", fontsize=14, fontweight="bold", color="darkblue"
        )
        ax_query.axis("off")
        for spine in ax_query.spines.values():
            spine.set_visible(True)
            spine.set_color("blue")
            spine.set_linewidth(3)

        # ── Retrieved images (bottom rows) ──
        for i, result in enumerate(top_results):
            row = 1 + i // n_cols
            col = i % n_cols
            ax = fig.add_subplot(gs[row, col])

            retrieved_img = self._load_image_by_index(result["original_index"])

            if retrieved_img is not None:
                ax.imshow(retrieved_img)
            else:
                ax.text(
                    0.5, 0.5, "Image\nUnavailable",
                    ha="center", va="center",
                    fontsize=12, color="gray",
                    transform=ax.transAxes,
                )
                ax.set_facecolor("#f0f0f0")

            score = result.get("score", 0)
            rank = result.get("rank", i + 1)

            if score >= 0.8:
                color, label = "darkgreen", "Very Similar"
            elif score >= 0.6:
                color, label = "orange", "Similar"
            elif score >= 0.4:
                color, label = "darkorange", "Moderate"
            else:
                color, label = "red", "Weak"

            ax.set_title(
                f"#{rank}  Score: {score:.3f} ({label})",
                fontsize=10, fontweight="bold", color=color,
            )

            caption = result.get("caption", "No caption")
            wrapped = self._wrap_text(caption, max_chars=50, max_lines=3)
            ax.set_xlabel(wrapped, fontsize=7, wrap=True)
            ax.tick_params(
                left=False, bottom=False,
                labelleft=False, labelbottom=False,
            )

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2)

        # Hide unused subplots
        total_slots = n_rows_results * n_cols
        for i in range(n_results, total_slots):
            row = 1 + i // n_cols
            col = i % n_cols
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")

        # ── Save ──
        if save_path is None:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "visual_gallery.png")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"Visual gallery saved: {save_path}")
        return save_path

    # ---------------------------------------------------------- #
    #  Detailed visual report (with conditions summary panel)
    # ---------------------------------------------------------- #

    def create_detailed_visual_report(
        self,
        query_image: Image.Image,
        retrieval_results: List[Dict],
        detected_conditions: Optional[Dict] = None,
        save_path: Optional[str] = None,
        output_dir: str = "./output",
    ) -> Optional[str]:
        """
        Extended visual report with conditions summary panel.
        Includes top 3-5 similar images with scores >= 0.3.
        """
        if not HAS_MATPLOTLIB:
            return None

        # Filter and take top 3-5
        filtered = [
            r for r in retrieval_results
            if r.get("score", 0) >= SCORE_THRESHOLD
        ]
        if not filtered:
            filtered = retrieval_results[:1] if retrieval_results else []
        top_results = filtered[:5]

        n_results = len(top_results)
        n_cols = min(n_results, 5)

        fig = plt.figure(figsize=(4 * max(n_cols, 3), 10))
        gs = gridspec.GridSpec(
            2, max(n_cols, 3),
            height_ratios=[1.5, 1.0],
            hspace=0.35, wspace=0.3,
        )

        fig.suptitle(
            "Medical Image Retrieval — Detailed Visual Report",
            fontsize=16, fontweight="bold", y=0.98,
        )

        # ── Top-left: Query image ──
        effective_cols = max(n_cols, 3)
        cols_for_query = max(2, effective_cols // 2)
        ax_query = fig.add_subplot(gs[0, :cols_for_query])
        ax_query.imshow(query_image)
        ax_query.set_title(
            "Query Image", fontsize=14, fontweight="bold", color="darkblue",
        )
        ax_query.axis("off")
        for spine in ax_query.spines.values():
            spine.set_visible(True)
            spine.set_color("blue")
            spine.set_linewidth(3)

        # ── Top-right: Conditions summary ──
        ax_text = fig.add_subplot(gs[0, cols_for_query:])
        ax_text.axis("off")

        text_lines = ["Detected Conditions", "─" * 30]

        if detected_conditions:
            pathological = {
                k: v for k, v in detected_conditions.items()
                if k not in ("normal", "support_devices")
            }
            if pathological:
                for cond, info in sorted(
                    pathological.items(),
                    key=lambda x: x[1].get(
                        "avg_score",
                        x[1].get("weighted_probability", 0),
                    ),
                    reverse=True,
                ):
                    label = cond.replace("_", " ").capitalize()
                    freq = info.get("frequency", info.get("weighted_probability", 0)) * 100
                    conf = info.get("avg_score", info.get("weighted_probability", 0))
                    bar = "█" * int(freq / 10) + "░" * (10 - int(freq / 10))
                    text_lines.append(
                        f"• {label}\n  {bar} {freq:.0f}%\n  Confidence: {conf:.2f}"
                    )
            else:
                text_lines.append("✅ No acute abnormality detected")
                if "normal" in detected_conditions:
                    freq = detected_conditions["normal"].get(
                        "frequency",
                        detected_conditions["normal"].get("weighted_probability", 0),
                    ) * 100
                    text_lines.append(f"Normal in {freq:.0f}% of matches")

            if "support_devices" in detected_conditions:
                freq = detected_conditions["support_devices"].get(
                    "frequency",
                    detected_conditions["support_devices"].get("weighted_probability", 0),
                ) * 100
                text_lines.append(f"\nDevices noted ({freq:.0f}%)")
        else:
            text_lines.append("No condition analysis available")

        scores = [r.get("score", 0) for r in top_results]
        if scores:
            text_lines.extend([
                "", "─" * 30, "Retrieval Stats",
                f"Cases shown: {n_results}",
                f"Top score: {max(scores):.4f}",
                f"Mean score: {np.mean(scores):.4f}",
                f"Score range: {min(scores):.4f}–{max(scores):.4f}",
                f"Threshold: >= {SCORE_THRESHOLD}",
            ])

        ax_text.text(
            0.05, 0.95,
            "\n".join(text_lines),
            transform=ax_text.transAxes,
            verticalalignment="top",
            fontsize=9,
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#f8f8f8",
                edgecolor="#cccccc",
            ),
        )

        # ── Bottom row: Retrieved images ──
        for i, result in enumerate(top_results[:effective_cols]):
            ax = fig.add_subplot(gs[1, i])
            retrieved_img = self._load_image_by_index(result["original_index"])

            if retrieved_img is not None:
                ax.imshow(retrieved_img)
            else:
                ax.text(
                    0.5, 0.5, "Image\nN/A",
                    ha="center", va="center",
                    fontsize=11, color="gray",
                    transform=ax.transAxes,
                )
                ax.set_facecolor("#f5f5f5")

            score = result.get("score", 0)
            rank = result.get("rank", i + 1)
            color = (
                "darkgreen" if score >= 0.8
                else "orange" if score >= 0.6
                else "darkorange" if score >= 0.4
                else "red"
            )

            ax.set_title(
                f"#{rank}  {score:.3f}",
                fontsize=11, fontweight="bold", color=color,
            )

            caption = result.get("caption", "")
            wrapped = self._wrap_text(caption, max_chars=45, max_lines=3)
            ax.set_xlabel(wrapped, fontsize=7)
            ax.tick_params(
                left=False, bottom=False,
                labelleft=False, labelbottom=False,
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2)

        # Hide unused bottom slots
        for i in range(len(top_results), effective_cols):
            ax = fig.add_subplot(gs[1, i])
            ax.axis("off")

        if save_path is None:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "detailed_visual_report.png")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"Detailed visual report saved: {save_path}")
        return save_path

    # ---------------------------------------------------------- #
    #  PIL-based gallery (no matplotlib needed)
    # ---------------------------------------------------------- #

    def create_pil_gallery(
        self,
        query_image: Image.Image,
        retrieval_results: List[Dict],
        image_size: int = 300,
        padding: int = 15,
        bg_color: str = "white",
    ) -> Image.Image:
        """
        Create a pure-PIL gallery image (no matplotlib dependency).
        Shows top 3-5 similar cases side-by-side.
        """
        # Filter and take top 3-5
        filtered = [
            r for r in retrieval_results
            if r.get("score", 0) >= SCORE_THRESHOLD
        ]
        if not filtered:
            filtered = retrieval_results[:1] if retrieval_results else []
        top_results = filtered[:5]

        n_results = len(top_results)
        total_items = 1 + n_results

        n_cols = min(3, total_items)
        n_rows = (total_items + n_cols - 1) // n_cols

        cell_w = image_size + 2 * padding
        cell_h = image_size + 80 + 2 * padding

        canvas_w = n_cols * cell_w + padding
        canvas_h = n_rows * cell_h + padding + 50

        canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
        draw = ImageDraw.Draw(canvas)

        try:
            font_title = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
            )
            font_score = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12
            )
            font_caption = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9
            )
            font_header = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
            )
        except Exception:
            font_title = ImageFont.load_default()
            font_score = ImageFont.load_default()
            font_caption = ImageFont.load_default()
            font_header = ImageFont.load_default()

        draw.text(
            (padding, padding),
            "Medical Image Retrieval — Visual Comparison",
            fill="darkblue", font=font_header,
        )

        y_offset = 50

        # ── Query image ──
        self._place_image_cell(
            canvas, draw, query_image,
            x=padding, y=y_offset,
            cell_w=cell_w, cell_h=cell_h,
            image_size=image_size, padding=padding,
            title="QUERY IMAGE",
            subtitle="",
            caption="",
            border_color="blue",
            font_title=font_title,
            font_score=font_score,
            font_caption=font_caption,
        )

        # ── Retrieved images ──
        for i, result in enumerate(top_results):
            item_idx = i + 1
            row = item_idx // n_cols
            col = item_idx % n_cols

            x = padding + col * cell_w
            y = y_offset + row * cell_h

            score = result.get("score", 0)
            rank = result.get("rank", i + 1)
            caption = result.get("caption", "No caption")

            retrieved_img = self._load_image_by_index(result["original_index"])

            if score >= 0.8:
                border_color = "green"
            elif score >= 0.6:
                border_color = "orange"
            elif score >= 0.4:
                border_color = "darkorange"
            else:
                border_color = "red"

            self._place_image_cell(
                canvas, draw,
                retrieved_img,
                x=x, y=y,
                cell_w=cell_w, cell_h=cell_h,
                image_size=image_size, padding=padding,
                title=f"Match #{rank}",
                subtitle=f"Score: {score:.4f}",
                caption=caption[:150],
                border_color=border_color,
                font_title=font_title,
                font_score=font_score,
                font_caption=font_caption,
            )

        return canvas

    def _create_pil_gallery_and_save(
        self,
        query_image: Image.Image,
        retrieval_results: List[Dict],
        save_path: Optional[str] = None,
        output_dir: str = "./output",
    ) -> Optional[str]:
        """Create PIL gallery and save to disk."""
        gallery = self.create_pil_gallery(query_image, retrieval_results)

        if save_path is None:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "visual_gallery.png")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        gallery.save(save_path)
        logger.info(f"PIL gallery saved: {save_path}")
        return save_path

    def _place_image_cell(
        self,
        canvas: Image.Image,
        draw: ImageDraw.Draw,
        image: Optional[Image.Image],
        x: int, y: int,
        cell_w: int, cell_h: int,
        image_size: int, padding: int,
        title: str, subtitle: str, caption: str,
        border_color: str,
        font_title, font_score, font_caption,
    ):
        draw.rectangle(
            [x, y, x + cell_w - 5, y + cell_h - 5],
            outline=border_color, width=3, fill="#fafafa",
        )
        draw.text(
            (x + padding, y + 5),
            title, fill=border_color, font=font_title,
        )
        if subtitle:
            draw.text(
                (x + padding, y + 22),
                subtitle, fill="gray", font=font_score,
            )
        img_y = y + 40
        if image is not None:
            resized = image.resize(
                (image_size - 2 * padding, image_size - 2 * padding),
                Image.LANCZOS,
            )
            canvas.paste(resized, (x + padding, img_y))
        else:
            draw.rectangle(
                [x + padding, img_y,
                 x + image_size - padding, img_y + image_size - 2 * padding],
                fill="#e0e0e0", outline="gray",
            )
            draw.text(
                (x + image_size // 3, img_y + image_size // 3),
                "No Image", fill="gray", font=font_score,
            )
        if caption:
            caption_y = img_y + image_size - 2 * padding + 5
            wrapped = self._wrap_text(caption, max_chars=40, max_lines=3)
            draw.text(
                (x + padding, caption_y),
                wrapped, fill="black", font=font_caption,
            )

    # ---------------------------------------------------------- #
    #  Text-based visual report (always works)
    # ---------------------------------------------------------- #

    def create_text_visual_report(
        self,
        retrieval_results: List[Dict],
    ) -> str:
        """
        Text-only visual report with ASCII similarity bars.
        Shows top 3-5 results with score >= 0.3.
        """
        # Filter and take top 3-5
        filtered = [
            r for r in retrieval_results
            if r.get("score", 0) >= SCORE_THRESHOLD
        ]
        if not filtered:
            filtered = retrieval_results[:1] if retrieval_results else []
        top_results = filtered[:5]

        lines = []
        lines.append("=" * 65)
        lines.append("  VISUAL RETRIEVAL REPORT")
        lines.append(
            f"  Top {len(top_results)} similar cases "
            f"(score >= {SCORE_THRESHOLD})"
        )
        lines.append("=" * 65)
        lines.append("")
        lines.append("  Query Image -> Top Similar Cases Retrieved")
        lines.append("")

        for i, result in enumerate(top_results):
            score = result.get("score", 0)
            rank = result.get("rank", i + 1)
            caption = result.get("caption", "No caption")
            orig_idx = result.get("original_index", "?")

            bar_len = int(score * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)

            if score >= 0.8:
                label = "Very Similar"
            elif score >= 0.6:
                label = "Similar"
            elif score >= 0.4:
                label = "Moderate"
            else:
                label = "Weak"

            lines.append(f"  ┌─── Match #{rank} {'─' * 45}")
            lines.append(f"  │ Score:    {score:.4f}  {label}")
            lines.append(f"  │ Bar:      [{bar}]")
            lines.append(f"  │ Image ID: {orig_idx}")
            lines.append(f"  │")
            lines.append(f"  │ Caption:")

            words = caption.split()
            line = "  │   "
            for word in words:
                if len(line) + len(word) + 1 > 63:
                    lines.append(line)
                    line = "  │   "
                line += word + " "
            if line.strip() != "│":
                lines.append(line)

            lines.append(f"  └{'─' * 60}")
            lines.append("")

        all_scores = [r.get("score", 0) for r in top_results]
        if all_scores:
            lines.append("─" * 65)
            lines.append(f"  Retrieved: {len(top_results)} cases")
            lines.append(
                f"  Score range: {min(all_scores):.4f} – {max(all_scores):.4f}"
            )
            lines.append(f"  Mean score: {np.mean(all_scores):.4f}")
        lines.append("=" * 65)

        return "\n".join(lines)

    # ---------------------------------------------------------- #
    #  Utility
    # ---------------------------------------------------------- #

    @staticmethod
    def _wrap_text(text: str, max_chars: int = 50, max_lines: int = 3) -> str:
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
                if len(lines) >= max_lines - 1:
                    current_line += "..."
                    break
        if current_line:
            lines.append(current_line)
        return "\n".join(lines[:max_lines])