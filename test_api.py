# test_api.py
"""
Example script showing how to use the Medical Image Retrieval API
from Python code.

Usage:
    1. Start the API server first:
       python api.py --vlm_backend local --vlm_4bit

    2. Run this script:
       python test_api.py --image chest_xray.jpg
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Optional, Dict


# ================================================================== #
#  API Client
# ================================================================== #

class MedVisionAPIClient:
    """
    Python client for the Medical Image Retrieval API.

    Usage:
        client = MedVisionAPIClient("http://localhost:8000")

        # Check health
        client.health()

        # Upload + generate report
        result = client.generate_report("chest_xray.jpg")
        print(result["report"])
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    # ---------------------------------------------------------- #
    #  Health & Status
    # ---------------------------------------------------------- #

    def health(self) -> Dict:
        """Check API health and VLM status."""
        resp = self.session.get(f"{self.base_url}/api/health")
        resp.raise_for_status()
        return resp.json()

    def status(self) -> Dict:
        """Get detailed pipeline and VLM status."""
        resp = self.session.get(f"{self.base_url}/api/status")
        resp.raise_for_status()
        return resp.json()

    # ---------------------------------------------------------- #
    #  VLM Model Management
    # ---------------------------------------------------------- #

    def vlm_status(self) -> Dict:
        """Check MedGemma download/load status."""
        resp = self.session.get(
            f"{self.base_url}/api/vlm/status"
        )
        resp.raise_for_status()
        return resp.json()

    def vlm_download(self) -> Dict:
        """
        Download MedGemma model (~8 GB).
        Only needed once. Blocks until complete.
        """
        print("Downloading MedGemma model (this may take 10-30 min)...")
        resp = self.session.post(
            f"{self.base_url}/api/vlm/download",
            timeout=3600,  # 1 hour timeout for download
        )
        resp.raise_for_status()
        return resp.json()

    def vlm_load(self) -> Dict:
        """
        Pre-load MedGemma into GPU/CPU memory.
        Optional — model auto-loads on first query.
        """
        print("Loading MedGemma into memory...")
        resp = self.session.post(
            f"{self.base_url}/api/vlm/load",
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    def vlm_unload(self) -> Dict:
        """Free MedGemma from GPU/CPU memory."""
        resp = self.session.post(
            f"{self.base_url}/api/vlm/unload"
        )
        resp.raise_for_status()
        return resp.json()

    # ---------------------------------------------------------- #
    #  Pipeline Management
    # ---------------------------------------------------------- #

    def load_pipeline(
        self, checkpoint: str = "final_model"
    ) -> Dict:
        """Load CLIP checkpoint and FAISS index."""
        resp = self.session.post(
            f"{self.base_url}/api/load",
            data={"checkpoint": checkpoint},
        )
        resp.raise_for_status()
        return resp.json()

    def build_index(self) -> Dict:
        """Rebuild the FAISS index."""
        resp = self.session.post(
            f"{self.base_url}/api/index/build",
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()

    # ---------------------------------------------------------- #
    #  Core: Retrieve Similar Images
    # ---------------------------------------------------------- #

    def retrieve(
        self,
        image_path: str,
        top_k: int = 3,
        text_query: Optional[str] = None,
        min_score: float = 0.3,
        use_query_expansion: bool = True,
        use_reranking: bool = True,
    ) -> Dict:
        """
        Retrieve top-K similar images and their captions.

        Args:
            image_path: Path to query image
            top_k: Number of results to return
            text_query: Optional text to combine with image
            min_score: Minimum similarity score threshold
            use_query_expansion: Enable query expansion
            use_reranking: Enable re-ranking

        Returns:
            Dict with query_id, results list, etc.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with open(image_path, "rb") as f:
            files = {"image": (
                os.path.basename(image_path), f, "image/jpeg"
            )}
            data = {
                "top_k": top_k,
                "min_score": min_score,
                "use_query_expansion": use_query_expansion,
                "use_reranking": use_reranking,
            }
            if text_query:
                data["text_query"] = text_query

            resp = self.session.post(
                f"{self.base_url}/api/retrieve",
                files=files,
                data=data,
                timeout=120,
            )

        resp.raise_for_status()
        return resp.json()

    # ---------------------------------------------------------- #
    #  Core: Generate Report
    # ---------------------------------------------------------- #

    def generate_report(
        self,
        image_path: str,
        report_method: str = "vlm_few_shot",
        top_k: int = 3,
        min_score: float = 0.3,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        num_examples: Optional[int] = None,
    ) -> Dict:
        """
        Generate a medical report from a query image.

        Args:
            image_path: Path to query image
            report_method: One of:
                - "vlm_few_shot"    — MedGemma + retrieved images
                - "vlm_zero_shot"   — MedGemma without examples
                - "ollama_few_shot" — Ollama + captions
                - "template"        — Template with weighted probs
                - "visual"          — Side-by-side gallery
                - "weighted"        — Weighted retrieval
                - "majority"        — Majority vote
                - "concat"          — Concatenated captions
            top_k: Number of similar cases to retrieve
            min_score: Minimum similarity score
            temperature: Override VLM temperature
            max_tokens: Override max generation tokens
            num_examples: Override number of few-shot examples

        Returns:
            Dict with report, method, success, timing, etc.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with open(image_path, "rb") as f:
            files = {"image": (
                os.path.basename(image_path), f, "image/jpeg"
            )}
            data = {
                "report_method": report_method,
                "top_k": top_k,
                "min_score": min_score,
            }
            if temperature is not None:
                data["temperature"] = temperature
            if max_tokens is not None:
                data["max_tokens"] = max_tokens
            if num_examples is not None:
                data["num_examples"] = num_examples

            resp = self.session.post(
                f"{self.base_url}/api/generate-report",
                files=files,
                data=data,
                timeout=300,  # VLM can take a while
            )

        resp.raise_for_status()
        return resp.json()

    # ---------------------------------------------------------- #
    #  Submit Feedback
    # ---------------------------------------------------------- #

    def submit_feedback(
        self,
        image_path: str,
        feedback: str,
        generated_caption: str = "",
        query_id: Optional[str] = None,
        report_method: Optional[str] = None,
        add_to_index: bool = True,
    ) -> Dict:
        """
        Submit user feedback for a query.
        Optionally adds the feedback caption to the FAISS index.

        Args:
            image_path: Path to the query image
            feedback: User feedback text
            generated_caption: Caption to store in index
            query_id: ID from a previous query
            report_method: Method used for the report
            add_to_index: Whether to add to FAISS index
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with open(image_path, "rb") as f:
            files = {"image": (
                os.path.basename(image_path), f, "image/jpeg"
            )}
            data = {
                "feedback": feedback,
                "generated_caption": generated_caption,
                "add_to_index": add_to_index,
            }
            if query_id:
                data["query_id"] = query_id
            if report_method:
                data["report_method"] = report_method

            resp = self.session.post(
                f"{self.base_url}/api/feedback",
                files=files,
                data=data,
                timeout=120,
            )

        resp.raise_for_status()
        return resp.json()

    # ---------------------------------------------------------- #
    #  Append Data to Index
    # ---------------------------------------------------------- #

    def append_to_index(
        self,
        data_folder: str,
        rebuild_after: bool = True,
    ) -> Dict:
        """
        Append new image-caption pairs from a CSV folder.

        Args:
            data_folder: Path to folder with CSV + images
            rebuild_after: Save index after appending
        """
        resp = self.session.post(
            f"{self.base_url}/api/index/append",
            data={
                "data_folder": data_folder,
                "rebuild_after": rebuild_after,
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()

    # ---------------------------------------------------------- #
    #  Get Feedback Log
    # ---------------------------------------------------------- #

    def get_feedback_log(
        self,
        date: Optional[str] = None,
        limit: int = 100,
    ) -> Dict:
        """
        Retrieve the feedback log.

        Args:
            date: Filter by date (YYYY-MM-DD)
            limit: Max entries to return
        """
        params = {"limit": limit}
        if date:
            params["date"] = date

        resp = self.session.get(
            f"{self.base_url}/api/feedback/log",
            params=params,
        )
        resp.raise_for_status()
        return resp.json()


# ================================================================== #
#  Helper: Pretty Print
# ================================================================== #

def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def print_json(data: Dict, indent: int = 2):
    print(json.dumps(data, indent=indent, default=str))


def print_report(result: Dict):
    """Pretty-print a report generation result."""
    print_header("GENERATED REPORT")

    if result.get("success"):
        print(result.get("report", "No report"))
        print()
        print(f"  Method:       {result.get('method')}")
        print(f"  Model:        {result.get('model')}")
        print(f"  Backend:      {result.get('backend')}")
        print(f"  Examples:     {result.get('num_examples', 0)}")
        print(f"  Images sent:  {result.get('num_images_sent', 0)}")
        print(f"  Time:         {result.get('generation_time_s')}s")
    else:
        print(f"  ❌ FAILED: {result.get('error')}")

    if result.get("retrieval_results"):
        print()
        print("  Retrieved cases:")
        for r in result["retrieval_results"]:
            print(
                f"    [{r['rank']}] score={r['score']:.4f}  "
                f"{r['caption'][:80]}..."
            )


def print_retrieval(result: Dict):
    """Pretty-print retrieval results."""
    print_header("RETRIEVAL RESULTS")
    print(f"  Query ID:  {result.get('query_id')}")
    print(f"  Results:   {result.get('num_results')}")
    print()

    for r in result.get("results", []):
        score = r["score"]
        if score >= 0.8:
            label = "🟢"
        elif score >= 0.6:
            label = "🟡"
        elif score >= 0.4:
            label = "🟠"
        else:
            label = "🔴"

        print(
            f"  {label} [{r['rank']}] "
            f"score={score:.4f}  "
            f"{r['caption'][:80]}"
        )


# ================================================================== #
#  Example Workflows
# ================================================================== #

def workflow_first_time_setup(client: MedVisionAPIClient):
    """Complete first-time setup: download model, load pipeline."""
    print_header("FIRST-TIME SETUP")

    # 1. Check health
    print("\n1. Checking API health...")
    health = client.health()
    print(f"   Status: {health['status']}")
    print(f"   Pipeline ready: {health['pipeline_ready']}")
    print(f"   VLM cached: {health['vlm_model_cached']}")

    # 2. Check VLM status
    print("\n2. Checking MedGemma status...")
    vlm = client.vlm_status()
    print(f"   Model: {vlm.get('model_name')}")
    print(f"   Cached: {vlm.get('is_cached')}")
    print(f"   In memory: {vlm.get('model_in_memory')}")

    # 3. Download if needed
    if not vlm.get("is_cached"):
        print("\n3. Downloading MedGemma model...")
        result = client.vlm_download()
        print(f"   Status: {result['status']}")
        print(f"   Path: {result.get('path')}")
    else:
        print("\n3. Model already downloaded ✅")

    # 4. Load pipeline if needed
    if not health["pipeline_ready"]:
        print("\n4. Loading pipeline...")
        result = client.load_pipeline("final_model")
        print(f"   Status: {result['status']}")
    else:
        print("\n4. Pipeline already loaded ✅")

    # 5. Pre-load VLM (optional)
    if not vlm.get("model_in_memory"):
        print("\n5. Pre-loading MedGemma into memory...")
        result = client.vlm_load()
        print(f"   Status: {result['status']}")
    else:
        print("\n5. Model already in memory ✅")

    print("\n✅ Setup complete! Ready for queries.")


def workflow_generate_report(
    client: MedVisionAPIClient,
    image_path: str,
    method: str = "vlm_few_shot",
    top_k: int = 3,
):
    """Generate a report for a single image."""
    print_header(f"GENERATING REPORT ({method})")
    print(f"  Image: {image_path}")
    print(f"  Method: {method}")
    print(f"  Top-K: {top_k}")

    start = time.time()
    result = client.generate_report(
        image_path=image_path,
        report_method=method,
        top_k=top_k,
    )
    elapsed = time.time() - start

    print_report(result)
    print(f"\n  Total API call time: {elapsed:.1f}s")

    return result


def workflow_compare_methods(
    client: MedVisionAPIClient,
    image_path: str,
):
    """Compare all report methods on the same image."""
    print_header("COMPARING ALL REPORT METHODS")
    print(f"  Image: {image_path}")

    methods = [
        ("template", "Template (no VLM)"),
        ("vlm_few_shot", "MedGemma Few-Shot"),
        ("vlm_zero_shot", "MedGemma Zero-Shot"),
        ("weighted", "Weighted Retrieval"),
    ]

    results = {}
    for method, label in methods:
        print(f"\n{'─' * 65}")
        print(f"  Running: {label} ({method})")
        print(f"{'─' * 65}")

        try:
            start = time.time()
            result = client.generate_report(
                image_path=image_path,
                report_method=method,
                top_k=3,
            )
            elapsed = time.time() - start

            results[method] = result

            if result.get("success"):
                # Print first 200 chars of report
                report_preview = result["report"][:200]
                print(f"  ✅ Success ({elapsed:.1f}s)")
                print(f"  Preview: {report_preview}...")
            else:
                print(f"  ❌ Failed: {result.get('error')}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[method] = {"success": False, "error": str(e)}

    # Summary
    print_header("COMPARISON SUMMARY")
    for method, label in methods:
        r = results.get(method, {})
        status = "✅" if r.get("success") else "❌"
        gen_time = r.get("generation_time_s", "?")
        print(f"  {status} {label:30s} — {gen_time}s")

    return results


def workflow_retrieve_and_feedback(
    client: MedVisionAPIClient,
    image_path: str,
):
    """Retrieve similar images, then submit feedback."""
    print_header("RETRIEVE + FEEDBACK WORKFLOW")

    # 1. Retrieve
    print("\n1. Retrieving similar images...")
    retrieval = client.retrieve(
        image_path=image_path,
        top_k=5,
        min_score=0.3,
    )
    print_retrieval(retrieval)

    # 2. Generate report
    print("\n2. Generating VLM report...")
    report = client.generate_report(
        image_path=image_path,
        report_method="vlm_few_shot",
        top_k=3,
    )
    print_report(report)

    # 3. Submit feedback
    print("\n3. Submitting feedback...")
    feedback_result = client.submit_feedback(
        image_path=image_path,
        feedback=(
            "Report is accurate. Correctly identified "
            "cardiomegaly and pleural effusions."
        ),
        generated_caption=report.get("raw_vlm_output", ""),
        query_id=report.get("query_id"),
        report_method="vlm_few_shot",
        add_to_index=True,
    )
    print(f"   Feedback logged: {feedback_result['query_id']}")
    print(
        f"   Added to index: "
        f"{feedback_result['added_to_index']}"
    )
    print(
        f"   Index size: {feedback_result['index_size']}"
    )

    return retrieval, report, feedback_result


# ================================================================== #
#  Main
# ================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="MedVision API Client — Example Usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First-time setup (download model + load pipeline)
  python test_api.py --setup

  # Generate a MedGemma few-shot report
  python test_api.py --image chest_xray.jpg

  # Generate a template report (no VLM needed)
  python test_api.py --image chest_xray.jpg --method template

  # Compare all methods
  python test_api.py --image chest_xray.jpg --compare

  # Retrieve similar images only
  python test_api.py --image chest_xray.jpg --retrieve_only

  # Full workflow: retrieve → report → feedback
  python test_api.py --image chest_xray.jpg --full_workflow

  # Check status
  python test_api.py --status
        """,
    )

    parser.add_argument(
        "--api_url", type=str,
        default="http://localhost:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to query image",
    )
    parser.add_argument(
        "--method", type=str, default="vlm_few_shot",
        choices=[
            "vlm_few_shot", "vlm_zero_shot",
            "ollama_few_shot", "ollama_zero_shot",
            "template", "visual",
            "weighted", "majority", "concat",
        ],
        help="Report generation method",
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="Number of similar cases to retrieve",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Override VLM temperature",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="Override max generation tokens",
    )

    # Workflow modes
    parser.add_argument(
        "--setup", action="store_true",
        help="Run first-time setup (download + load)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print API and VLM status",
    )
    parser.add_argument(
        "--retrieve_only", action="store_true",
        help="Only retrieve similar images (no report)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare all report methods",
    )
    parser.add_argument(
        "--full_workflow", action="store_true",
        help="Run retrieve → report → feedback",
    )
    parser.add_argument(
        "--unload", action="store_true",
        help="Unload VLM model from memory",
    )

    args = parser.parse_args()

    # Create client
    client = MedVisionAPIClient(args.api_url)

    # ── Check connection ──
    try:
        health = client.health()
        print(f"✅ Connected to API at {args.api_url}")
    except requests.ConnectionError:
        print(f"❌ Cannot connect to API at {args.api_url}")
        print(f"   Start the server first:")
        print(f"   python api.py --vlm_backend local")
        sys.exit(1)

    # ── Setup mode ──
    if args.setup:
        workflow_first_time_setup(client)
        return

    # ── Status mode ──
    if args.status:
        print_header("API HEALTH")
        print_json(client.health())

        print_header("DETAILED STATUS")
        print_json(client.status())

        print_header("VLM MODEL STATUS")
        print_json(client.vlm_status())
        return

    # ── Unload mode ──
    if args.unload:
        print("Unloading VLM model...")
        result = client.vlm_unload()
        print(f"Status: {result['status']}")
        return

    # ── Image-based modes ──
    if args.image is None:
        print("❌ --image is required for this mode.")
        print("   Example: python test_api.py --image chest_xray.jpg")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)

    # Compare all methods
    if args.compare:
        workflow_compare_methods(client, args.image)
        return

    # Full workflow
    if args.full_workflow:
        workflow_retrieve_and_feedback(client, args.image)
        return

    # Retrieve only
    if args.retrieve_only:
        result = client.retrieve(
            image_path=args.image,
            top_k=args.top_k,
        )
        print_retrieval(result)
        return

    # Default: generate report
    workflow_generate_report(
        client=client,
        image_path=args.image,
        method=args.method,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()