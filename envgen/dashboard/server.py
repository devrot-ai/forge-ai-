"""Flask-based web dashboard for visualizing training experiment results.

Reads JSON-lines metrics files from experiment log directories and provides
a web interface for viewing and comparing training runs. Uses Chart.js for
plotting rewards, skill scores, and difficulty over time.

Requires Flask: pip install flask

Example:
    server = DashboardServer(log_dir="./logs", port=8080)
    server.run()
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from flask import Flask, jsonify, render_template, request
except ImportError:
    Flask = None  # type: ignore
    render_template = None  # type: ignore
    request = None  # type: ignore
    jsonify = None  # type: ignore


class DashboardServer:
    """Web dashboard server for experiment visualization.

    Reads experiment log directories created by ExperimentLogger and serves
    a web interface for viewing metrics and comparing runs.

    Args:
        log_dir: Base directory containing experiment run folders
        port: HTTP port to serve on (default: 8080)
        host: Host address to bind to (default: 127.0.0.1)
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        port: int = 8080,
        host: str = "127.0.0.1",
    ):
        """Initialize the dashboard server.

        Args:
            log_dir: Path to the logs directory containing run folders
            port: Port number for the HTTP server
            host: Host address to bind to
        """
        if Flask is None:
            raise ImportError("Flask is required for the dashboard. Install it with: pip install flask")

        self.log_dir = log_dir
        self.port = port
        self.host = host

        # Create Flask app
        self.app = Flask(
            __name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        )

        # Register routes
        self.app.add_url_rule("/", "index", self._index)
        self.app.add_url_rule("/api/runs", "api_runs", self._api_runs)
        self.app.add_url_rule("/api/run/<run_name>", "api_run", self._api_run)
        self.app.add_url_rule("/api/run/<run_name>/metrics", "api_metrics", self._api_metrics)
        self.app.add_url_rule("/api/compare", "api_compare", self._api_compare)

    def _index(self) -> str:
        """Serve the main dashboard page."""
        runs = self._list_runs()
        return render_template("index.html", runs=runs)

    def _api_runs(self) -> Any:
        """API endpoint: List all available runs."""
        runs = self._list_runs()
        return jsonify({"runs": runs})

    def _api_run(self, run_name: str) -> Any:
        """API endpoint: Get details for a specific run."""
        run_data = self._load_run(run_name)
        if run_data is None:
            return jsonify({"error": f"Run '{run_name}' not found"}), 404
        return jsonify(run_data)

    def _api_metrics(self, run_name: str) -> Any:
        """API endpoint: Get metrics for a specific run."""
        metrics = self._load_metrics(run_name)
        if metrics is None:
            return jsonify({"error": f"Run '{run_name}' not found"}), 404
        return jsonify({"metrics": metrics})

    def _api_compare(self) -> Any:
        """API endpoint: Compare multiple runs."""
        run_names = request.args.getlist("runs")
        if not run_names:
            return jsonify({"error": "No runs specified. Use ?runs=run1&runs=run2"}), 400

        comparison = []
        for run_name in run_names:
            run_data = self._load_run(run_name)
            if run_data:
                comparison.append(run_data)

        return jsonify({"comparison": comparison})

    def _list_runs(self) -> List[Dict[str, Any]]:
        """List all available experiment runs.

        Returns:
            List of run info dicts with name, timestamp, and cycle count
        """
        runs = []
        if not os.path.isdir(self.log_dir):
            return runs

        for entry in sorted(os.listdir(self.log_dir)):
            run_dir = os.path.join(self.log_dir, entry)
            if not os.path.isdir(run_dir):
                continue

            summary_file = os.path.join(run_dir, "summary.json")
            summary = {}
            if os.path.isfile(summary_file):
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass

            runs.append(
                {
                    "name": entry,
                    "total_cycles": summary.get("total_cycles", 0),
                    "total_time_seconds": summary.get("total_time_seconds", 0),
                    "final_skill_scores": summary.get("final_skill_scores", {}),
                    "mean_reward": summary.get("aggregate", {}).get("mean_reward_across_cycles"),
                }
            )

        return runs

    def _load_run(self, run_name: str) -> Optional[Dict[str, Any]]:
        """Load all data for a specific run.

        Args:
            run_name: Name of the run folder

        Returns:
            Dict with run info, config, summary, and metrics, or None if not found
        """
        run_dir = os.path.join(self.log_dir, run_name)
        if not os.path.isdir(run_dir):
            return None

        result: Dict[str, Any] = {"name": run_name}

        # Load summary
        summary_file = os.path.join(run_dir, "summary.json")
        if os.path.isfile(summary_file):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    result["summary"] = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Load config
        config_file = os.path.join(run_dir, "config.json")
        if os.path.isfile(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    result["config"] = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Load metrics
        result["metrics"] = self._load_metrics(run_name)

        return result

    def _load_metrics(self, run_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load metrics from a run's JSON lines file.

        Args:
            run_name: Name of the run folder

        Returns:
            List of metric dicts, or None if not found
        """
        metrics_file = os.path.join(self.log_dir, run_name, "metrics.jsonl")
        if not os.path.isfile(metrics_file):
            return None

        metrics = []
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        metric = json.loads(line)
                        metrics.append(metric)
                    except json.JSONDecodeError:
                        continue
        except IOError:
            return None

        return metrics

    def run(self, debug: bool = False) -> None:
        """Start the dashboard server.

        Args:
            debug: Enable Flask debug mode (default: False)
        """
        print(f"Dashboard server starting at http://{self.host}:{self.port}")
        print(f"Watching logs in: {os.path.abspath(self.log_dir)}")
        self.app.run(host=self.host, port=self.port, debug=debug)
