"""CLI dashboard command — launch the web dashboard for experiment visualization."""

from __future__ import annotations

import argparse
import logging
import sys


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def dashboard(args: argparse.Namespace) -> None:
    """Launch the web dashboard from CLI arguments.

    Args:
        args: Parsed command-line arguments.
    """
    setup_logging(args.verbose)

    try:
        from envgen.dashboard.server import DashboardServer
    except ImportError as e:
        print(
            "Error: Flask is required for the dashboard.\n"
            "Install it with: pip install flask\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    server = DashboardServer(
        log_dir=args.log_dir,
        port=args.port,
        host=args.host,
    )

    try:
        server.run(debug=args.debug)
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")


def main():
    """Main entry point for the dashboard CLI."""
    parser = argparse.ArgumentParser(
        prog="envgen",
        description="EnvGen-LLM: Self-Improving AI Agent Training",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Launch web dashboard")
    dash_parser.add_argument(
        "--log-dir",
        "-l",
        default="./logs",
        help="Directory containing experiment logs (default: ./logs)",
    )
    dash_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Port to serve on (default: 8080)",
    )
    dash_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)",
    )
    dash_parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable Flask debug mode",
    )
    dash_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.command == "dashboard":
        dashboard(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
