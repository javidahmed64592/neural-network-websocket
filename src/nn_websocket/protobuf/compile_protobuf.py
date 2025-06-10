import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%d-%m-%Y|%H:%M:%S]", level=logging.DEBUG)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROTO_DIR = PROJECT_ROOT / "protobuf"
OUT_DIR = PROJECT_ROOT / "src" / "nn_websocket" / "protobuf" / "compiled"


def compile_protobuf() -> bool:
    """Compile protobuf files to Python classes."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not (proto_files := list(PROTO_DIR.glob("*.proto"))):
        logger.warning("No .proto files found in the protobuf directory.")
        return False

    logger.info("Generating Protobuf files...")
    logger.info("Protobuf directory: %s", PROTO_DIR)
    logger.info("Output directory: %s", OUT_DIR)

    try:
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={PROTO_DIR}",
            f"--python_out={OUT_DIR}",
            f"--pyi_out={OUT_DIR}",
            f"{PROTO_DIR}/*.proto",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, check=False)  # noqa: S603

        if result.returncode != 0:
            logger.error("Error during protobuf compilation:")
            logger.error(result.stderr)
            return False
    except Exception:
        logger.exception("Error during protobuf compilation")
        return False
    else:
        logger.info("Protobuf generation complete!")
        logger.info("Generated files for: %s", [f.name for f in proto_files])
        return True


def main() -> None:
    sys.exit(0 if compile_protobuf() else 1)
