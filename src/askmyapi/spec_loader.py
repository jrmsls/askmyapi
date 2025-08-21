import hashlib
import json
import logging
from typing import Any, Dict, Tuple

import yaml
from prance import ResolvingParser
from openapi_spec_validator import validate_spec

logger = logging.getLogger(__name__)


def _read_raw(path: str) -> Dict[str, Any]:
    """Read a JSON or YAML file into a Python dictionary without resolving $refs."""
    logger.debug("Reading raw spec file: %s", path)
    if path.endswith((".yaml", ".yml")):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    logger.debug(
        "Raw spec keys: %s",
        list(data.keys()) if isinstance(data, dict) else type(data),
    )
    return data


def load_and_deref_spec(
    path: str, *, validate: bool = True
) -> Tuple[Dict[str, Any], str]:
    """
    Load (JSON/YAML), resolve $refs (internal & external), optionally validate the spec.

    Returns:
        (spec, spec_hash): the dereferenced spec dict and a short stable hash (12 hex chars)
                           used to scope caches and vectorstore collections.
    """
    logger.info("Loading and dereferencing spec: %s", path)

    # Compute a stable hash of the raw file for cache scoping.
    raw = _read_raw(path)
    spec_str = json.dumps(raw, sort_keys=True, ensure_ascii=False)
    spec_hash = hashlib.sha256(spec_str.encode("utf-8")).hexdigest()[:12]
    logger.debug("Computed spec hash: %s", spec_hash)

    # ResolvingParser re-reads the file and expands $refs.
    parser = ResolvingParser(path, resolve=True)
    deref = parser.specification  # resolved dictionary
    logger.info(
        "Spec successfully dereferenced, top-level keys: %s",
        list(deref.keys()),
    )

    if validate:
        logger.info("Validating spec...")
        try:
            validate_spec(deref)
            logger.info("Spec validation succeeded")
        except Exception as e:
            logger.exception("Spec validation failed")
            raise

    return deref, spec_hash
