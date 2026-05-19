import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def unpack_parser_result(
    parser_result,
    ctx: dict[str, Any],
    check_name: str,
    raw_output: str,
) -> tuple[Any, Any]:
    if parser_result is None:
        preview = raw_output.replace("\n", "\\n")[:500]
        raise ValueError(
            f"Parser returned None for '{check_name}' on "
            f"{ctx['device_ip']} (raw_len={len(raw_output)}, "
            f"raw_preview={preview!r})"
        )

    try:
        status, output = parser_result
    except TypeError as ex:
        preview = raw_output.replace("\n", "\\n")[:500]
        raise ValueError(
            f"Parser returned invalid result for '{check_name}' on "
            f"{ctx['device_ip']}: {parser_result!r} "
            f"(raw_len={len(raw_output)}, raw_preview={preview!r})"
        ) from ex

    return status, output


def get_check_int_override(
    overrides: Optional[dict[str, int]],
    check_name: str,
    default: int,
    minimum: int = 1,
) -> int:
    if not overrides:
        return default

    value = overrides.get(check_name)
    if value is None:
        check_name_lower = check_name.lower()
        for key, candidate in overrides.items():
            if key.lower() == check_name_lower:
                value = candidate
                break

    if value is None:
        return default

    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        logger.warning(
            f"Ignoring invalid override for check '{check_name}': {value!r}"
        )
        return default


def result_check_key(check_name: str) -> str:
    return check_name if check_name != "Link Status" else "Interface Status"
