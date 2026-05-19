import re
from typing import Any, Iterable, Optional


def filter_devices(
    devices: list[Any],
    device_names: Optional[Iterable[str]] = None,
    device_ips: Optional[Iterable[str]] = None,
    device_name_regex: Optional[str] = None,
    exclude_names: Optional[Iterable[str]] = None,
    exclude_ips: Optional[Iterable[str]] = None,
) -> list[Any]:
    """
    Filter device objects from DB (must have device_name and device_ip attributes).

    Include logic:
      - If any include filter is provided (names/ips/regex): device must match ANY of them.
      - If no include filters are provided: include all.

    Exclude logic:
      - Excludes are applied last.
    """
    name_set = {str(x).strip() for x in (device_names or []) if x and str(x).strip()}
    ip_set = {str(x).strip() for x in (device_ips or []) if x and str(x).strip()}
    ex_name_set = {str(x).strip() for x in (exclude_names or []) if x and str(x).strip()}
    ex_ip_set = {str(x).strip() for x in (exclude_ips or []) if x and str(x).strip()}

    name_re = re.compile(device_name_regex) if device_name_regex else None

    def matches_include(d) -> bool:
        if not name_set and not ip_set and not name_re:
            return True

        dn = getattr(d, "device_name", "") or ""
        dip = getattr(d, "device_ip", "") or ""

        if name_set and dn in name_set:
            return True
        if ip_set and dip in ip_set:
            return True
        if name_re and name_re.search(dn):
            return True
        return False

    def matches_exclude(d) -> bool:
        dn = getattr(d, "device_name", "") or ""
        dip = getattr(d, "device_ip", "") or ""
        return (dn in ex_name_set) or (dip in ex_ip_set)

    out: list[Any] = []
    for d in devices:
        if matches_include(d) and not matches_exclude(d):
            out.append(d)
    return out
