import asyncio
import logging
from typing import Any, Optional

from app.utils.constants import HealthStatus
from app.utils.network_commands_factory.network_factory import NetworkFactory

from .parser_utils import (
    get_check_int_override,
    result_check_key,
    unpack_parser_result,
)
from .ssh_client import run_command, run_multiline_commands

logger = logging.getLogger(__name__)


async def execute_checks(
    conn: Any,
    ctx: dict[str, Any],
    read_timeout: int = 60,
    max_retries: int = 3,
    check_timeout_overrides: Optional[dict[str, int]] = None,
    check_retry_overrides: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """
    Execute all checks for a device and return result dict.
    Mirrors DeviceHelper.execute_show_commands().
    """
    result_dict: dict[str, Any] = {}
    checks = ctx["checks"]["commands"] or {}
    brand_category = ctx["brand_category"]
    host_group = ctx.get("host_group")
    network_command_executer = NetworkFactory.get_network_brand_object(brand_category)
    check_items = list(checks.items())

    def set_failed_check(check_name: str, err: Optional[str]) -> None:
        result_dict[result_check_key(check_name)] = {
            "status": HealthStatus.FAIL.value,
            "output": None,
            "err": err,
        }

    def add_reachable_ping_status() -> None:
        result_dict["Ping Status"] = {
            "status": "REACHABLE",
            "output": None,
            "err": None,
        }

    for each_check, command in check_items:
        cmd = command["command"]
        cmd_pattern = command["pattern"]
        check_overrides = None

        if host_group is not None:
            check_overrides = host_group.checks_override.get(each_check)
            if check_overrides is not None:
                cmd = check_overrides.get_command_str(cmd)

        commands_list = [c.strip() for c in cmd.split(",") if c.strip()]
        last_error = None
        check_read_timeout = get_check_int_override(
            check_timeout_overrides,
            each_check,
            read_timeout,
        )
        check_max_retries = get_check_int_override(
            check_retry_overrides,
            each_check,
            max_retries,
        )

        for attempt in range(check_max_retries):
            try:
                if len(commands_list) == 1:
                    raw_output = await run_command(
                        conn,
                        commands_list[0],
                        timeout=check_read_timeout,
                    )
                else:
                    raw_output = await run_multiline_commands(
                        conn,
                        commands_list,
                        timeout=check_read_timeout,
                    )

                if "override_" in (brand_category or ""):
                    status, output = unpack_parser_result(
                        network_command_executer(
                            ctx["device_type"],
                            each_check,
                            raw_output,
                            cmd_pattern,
                            ctx["dashboard"],
                            brand_category,
                        ),
                        ctx,
                        each_check,
                        raw_output,
                    )
                else:
                    if check_overrides is not None:
                        if check_overrides.parser_func_override is not None:
                            status, output = unpack_parser_result(
                                check_overrides.parser_func_override(),
                                ctx,
                                each_check,
                                raw_output,
                            )
                        elif check_overrides.args_override is not None:
                            status, output = unpack_parser_result(
                                network_command_executer(
                                    each_check,
                                    raw_output,
                                    cmd_pattern,
                                    check_overrides.args_override,
                                ),
                                ctx,
                                each_check,
                                raw_output,
                            )
                        else:
                            status, output = unpack_parser_result(
                                network_command_executer(
                                    each_check,
                                    raw_output,
                                    cmd_pattern,
                                ),
                                ctx,
                                each_check,
                                raw_output,
                            )
                    else:
                        status, output = unpack_parser_result(
                            network_command_executer(
                                each_check,
                                raw_output,
                                cmd_pattern,
                            ),
                            ctx,
                            each_check,
                            raw_output,
                        )

                result_dict[result_check_key(each_check)] = {
                    "status": status,
                    "output": str(output) if output is not None else None,
                    "err": None,
                }
                logger.info(
                    f"[{ctx['device_ip']}] Check '{each_check}' -> {status}"
                )
                break

            except asyncio.TimeoutError:
                last_error = f"Timeout after {check_read_timeout}s"
                logger.warning(
                    f"[{ctx['device_ip']}] Timeout on '{each_check}' "
                    f"attempt {attempt + 1}/{check_max_retries}"
                )
                if hasattr(conn, "interrupt_current_command"):
                    try:
                        await conn.interrupt_current_command()
                    except Exception as recover_ex:
                        last_error = (
                            f"Unable to recover shell after timeout on "
                            f"'{each_check}': {recover_ex}"
                        )
                        logger.warning(f"[{ctx['device_ip']}] {last_error}")
                        set_failed_check(each_check, last_error)
                        add_reachable_ping_status()
                        return result_dict
            except EOFError:
                last_error = "EOFError: connection dropped"
                logger.warning(
                    f"[{ctx['device_ip']}] EOFError on '{each_check}' "
                    f"attempt {attempt + 1}"
                )
                set_failed_check(each_check, last_error)
                add_reachable_ping_status()
                return result_dict
            except Exception as ex:
                last_error = str(ex)
                logger.warning(
                    f"[{ctx['device_ip']}] Error on '{each_check}' "
                    f"attempt {attempt + 1}: {ex}"
                )

        else:
            set_failed_check(each_check, last_error)

    add_reachable_ping_status()
    return result_dict
