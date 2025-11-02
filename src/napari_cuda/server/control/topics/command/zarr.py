"""Zarr dataset command handlers."""

from __future__ import annotations

from typing import Any

from napari_cuda.protocol import CallCommand
from napari_cuda.server.control.command_registry import register_command

from . import CommandRejected, CommandResult

COMMAND_LOAD_ZARR = "napari.load_zarr"


async def command_load_zarr(server: Any, frame: CallCommand, ws: Any) -> CommandResult:
    kwargs = dict(frame.payload.kwargs or {})
    path = kwargs.get("path")
    if not isinstance(path, str) or not path:
        raise CommandRejected(
            code="command.invalid",
            message="'path' must be a non-empty string",
        )
    try:
        await server._handle_zarr_load(path)
    except RuntimeError as exc:
        raise CommandRejected(code="fs.forbidden", message=str(exc)) from exc
    except FileNotFoundError:
        raise CommandRejected(
            code="fs.not_found",
            message="Dataset not found",
            details={"path": path},
        )
    except NotADirectoryError:
        raise CommandRejected(
            code="fs.not_found",
            message="Dataset path is not a directory",
            details={"path": path},
        )
    except ValueError as exc:
        raise CommandRejected(
            code="zarr.invalid",
            message=str(exc),
            details={"path": path},
        ) from exc
    return CommandResult(result={"ok": True})


def register_zarr_commands() -> None:
    """Register zarr-related command handlers."""

    register_command(COMMAND_LOAD_ZARR, command_load_zarr)


__all__ = ["COMMAND_LOAD_ZARR", "command_load_zarr", "register_zarr_commands"]
