"""Filesystem command handlers."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from napari_cuda.protocol import CallCommand
from napari_cuda.server.control.command_registry import register_command

from . import CommandRejected, CommandResult

COMMAND_LISTDIR = "fs.listdir"


async def command_fs_listdir(server: Any, frame: CallCommand, ws: Any) -> CommandResult:
    kwargs = dict(frame.payload.kwargs or {})
    path = kwargs.get("path")
    show_hidden = bool(kwargs.get("show_hidden", False))
    only = kwargs.get("only")
    filters: Optional[Sequence[str]] = None
    if only is not None:
        if isinstance(only, (list, tuple)):
            filters = tuple(str(item) for item in only if item is not None)
        else:
            raise CommandRejected(
                code="command.invalid",
                message="'only' filter must be a list of suffixes",
            )
    if only is None:
        filters = (".zarr",)
    try:
        listing = server._list_directory(
            path,
            only=filters,
            show_hidden=show_hidden,
        )
    except RuntimeError as exc:
        raise CommandRejected(code="fs.forbidden", message=str(exc)) from exc
    except FileNotFoundError:
        raise CommandRejected(
            code="fs.not_found",
            message="Directory not found",
            details={"path": path},
        )
    except NotADirectoryError:
        raise CommandRejected(
            code="fs.not_found",
            message="Path is not a directory",
            details={"path": path},
        )
    return CommandResult(result=listing)


def register_fs_commands() -> None:
    """Register filesystem command handlers."""

    register_command(COMMAND_LISTDIR, command_fs_listdir)

__all__ = ["COMMAND_LISTDIR", "command_fs_listdir", "register_fs_commands"]
