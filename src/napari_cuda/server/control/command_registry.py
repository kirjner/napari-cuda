"""Simple registry for mapping protocol commands to server callbacks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from napari_cuda.protocol import CallCommand

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.server.control.topics.command import CommandResult

CommandHandler = Callable[[Any, CallCommand, Any], Awaitable["CommandResult"]]


@dataclass(frozen=True)
class CommandRegistration:
    name: str
    handler: CommandHandler
    version: int = 1


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, CommandRegistration] = {}

    def register(self, registration: CommandRegistration) -> None:
        name = registration.name
        if name in self._commands:
            raise ValueError(f"command '{name}' already registered")
        self._commands[name] = registration

    def get_handler(self, name: str) -> CommandHandler | None:
        entry = self._commands.get(name)
        if entry is None:
            return None
        return entry.handler

    def command_names(self) -> tuple[str, ...]:
        return tuple(self._commands.keys())

    def clear(self) -> None:
        self._commands.clear()


COMMAND_REGISTRY = CommandRegistry()


def register_command(name: str, handler: CommandHandler, *, version: int = 1) -> None:
    COMMAND_REGISTRY.register(CommandRegistration(name=name, handler=handler, version=version))


def advertised_commands() -> tuple[str, ...]:
    return COMMAND_REGISTRY.command_names()
