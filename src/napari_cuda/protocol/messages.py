"""
Message definitions for napari-cuda protocol.

Uses dataclasses for type safety and easy serialization.
"""

import json
import enum
from dataclasses import dataclass, asdict
from typing import List, Optional, Any, Union
import numpy as np


class MessageType(enum.Enum):
    """Types of messages in the protocol."""
    # Client -> Server
    SET_CAMERA = "set_camera"
    SET_DIMS = "set_dims"
    REQUEST_FRAME = "request_frame"
    PING = "ping"
    
    # Server -> Client
    CAMERA_UPDATE = "camera_update"
    DIMS_UPDATE = "dims_update"
    FRAME_READY = "frame_ready"
    PONG = "pong"
    ERROR = "error"


@dataclass
class StateMessage:
    """Base class for state synchronization messages."""
    type: str
    timestamp: float = None
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, data: str):
        """Deserialize from JSON string."""
        return cls(**json.loads(data))


@dataclass
class CameraUpdate(StateMessage):
    """Camera state update message."""
    type: str = MessageType.CAMERA_UPDATE.value
    center: List[float] = None
    zoom: float = None
    angles: List[float] = None  # For 3D
    perspective: float = None
    
    def __post_init__(self):
        if self.center:
            self.center = list(self.center)
        if self.angles:
            self.angles = list(self.angles)


@dataclass 
class DimsUpdate(StateMessage):
    """Dimensions state update message."""
    type: str = MessageType.DIMS_UPDATE.value
    current_step: List[int] = None
    ndisplay: int = None
    order: List[int] = None
    
    def __post_init__(self):
        if self.current_step:
            self.current_step = list(self.current_step)
        if self.order:
            self.order = list(self.order)


@dataclass
class Command(StateMessage):
    """Generic command from client to server."""
    type: str
    payload: dict = None


@dataclass
class Response(StateMessage):
    """Generic response from server to client."""
    type: str
    success: bool = True
    payload: dict = None
    error: str = None


@dataclass
class FrameMessage:
    """
    Frame data message for pixel stream.
    
    This is typically sent as binary data, not JSON.
    """
    frame_number: int
    timestamp: float
    width: int
    height: int
    encoding: str  # "h264", "jpeg", "raw"
    data: bytes
    
    def to_bytes(self) -> bytes:
        """Pack into binary format for transmission."""
        # Simple format: 
        # [4 bytes frame_num][8 bytes timestamp][4 bytes w][4 bytes h][1 byte encoding][data]
        import struct
        
        encoding_byte = {
            "h264": 0,
            "jpeg": 1, 
            "raw": 2
        }.get(self.encoding, 255)
        
        header = struct.pack(
            "!Id2IB",  # Network byte order: int, double, 2 ints, byte
            self.frame_number,
            self.timestamp,
            self.width,
            self.height,
            encoding_byte
        )
        
        return header + self.data
    
    @classmethod
    def from_bytes(cls, data: bytes):
        """Unpack from binary format."""
        import struct
        
        header_size = 4 + 8 + 4 + 4 + 1  # 21 bytes
        header = data[:header_size]
        frame_data = data[header_size:]
        
        frame_num, timestamp, width, height, encoding_byte = struct.unpack(
            "!Id2IB", header
        )
        
        encoding = {
            0: "h264",
            1: "jpeg",
            2: "raw"
        }.get(encoding_byte, "unknown")
        
        return cls(
            frame_number=frame_num,
            timestamp=timestamp,
            width=width,
            height=height,
            encoding=encoding,
            data=frame_data
        )


class StreamProtocol:
    """
    Helper class for protocol operations.
    """
    
    @staticmethod
    def parse_message(data: Union[str, bytes]) -> Union[StateMessage, FrameMessage]:
        """Parse incoming message."""
        if isinstance(data, bytes):
            # Binary frame data
            return FrameMessage.from_bytes(data)
        
        # JSON state message
        msg_dict = json.loads(data)
        msg_type = msg_dict.get('type')
        
        if msg_type == MessageType.CAMERA_UPDATE.value:
            return CameraUpdate(**msg_dict)
        elif msg_type == MessageType.DIMS_UPDATE.value:
            return DimsUpdate(**msg_dict)
        elif msg_type in [t.value for t in MessageType]:
            return StateMessage(**msg_dict)
        else:
            return Command(**msg_dict)
    
    @staticmethod
    def create_camera_command(center=None, zoom=None, angles=None) -> str:
        """Create camera update command."""
        cmd = CameraUpdate(
            type=MessageType.SET_CAMERA.value,
            center=center,
            zoom=zoom,
            angles=angles
        )
        return cmd.to_json()
    
    @staticmethod
    def create_dims_command(current_step=None, ndisplay=None) -> str:
        """Create dimensions update command."""
        cmd = DimsUpdate(
            type=MessageType.SET_DIMS.value,
            current_step=current_step,
            ndisplay=ndisplay
        )
        return cmd.to_json()
    
    @staticmethod
    def create_error_response(error_msg: str) -> str:
        """Create error response."""
        resp = Response(
            type=MessageType.ERROR.value,
            success=False,
            error=error_msg
        )
        return resp.to_json()