"""CPU fallback for memory management."""

import psutil


class CPUMemoryManager:
    """CPU memory manager (fallback when CUDA not available)."""
    
    def __init__(self):
        self.backend = 'cpu'
    
    def get_available_memory(self) -> int:
        """Get available system RAM in bytes."""
        return psutil.virtual_memory().available
    
    def get_used_memory(self) -> int:
        """Get used system RAM in bytes."""
        return psutil.virtual_memory().used
    
    def __repr__(self):
        available_gb = self.get_available_memory() / 1e9
        return f"CPUMemoryManager(available={available_gb:.1f}GB)"