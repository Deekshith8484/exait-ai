"""ECG Simulator Module for EXRT AI.

This module provides real-time ECG signal generation and streaming
for continuous heart health monitoring demonstrations.
"""

from .ecg_generator import ECGGenerator, LiveECGStream

__all__ = ['ECGGenerator', 'LiveECGStream']
