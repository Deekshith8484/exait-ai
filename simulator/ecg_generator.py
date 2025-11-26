"""Real-time ECG signal generator for live simulation.

This module generates synthetic ECG signals with realistic characteristics
including different physiological states (rest, activity, stress).
"""

import numpy as np
from scipy import signal
import time
from typing import Dict, Optional, Tuple


class ECGGenerator:
    """Generate synthetic ECG signals with controllable heart rate and variability."""
    
    def __init__(self, fs: int = 700, base_hr: int = 70):
        """
        Initialize ECG generator.
        
        Args:
            fs: Sampling frequency in Hz (default 700)
            base_hr: Base heart rate in BPM (default 70)
        """
        self.fs = fs
        self.base_hr = base_hr
        self.time_offset = 0.0
        
    def generate_pqrst_complex(self, hr: float, duration: float = 0.8) -> np.ndarray:
        """
        Generate a single PQRST complex (one heartbeat).
        
        Args:
            hr: Heart rate in BPM
            duration: Duration of one cardiac cycle in seconds
            
        Returns:
            Array containing one PQRST complex
        """
        n_samples = int(duration * self.fs)
        t = np.linspace(0, duration, n_samples)
        
        # Initialize signal
        ecg = np.zeros(n_samples)
        
        # P wave (atrial depolarization)
        p_center = 0.08
        p_width = 0.04
        p_amp = 0.15
        ecg += p_amp * np.exp(-((t - p_center) ** 2) / (2 * p_width ** 2))
        
        # Q wave
        q_center = 0.16
        q_width = 0.01
        q_amp = -0.1
        ecg += q_amp * np.exp(-((t - q_center) ** 2) / (2 * q_width ** 2))
        
        # R wave (ventricular depolarization - main spike)
        r_center = 0.18
        r_width = 0.015
        r_amp = 1.5
        ecg += r_amp * np.exp(-((t - r_center) ** 2) / (2 * r_width ** 2))
        
        # S wave
        s_center = 0.22
        s_width = 0.015
        s_amp = -0.3
        ecg += s_amp * np.exp(-((t - s_center) ** 2) / (2 * s_width ** 2))
        
        # T wave (ventricular repolarization)
        t_center = 0.45
        t_width = 0.08
        t_amp = 0.3
        ecg += t_amp * np.exp(-((t - t_center) ** 2) / (2 * t_width ** 2))
        
        return ecg
    
    def add_hrv(self, base_rr: float, hrv_level: str = 'normal') -> float:
        """
        Add heart rate variability to RR interval.
        
        Args:
            base_rr: Base RR interval in seconds
            hrv_level: Level of HRV ('low', 'normal', 'high')
            
        Returns:
            Modified RR interval with HRV
        """
        hrv_std_map = {
            'low': 0.01,      # Low HRV - fatigue/stress
            'normal': 0.03,   # Normal HRV - neutral state
            'high': 0.06,     # High HRV - ready/recovered
        }
        
        std = hrv_std_map.get(hrv_level, 0.03)
        rr_variation = np.random.normal(0, std)
        
        return base_rr + rr_variation
    
    def generate_segment(
        self,
        duration_sec: float,
        state: str = 'rest',
        noise_level: float = 0.05
    ) -> Dict[str, any]:
        """
        Generate a continuous ECG segment.
        
        Args:
            duration_sec: Duration in seconds
            state: Physiological state ('rest', 'active', 'stress', 'recovery')
            noise_level: Amount of noise to add (0.0 to 0.2)
            
        Returns:
            Dictionary with ECG signal, heart rate, and state info
        """
        # State-dependent parameters
        state_params = {
            'rest': {'hr': 65, 'hrv': 'high', 'readiness': 85},
            'active': {'hr': 95, 'hrv': 'normal', 'readiness': 65},
            'stress': {'hr': 110, 'hrv': 'low', 'readiness': 35},
            'recovery': {'hr': 75, 'hrv': 'normal', 'readiness': 55},
        }
        
        params = state_params.get(state, state_params['rest'])
        hr = params['hr']
        hrv_level = params['hrv']
        expected_readiness = params['readiness']
        
        # Calculate base RR interval
        base_rr = 60.0 / hr
        
        # Generate ECG
        ecg_signal = []
        current_time = 0.0
        actual_hr_samples = []
        
        while current_time < duration_sec:
            # Add HRV to RR interval
            rr_interval = self.add_hrv(base_rr, hrv_level)
            rr_interval = max(0.4, min(rr_interval, 1.5))  # Physiological limits
            
            # Generate one heartbeat
            heartbeat = self.generate_pqrst_complex(60.0 / rr_interval, rr_interval)
            ecg_signal.extend(heartbeat)
            
            # Track actual HR
            actual_hr_samples.append(60.0 / rr_interval)
            
            current_time += rr_interval
        
        # Convert to numpy array and truncate to exact duration
        ecg_signal = np.array(ecg_signal[:int(duration_sec * self.fs)])
        
        # Add realistic noise
        if noise_level > 0:
            # Baseline wander (low frequency)
            baseline_wander = 0.1 * noise_level * np.sin(2 * np.pi * 0.3 * np.arange(len(ecg_signal)) / self.fs)
            
            # High frequency noise
            hf_noise = noise_level * np.random.normal(0, 0.02, len(ecg_signal))
            
            ecg_signal += baseline_wander + hf_noise
        
        return {
            'signal': ecg_signal,
            'fs': self.fs,
            'duration': len(ecg_signal) / self.fs,
            'state': state,
            'target_hr': hr,
            'actual_hr': np.mean(actual_hr_samples),
            'hrv_level': hrv_level,
            'expected_readiness': expected_readiness,
            'timestamp': time.time(),
        }
    
    def generate_scenario(
        self,
        scenario_name: str,
        total_duration: int = 300
    ) -> Dict[str, any]:
        """
        Generate a pre-defined scenario with state transitions.
        
        Args:
            scenario_name: Name of scenario ('workout', 'stress_test', 'recovery', 'daily')
            total_duration: Total duration in seconds
            
        Returns:
            Dictionary with complete ECG signal and metadata
        """
        scenarios = {
            'workout': [
                ('rest', 60),
                ('active', 120),
                ('active', 90),
                ('recovery', 90),
            ],
            'stress_test': [
                ('rest', 60),
                ('stress', 120),
                ('stress', 60),
                ('recovery', 120),
            ],
            'recovery': [
                ('stress', 30),
                ('recovery', 90),
                ('recovery', 90),
                ('rest', 150),
            ],
            'daily': [
                ('rest', 100),
                ('active', 80),
                ('rest', 80),
                ('active', 100),
            ],
        }
        
        if scenario_name not in scenarios:
            scenario_name = 'daily'
        
        sequence = scenarios[scenario_name]
        
        # Generate each segment
        all_segments = []
        metadata = []
        
        for state, duration in sequence:
            segment = self.generate_segment(duration, state)
            all_segments.append(segment['signal'])
            metadata.append({
                'state': state,
                'duration': duration,
                'expected_readiness': segment['expected_readiness'],
                'hr': segment['actual_hr'],
            })
        
        # Concatenate all segments
        full_signal = np.concatenate(all_segments)
        
        return {
            'signal': full_signal,
            'fs': self.fs,
            'duration': len(full_signal) / self.fs,
            'scenario': scenario_name,
            'metadata': metadata,
            'timestamp': time.time(),
        }


class LiveECGStream:
    """Stream ECG data in real-time chunks for live monitoring."""
    
    def __init__(self, fs: int = 700, chunk_duration: float = 5.0):
        """
        Initialize live ECG stream.
        
        Args:
            fs: Sampling frequency
            chunk_duration: Duration of each chunk in seconds
        """
        self.generator = ECGGenerator(fs=fs)
        self.fs = fs
        self.chunk_duration = chunk_duration
        self.current_state = 'rest'
        self.state_timer = 0
        self.state_duration = 60
        
    def set_state(self, state: str, duration: int = 60):
        """Set the physiological state for streaming."""
        self.current_state = state
        self.state_duration = duration
        self.state_timer = 0
    
    def get_next_chunk(self) -> Dict[str, any]:
        """
        Get the next chunk of ECG data.
        
        Returns:
            Dictionary with ECG chunk and metadata
        """
        # Check if we need to transition state
        if self.state_timer >= self.state_duration:
            # Auto-transition to recovery or rest
            if self.current_state in ['stress', 'active']:
                self.current_state = 'recovery'
            else:
                self.current_state = 'rest'
            self.state_timer = 0
        
        # Generate chunk
        chunk = self.generator.generate_segment(
            self.chunk_duration,
            self.current_state
        )
        
        self.state_timer += self.chunk_duration
        
        return chunk


if __name__ == "__main__":
    # Test the generator
    print("Testing ECG Generator...")
    
    gen = ECGGenerator(fs=700)
    
    # Generate rest state
    rest_data = gen.generate_segment(duration_sec=10, state='rest')
    print(f"Rest state: HR={rest_data['actual_hr']:.1f}, Expected Readiness={rest_data['expected_readiness']}")
    
    # Generate stress state
    stress_data = gen.generate_segment(duration_sec=10, state='stress')
    print(f"Stress state: HR={stress_data['actual_hr']:.1f}, Expected Readiness={stress_data['expected_readiness']}")
    
    # Generate scenario
    scenario = gen.generate_scenario('workout', total_duration=360)
    print(f"\nWorkout scenario generated: {len(scenario['signal'])} samples, {scenario['duration']:.1f}s")
    print(f"States: {[m['state'] for m in scenario['metadata']]}")


# Convenience function for simple usage
def generate_ecg_segment(duration_sec: float, fs: int = 700, scenario: str = 'normal_resting') -> np.ndarray:
    """
    Generate a synthetic ECG segment quickly.
    
    Args:
        duration_sec: Duration in seconds
        fs: Sampling frequency
        scenario: Physiological scenario name
        
    Returns:
        ECG signal as numpy array
    """
    # Map scenario names to states
    scenario_map = {
        'normal_resting': 'rest',
        'light_activity': 'active',
        'high_stress': 'stress',
        'post-exercise_recovery': 'recovery',
        'sleep/deep_rest': 'rest',
    }
    
    state = scenario_map.get(scenario, 'rest')
    gen = ECGGenerator(fs=fs)
    result = gen.generate_segment(duration_sec, state=state, noise_level=0.03)
    
    return result['signal']
