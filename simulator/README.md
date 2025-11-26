# EXRT AI - ECG Simulator

Real-time ECG signal generation and live monitoring system for continuous heart health assessment.

## Features

### 🎯 ECG Generator (`ecg_generator.py`)
- **Realistic PQRST Complexes**: Generates physiologically accurate heartbeats
- **Heart Rate Variability (HRV)**: Simulates natural variation in heart rate
- **Multiple States**: Rest, Active, Stress, Recovery with different characteristics
- **Customizable Noise**: Add realistic baseline wander and high-frequency noise
- **Scenario Generation**: Pre-defined sequences (workout, stress test, recovery, daily)

### 📡 Live Monitor (`live_monitor.py`)
- **Real-time Streaming**: Continuous ECG data generation and analysis
- **Live Readiness Assessment**: ML-powered predictions updated every 5 seconds
- **Interactive Controls**: Switch between physiological states on-the-fly
- **Trend Visualization**: Real-time plots of ECG, readiness, and heart rate
- **State Transitions**: Auto-transitions between states for realistic scenarios

## Usage

### 1. Test ECG Generator

```python
from simulator.ecg_generator import ECGGenerator

# Create generator
gen = ECGGenerator(fs=700)

# Generate rest state ECG
rest_data = gen.generate_segment(duration_sec=90, state='rest')
print(f"Generated {len(rest_data['signal'])} samples")
print(f"Heart Rate: {rest_data['actual_hr']:.1f} BPM")
print(f"Expected Readiness: {rest_data['expected_readiness']}%")

# Generate workout scenario
scenario = gen.generate_scenario('workout', total_duration=360)
```

### 2. Run Live Monitor

```bash
cd "g:\exait ai"
streamlit run simulator\live_monitor.py
```

Then:
1. Select a physiological state (rest, active, stress, recovery)
2. Click "▶️ Start Monitoring"
3. Watch real-time ECG and readiness updates
4. Switch states to see how readiness changes

## Physiological States

| State | Heart Rate | HRV Level | Expected Readiness |
|-------|-----------|-----------|-------------------|
| **Rest** | 65 BPM | High | 85% |
| **Active** | 95 BPM | Normal | 65% |
| **Stress** | 110 BPM | Low | 35% |
| **Recovery** | 75 BPM | Normal | 55% |

## Quick Scenarios

### Workout (360 seconds)
1. Rest (60s) → Active (120s) → Active (90s) → Recovery (90s)

### Stress Test (360 seconds)
1. Rest (60s) → Stress (120s) → Stress (60s) → Recovery (120s)

### Recovery (360 seconds)
1. Stress (30s) → Recovery (90s) → Recovery (90s) → Rest (150s)

### Daily (360 seconds)
1. Rest (100s) → Active (80s) → Rest (80s) → Active (100s)

## Technical Details

### ECG Generation
- **Sampling Rate**: 700 Hz (default, configurable)
- **PQRST Components**: P-wave, Q-wave, R-wave (main spike), S-wave, T-wave
- **HRV Implementation**: Gaussian noise added to RR intervals
- **Noise Types**: Baseline wander (0.3 Hz) + high-frequency noise

### Live Streaming
- **Chunk Size**: 5 seconds (3500 samples at 700 Hz)
- **Analysis Window**: 90 seconds (63000 samples)
- **Update Frequency**: Every 5 seconds
- **Buffer Size**: 10 seconds rolling window for visualization

### Readiness Calculation
- Uses trained ML model (PCA + GMM)
- Extracts 31 HRV features from 90-second windows
- Returns readiness score (0-100%) and confidence

## File Structure

```
simulator/
├── __init__.py              # Module exports
├── ecg_generator.py         # Core ECG generation logic
├── live_monitor.py          # Streamlit live monitoring app
└── README.md               # This file
```

## Requirements

- Python 3.8+
- numpy
- scipy
- streamlit
- plotly
- Trained readiness model in `analysis/models/`

## Examples

### Generate and Save ECG Data

```python
from simulator.ecg_generator import ECGGenerator
import numpy as np

gen = ECGGenerator(fs=700)

# Generate 5 minutes of stress state ECG
data = gen.generate_segment(duration_sec=300, state='stress', noise_level=0.05)

# Save to file
np.save('stress_ecg.npy', data['signal'])
```

### Custom State Parameters

```python
# Modify state parameters in ecg_generator.py
state_params = {
    'custom': {'hr': 80, 'hrv': 'normal', 'readiness': 70}
}

data = gen.generate_segment(duration_sec=60, state='custom')
```

## Integration with Main App

The simulator integrates seamlessly with the main EXRT AI Streamlit app:

1. **Standalone Mode**: Run `live_monitor.py` for real-time monitoring
2. **Data Generation**: Use `ecg_generator.py` to create test datasets
3. **Scenario Testing**: Validate readiness model with controlled scenarios

## Future Enhancements

- [ ] Multi-lead ECG simulation
- [ ] Arrhythmia patterns
- [ ] Configurable noise profiles
- [ ] Export to standard formats (EDF, WAV)
- [ ] Real device integration (via Bluetooth/serial)

---

**EXRT AI LTD** | Empowering Continuous Heart Health Monitoring
