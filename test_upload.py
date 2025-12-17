import requests
import pickle
import csv
import json
import numpy as np
from pathlib import Path

# Create realistic ECG with proper morphology
fs = 700
duration = 180  # 3 minutes for multiple windows
t = np.linspace(0, duration, int(fs * duration))

# Simulate ECG with QRS complex
hr = 70  # Heart rate in bpm
ecg = 0

# Generate heartbeats
beat_period = 60 / hr
for beat_idx in range(int(duration * hr / 60)):
    beat_time = beat_idx * beat_period
    t_beat = t - beat_time
    
    # QRS complex (main deflection)
    qrs_mask = np.abs(t_beat) < 0.04
    ecg += 1500 * np.exp(-100 * t_beat**2) * qrs_mask
    
    # T wave
    t_mask = (t_beat > 0.04) & (t_beat < 0.25)
    ecg += 400 * np.exp(-20 * (t_beat - 0.15)**2) * t_mask

# Add noise
ecg += 50 * np.random.randn(len(t))
ecg = np.clip(ecg, -500, 2000)

print(f'Signal: {len(ecg)} samples, {duration}s duration')

# Test 1: Upload as .pkl
test_pkl = Path('test_signal.pkl')
with open(test_pkl, 'wb') as f:
    pickle.dump(ecg, f)

with open(test_pkl, 'rb') as f:
    files = {'file': (test_pkl.name, f)}
    r = requests.post('http://127.0.0.1:8000/upload/signal', files=files, timeout=30)
    print(f'\n.pkl upload: {r.status_code}')
    if r.status_code == 200:
        print(f'  Avg Readiness: {r.json()["summary"]["avg_readiness"]:.1f}%')
    else:
        print(f'  Error: {r.json()["detail"]}')

# Test 2: Upload as .csv
test_csv = Path('test_signal.csv')
with open(test_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ecg'])
    for sample in ecg:
        writer.writerow([sample])

with open(test_csv, 'rb') as f:
    files = {'file': (test_csv.name, f)}
    r = requests.post('http://127.0.0.1:8000/upload/signal', files=files, timeout=30)
    print(f'\n.csv upload: {r.status_code}')
    if r.status_code == 200:
        print(f'  Avg Readiness: {r.json()["summary"]["avg_readiness"]:.1f}%')
    else:
        print(f'  Error: {r.json()["detail"]}')

# Test 3: Upload as .json
test_json = Path('test_signal.json')
with open(test_json, 'w') as f:
    json.dump({'ecg': ecg.tolist()}, f)

with open(test_json, 'rb') as f:
    files = {'file': (test_json.name, f)}
    r = requests.post('http://127.0.0.1:8000/upload/signal', files=files, timeout=30)
    print(f'\n.json upload: {r.status_code}')
    if r.status_code == 200:
        print(f'  Avg Readiness: {r.json()["summary"]["avg_readiness"]:.1f}%')
    else:
        print(f'  Error: {r.json()["detail"]}')

print('\n✅ All file formats work!')
