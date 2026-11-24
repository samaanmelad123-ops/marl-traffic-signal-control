# 🚦 Cooperative Smart Traffic Management System
### Multi-Agent Reinforcement Learning with PPO + SUMO Simulation

A complete implementation of a **cooperative multi-agent traffic signal control system** using **Proximal Policy Optimization (PPO)** and the **SUMO traffic simulator**. Each traffic light is controlled by an independent PPO agent that coordinates with neighboring intersections to minimize congestion, waiting time, and CO₂ emissions.

---

## 📄 Research Paper

This project is based on the paper:
> **"Cooperative Smart Traffic Management System Using Multi-Agent Reinforcement Learning"**
> — included in this repository as a PDF.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   SUMO Simulator                    │
│         (Real-world road network via OSM)           │
└──────────────────┬──────────────────────────────────┘
                   │ TraCI Interface
       ┌───────────┼───────────┐
       ▼           ▼           ▼
   [Agent 1]   [Agent 2]   [Agent N]     ← One PPO agent per TLS
   Actor-Critic Actor-Critic Actor-Critic
       │           │           │
       └───────────┴───────────┘
           Neighbor Coordination
```

Each agent uses:
- **Actor**: Selects the next traffic phase + adjusts green duration
- **Critic**: Estimates state value for PPO updates
- **PPO Buffer**: Stores experience tuples for batch updates

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **Multi-Agent PPO** | Independent PPO agent per traffic light with shared coordination |
| 🚨 **Emergency Green Wave** | Automatic priority corridor for emergency vehicles |
| 🌦️ **Weather Factor** | Adjustable weather multiplier affecting reward computation |
| 📐 **308-D State Space** | Queue lengths, CO₂, delay, trip time, neighbor phases, occupancy & more |
| 🏆 **Multi-Objective Reward** | Balances waiting time, throughput, stops, CO₂, delay, and emergency response |
| 📊 **Live Logging** | Per-step stats printed every 50 steps |
| 💾 **Model Checkpointing** | Actor/Critic weights saved every 1,000 steps |
| 📈 **Training Visualization** | 4-panel plot: reward, queue, waiting time, trip time |

---

## 📐 State Space (per agent)

| Feature | Description |
|---------|-------------|
| Lane queues | Jam length per detector (normalized) |
| Heavy vehicle ratio | Trucks/buses proportion |
| Load estimate | Mean occupancy across detectors |
| Stop frequency | Halting vehicles / max cars |
| Acceleration spikes | Harsh acceleration events |
| Total waiting time | Cumulative wait (normalized) |
| CO₂ emission | Total CO₂ from all vehicles |
| Mean delay | Actual vs ideal travel time |
| Mean trip time | Average trip duration |
| Emergency presence | 1.0 if emergency vehicle detected |
| Mean occupancy | Lane area occupancy |
| Mean jam length | Average jam in meters |
| Current phase | Normalized phase index |
| Previous duration | Last green phase duration |
| Weather factor | Adjustable weather multiplier |
| Neighbor phases | Normalized phases of adjacent TLS |

---

## 🏆 Reward Function

```python
reward = (
    0.35 * delta_waiting_time      # Reward for reducing wait
  + throughput                     # Reward for vehicle flow
  - 0.25 * stop_frequency         # Penalty for stops
  - 0.20 * co2_emission           # Penalty for emissions
  - 0.10 * acceleration_spikes    # Penalty for harsh driving
  - 0.10 * mean_delay             # Penalty for delays
  - 0.10 * mean_trip_time         # Penalty for long trips
  - 1.00 * emergency_presence     # Heavy penalty if emergency blocked
)
```

---

## 📁 Project Structure

```
project/
├── sumo/
│   ├── Final_PPO_Integration(1).py   # Main PPO training script
│   ├── project (2).sumocfg           # SUMO simulation config
│   ├── project (2).add.xml           # Detectors & additional elements
│   ├── project (1).rou.xml           # Vehicle routes
│   ├── Version 3 George.net.xml      # Road network
│   ├── e2_*.xml                      # Individual E2 lane detectors
│   └── ...                           # Other SUMO config files
├── Cooperative Smart Traffic Management System ... .pdf
├── project_poster.pdf
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9+
- [SUMO Traffic Simulator](https://sumo.dlr.de/docs/Installing/index.html) (≥ 1.18)
- CUDA-capable GPU (optional but recommended)

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Set SUMO Environment Variable

```bash
# Windows
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"

# Linux / macOS
export SUMO_HOME="/usr/share/sumo"
```

### Configure Paths

In `Final_PPO_Integration(1).py`, update:
```python
SUMO_CFG        = r"path\to\project (2).sumocfg"
MODEL_SAVE_PATH = r"path\to\saved_models"
```

### Run Training

```bash
python sumo/Final_PPO_Integration\(1\).py
```

---

## 📊 Output

After training completes, you get:
- **Saved models**: `actor_<tls_id>_<step>.pt` and `critic_<tls_id>_<step>.pt`
- **Training plot**: 4-panel chart saved as `training_results.png`

```
Step     0 | Rew:     0.00 | Queue:   12 | Wait:  34.2s | Delay:  18.1s | TripTime:  45.3s | CO2:    1240
Step    50 | Rew:    12.34 | Queue:    8 | Wait:  21.7s | Delay:  11.3s | TripTime:  38.6s | CO2:     890
...
```

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| `PyTorch` | PPO Actor-Critic neural networks |
| `SUMO + TraCI` | Traffic simulation & real-time control |
| `NumPy` | State computation & GAE calculation |
| `Matplotlib` | Training visualization |

---

## 👤 Author

**Samaan Melad**  
GitHub: [@samaanmelad123-ops](https://github.com/samaanmelad123-ops)
