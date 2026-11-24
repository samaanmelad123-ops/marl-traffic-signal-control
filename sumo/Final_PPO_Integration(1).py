import os
import sys
import numpy as np
import traci
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError("Please set the SUMO_HOME environment variable.")

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# ============================================================
#  HYPERPARAMETERS & CONFIG
# ============================================================

GAMMA           = 0.99
LAMBDA          = 0.95
CLIP_EPS        = 0.2
ENTROPY_COEF    = 0.01
ACTOR_LR        = 3e-4
CRITIC_LR       = 5e-4
MAX_GRAD_NORM   = 0.5
UPDATE_EVERY    = 128
DECISION_INTERVAL = 5
TOTAL_STEPS     = 10_000
MIN_GREEN       = 10
MAX_GREEN       = 60
SAVE_EVERY      = 1_000
MAX_CARS        = 300.0
MAX_WAIT        = 600.0
MAX_SPIKES      = 50.0
MAX_CO2         = 5_000.0
MAX_JAM         = 100.0
MAX_OCC         = 1.0

EMERGENCY_TYPES = {"emergency", "ambulance", "fire"}
HEAVY_TYPES     = {"truck", "bus", "trailer"}


SUMO_CFG        = r"C:\Users\ASUS\Downloads\project final result(1)\project (2).sumocfg"
MODEL_SAVE_PATH = r"C:\project final result(1)\saved_models"

SIM_STEP_LENGTH = 0.1
weather_factor  = 1.0 

# ============================================================
#  SUMO CONFIG
# ============================================================

sumoBinary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe")
sumo_cmd = [
    sumoBinary,
    "-c", SUMO_CFG,
    "--step-length", str(SIM_STEP_LENGTH),
    "--delay", "0",
    "--lateral-resolution", "0",
]

# ============================================================
#  DETECTOR
# ============================================================

def build_detector_map(detector_ids):
    tls_det_map = defaultdict(list)
    for det_id in detector_ids:
        if not det_id.startswith("det_TLS_"):
            continue
        rest = det_id[len("det_"):] 
        parts = rest.split("det_")
        for part in parts:
            tokens = part.split("_") 
            if len(tokens) >= 2 and tokens[0] == "TLS":
                tls_key = f"TLS_{tokens[1]}" 
                tls_det_map[tls_key].append(det_id)
    return dict(tls_det_map)


def match_tls_detectors(tls_ids, tls_det_map):
    tls_to_dets = {}

    for tls_id in tls_ids:
        
        tls_to_dets[tls_id] = tls_det_map.get(tls_id, [])

    return tls_to_dets


# ============================================================
#  NEIGHBOR GRAPH
# ============================================================

def build_neighbor_graph(tls_ids, detector_ids, tls_det_map):

    shared_pairs = set()
    for det_id in detector_ids:
        if not det_id.startswith("det_TLS_"):
            continue
        rest  = det_id[len("det_"):]
        parts = rest.split("det_")
        if len(parts) < 2:
            continue
        tls_keys = []
        for part in parts:
            tokens = part.split("_")
            if len(tokens) >= 2 and tokens[0] == "TLS":
                tls_keys.append(f"TLS_{tokens[1]}")
        if len(tls_keys) == 2:
            shared_pairs.add((tls_keys[0], tls_keys[1]))
            shared_pairs.add((tls_keys[1], tls_keys[0]))

    num_to_real = {}
    for i, tls_id in enumerate(sorted(tls_ids), start=1):
        num_to_real[f"TLS_{i}"] = tls_id

    neighbor_graph = {tls_id: [] for tls_id in tls_ids}
    for (k1, k2) in shared_pairs:
        real1 = num_to_real.get(k1)
        real2 = num_to_real.get(k2)
        if real1 and real2:
            if real2 not in neighbor_graph[real1]:
                neighbor_graph[real1].append(real2)
            if real1 not in neighbor_graph[real2]:
                neighbor_graph[real2].append(real1)

    sorted_ids = sorted(tls_ids)
    for tls_id in tls_ids:
        if not neighbor_graph[tls_id]:
            idx = sorted_ids.index(tls_id)
            if idx > 0:
                neighbor_graph[tls_id].append(sorted_ids[idx - 1])
            if idx < len(sorted_ids) - 1:
                neighbor_graph[tls_id].append(sorted_ids[idx + 1])

    return neighbor_graph


# ============================================================
#  FEATURES
# ============================================================

def get_lane_queues(detectors):
    return [
        traci.lanearea.getJamLengthVehicle(d) / MAX_CARS
        for d in detectors
    ]


def get_load_estimate(detectors):
    occ = sum(traci.lanearea.getLastStepOccupancy(d) for d in detectors)
    return occ / max(len(detectors), 1)


def get_heavy_vehicle_ratio(detectors):
    heavy, total = 0, 0
    for d in detectors:
        for v in traci.lanearea.getLastStepVehicleIDs(d):
            total += 1
            if traci.vehicle.getTypeID(v) in HEAVY_TYPES:
                heavy += 1
    return heavy / max(total, 1)


def get_stop_frequency(detectors):
    stops = sum(traci.lanearea.getLastStepHaltingNumber(d) for d in detectors)
    return stops / MAX_CARS


def get_acceleration_spikes(detectors, threshold=2.5):
    spikes = sum(
        1
        for d in detectors
        for v in traci.lanearea.getLastStepVehicleIDs(d)
        if abs(traci.vehicle.getAcceleration(v)) > threshold
    )
    return min(spikes / MAX_SPIKES, 1.0)


def get_total_waiting_time(detectors):
    wt = sum(
        traci.vehicle.getWaitingTime(v)
        for d in detectors
        for v in traci.lanearea.getLastStepVehicleIDs(d)
    )
    return wt / MAX_WAIT


def get_co2_emission(detector_ids):
    total_co2 = 0.0

    for det in detector_ids:
        vehs = traci.lanearea.getLastStepVehicleIDs(det)
        for v in vehs:
            total_co2 += traci.vehicle.getCO2Emission(v)

    return total_co2

def get_emergency_presence(detectors):
    for d in detectors:
        for v in traci.lanearea.getLastStepVehicleIDs(d):
            if traci.vehicle.getTypeID(v) in EMERGENCY_TYPES:
                return 1.0
    return 0.0


def get_mean_occupancy(detectors):
    if not detectors:
        return 0.0
    occ = [traci.lanearea.getLastStepOccupancy(d) for d in detectors]
    return float(np.mean(occ)) / 100.0


def get_mean_jam_length(detectors):
    if not detectors:
        return 0.0
    jams = [traci.lanearea.getJamLengthMeters(d) for d in detectors]
    return min(float(np.mean(jams)) / MAX_JAM, 1.0)


# ── delay & trip-time tracking ───────────────────────────────

_trip_log: dict[str, float] = {}

def update_trip_log(step: int):
    for v in traci.simulation.getDepartedIDList():
        _trip_log[v] = step * SIM_STEP_LENGTH
    for v in traci.simulation.getArrivedIDList():
        _trip_log.pop(v, None)


def get_mean_delay(detectors):
    delays = []
    for d in detectors:
        for v in traci.lanearea.getLastStepVehicleIDs(d):
            try:
                tt_actual = traci.vehicle.getAccumulatedWaitingTime(v)
                tt_ideal = traci.vehicle.getDistance(v) / max(
                traci.vehicle.getMaxSpeed(v), 0.1
                                                )
                delays.append(max(0.0, tt_actual - tt_ideal))
            except traci.exceptions.TraCIException:
                pass
    return (np.mean(delays) / MAX_WAIT) if delays else 0.0


def get_mean_trip_time(current_step: int):
    now = current_step * SIM_STEP_LENGTH
    if not _trip_log:
        return 0.0
    times = [now - t for t in _trip_log.values()]
    return min(np.mean(times) / MAX_WAIT, 1.0)


# ============================================================
#  STATE BUILDER
# ============================================================

def get_state(tls_id: str, current_step: int, prev_phase_duration: float,
              neighbors: list, detectors: list) -> np.ndarray:

    det_queues         = get_lane_queues(detectors)
    heavy_ratio        = get_heavy_vehicle_ratio(detectors)
    load               = get_load_estimate(detectors)
    stop_freq          = get_stop_frequency(detectors)
    accel_spikes       = get_acceleration_spikes(detectors)
    waiting_time       = get_total_waiting_time(detectors)
    co2                = get_co2_emission(detectors)
    delay              = get_mean_delay(detectors)
    trip_time          = get_mean_trip_time(current_step)
    emergency          = get_emergency_presence(detectors)
    occupancy          = get_mean_occupancy(detectors)
    jam_length         = get_mean_jam_length(detectors)
    current_phase      = traci.trafficlight.getPhase(tls_id)
    num_phases         = len(
        traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
    )
    current_phase_norm = current_phase / max(num_phases - 1, 1)
    prev_dur_norm      = prev_phase_duration / MAX_GREEN
    neighbor_phases    = [
        traci.trafficlight.getPhase(n) / max(
            len(traci.trafficlight.getAllProgramLogics(n)[0].phases) - 1, 1
        )
        for n in neighbors
    ]

    state = np.array(
        det_queues
        + [heavy_ratio, load, stop_freq, accel_spikes,
           waiting_time, co2, delay, trip_time, emergency,
           occupancy, jam_length,
           current_phase_norm, prev_dur_norm, weather_factor]
        + neighbor_phases,
        dtype=np.float32,
    )
    return state


# ============================================================
#  REWARD FUNCTION
# ============================================================

def compute_reward(detectors: list, current_step: int, prev_waiting_time: float):
 
    current_wt  = get_total_waiting_time(detectors)
    delta_wt    = prev_waiting_time - current_wt 
    throughput  = sum(
        traci.lanearea.getLastStepVehicleNumber(d) for d in detectors
    ) / MAX_CARS * 0.1

    stops       = get_stop_frequency(detectors)
    co2         = get_co2_emission(detectors)
    spikes      = get_acceleration_spikes(detectors)
    delay       = get_mean_delay(detectors)
    trip_time   = get_mean_trip_time(current_step)
    emergency   = get_emergency_presence(detectors)

    reward = (
          0.35 * delta_wt
        + throughput
        - 0.25 * stops
        - 0.20 * co2
        - 0.10 * spikes
        - 0.10 * delay
        - 0.10 * trip_time
        - 1.00 * emergency
    )

    reward = float(np.clip(reward, -10.0, 10.0))
    return reward, current_wt


# ============================================================
#  EMERGENCY GREEN WAVE
# ============================================================

def apply_emergency_green_wave(detectors: list, tls_id: str, neighbors: list,emergency_phase: int = 0):

    if get_emergency_presence(detectors) == 1.0:
        traci.trafficlight.setPhase(tls_id, emergency_phase)

        for neighbor in neighbors:
            try:
                n_phases = traci.trafficlight.getAllProgramLogics(neighbor)[0].phases
                red_phase = len(n_phases) - 1 
                traci.trafficlight.setPhase(neighbor, red_phase)
            except Exception:
                pass
        return True
    return False


# ============================================================
#  PPO BUFFER
# ============================================================

class PPOBuffer:
    def __init__(self):
        self.clear()

    def store(self, state, action, reward, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states     = []
        self.actions    = []
        self.rewards    = []
        self.log_probs  = []
        self.values     = []

    def size(self):
        return len(self.rewards)


# ============================================================
#  ACTOR
# ============================================================

class Actor(nn.Module):
    def __init__(self, state_dim: int, num_phases: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.phase_head    = nn.Linear(128, num_phases)
        self.duration_head = nn.Linear(128, 1) 
    def forward(self, x):
        h = self.net(x)
        phase_logits  = self.phase_head(h)
        duration_delta = torch.tanh(self.duration_head(h)) 
        return phase_logits, duration_delta


# ============================================================
#  CRITIC
# ============================================================

class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
#  PPO UPDATE
# ============================================================

def compute_gae(rewards, values, gamma=GAMMA, lam=LAMBDA):
    advantages, gae = [], 0.0
    values_ext = values + [0.0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
        gae   = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages


def ppo_update(buffer: PPOBuffer, actor: Actor, critic: Critic,
               actor_opt, critic_opt, num_phases: int):
    if buffer.size() < 2:
        return

    states   = torch.tensor(np.array(buffer.states),    dtype=torch.float32).to(DEVICE)
    actions  = torch.tensor(np.array(buffer.actions),   dtype=torch.float32).to(DEVICE)
    old_lp   = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32).to(DEVICE)
    rewards  = buffer.rewards
    values   = buffer.values

    advantages = compute_gae(rewards, values)
    returns    = [a + v for a, v in zip(advantages, values)]

    advantages_t = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
    returns_t    = torch.tensor(returns,    dtype=torch.float32).to(DEVICE)

    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    # ── Actor loss ──────────────────────────────────────────
    actor_opt.zero_grad()
    phase_logits, duration_delta = actor(states)

    log_probs_new = torch.log_softmax(phase_logits, dim=-1)
    phase_idx     = actions[:, 0].long()
    log_probs_act = log_probs_new[torch.arange(len(phase_idx)), phase_idx]

    ratios  = torch.exp(log_probs_act - old_lp)
    clipped = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS)
    phase_loss = -torch.mean(
        torch.min(ratios * advantages_t, clipped * advantages_t)
    )

    duration_loss = torch.mean((actions[:, 1] - duration_delta[:, 0]) ** 2)

    probs   = torch.softmax(phase_logits, dim=-1)
    entropy = -torch.mean(torch.sum(probs * log_probs_new, dim=-1))

    actor_loss = phase_loss + 0.1 * duration_loss - ENTROPY_COEF * entropy
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
    actor_opt.step()

    # ── Critic loss ─────────────────────────────────────────
    critic_opt.zero_grad()
    values_pred = critic(states)
    critic_loss = torch.mean((returns_t - values_pred) ** 2)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
    critic_opt.step()

    buffer.clear()


# ============================================================
#  SELECT ACTION
# ============================================================

def select_action(state: np.ndarray, actor: Actor, critic: Critic,
                  temperature: float = 1.2):
    s_t       = torch.tensor(state[None], dtype=torch.float32).to(DEVICE)
    phase_logits, duration_delta = actor(s_t)
    value     = critic(s_t).item()

    dist         = torch.distributions.Categorical(
        logits=phase_logits / temperature
    )
    phase_action = dist.sample().item()
    log_prob     = dist.log_prob(
        torch.tensor(phase_action, device=DEVICE)
    ).item()

    delta_s = float(duration_delta[0, 0].item()) * 5.0
    return phase_action, delta_s, log_prob, value


# ============================================================
#  PRINT STATISTICS HELPER
# ============================================================

def print_stats(step, cumulative_reward, detectors, current_step):
    total_queue = sum(traci.lanearea.getJamLengthVehicle(d) for d in detectors)
    total_wt    = get_total_waiting_time(detectors) * MAX_WAIT
    total_co2   = sum(                                                  # ── FIXED ──
        traci.vehicle.getCO2Emission(v)
        for d in detectors
        for v in traci.lanearea.getLastStepVehicleIDs(d)
    )
    mean_delay  = get_mean_delay(detectors) * MAX_WAIT
    mean_trip   = get_mean_trip_time(current_step) * MAX_WAIT

    print(
        f"Step {step:5d} | "
        f"Rew: {cumulative_reward:8.2f} | "
        f"Queue: {total_queue:4.0f} | "
        f"Wait: {total_wt:6.1f}s | "
        f"Delay: {mean_delay:6.1f}s | "
        f"TripTime: {mean_trip:6.1f}s | "
        f"CO2: {total_co2:7.0f}"
    )


# ============================================================
#  MAIN TRAINING LOOP
# ============================================================

def train():
    global weather_factor

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    traci.start(sumo_cmd)

    TLS_IDS      = list(traci.trafficlight.getIDList())
    detector_ids = list(traci.lanearea.getIDList())
    print(f"[INFO] Found {len(TLS_IDS)} traffic lights: {TLS_IDS}")
    print(f"[INFO] Found {len(detector_ids)} detectors")

    tls_det_map    = build_detector_map(detector_ids)
    neighbor_graph = build_neighbor_graph(TLS_IDS, detector_ids, tls_det_map)

    print("[INFO] Neighbor graph:")
    for tls_id, nbrs in neighbor_graph.items():
        print(f"  {tls_id[:35]} → {[n[:25] for n in nbrs]}")

    agents    = {}
    sorted_ids = sorted(TLS_IDS)
    for i, TLS_ID in enumerate(sorted_ids, start=1):
        tls_key  = f"TLS_{i}"
        DETS     = tls_det_map.get(tls_key, [])
        if not DETS:
            DETS = detector_ids
            print(f"[WARNING] No detectors matched for {TLS_ID} ({tls_key}), using all detectors as fallback")
        else:
            print(f"[INFO] {TLS_ID} ({tls_key}) → {len(DETS)} detectors")

        NEIGHBORS  = neighbor_graph.get(TLS_ID, [])
        NUM_PHASES = len(traci.trafficlight.getAllProgramLogics(TLS_ID)[0].phases)

        prev_phase_duration = MIN_GREEN
        dummy_state = get_state(TLS_ID, 0, prev_phase_duration, NEIGHBORS, DETS)
        STATE_DIM   = len(dummy_state)
        print(f"  state_dim={STATE_DIM} | phases={NUM_PHASES} | neighbors={len(NEIGHBORS)}")

        actor      = Actor(STATE_DIM, NUM_PHASES).to(DEVICE)
        critic     = Critic(STATE_DIM).to(DEVICE)
        agents[TLS_ID] = {
            "detectors":   DETS,
            "neighbors":   NEIGHBORS,
            "num_phases":  NUM_PHASES,
            "actor":       actor,
            "critic":      critic,
            "actor_opt":   optim.Adam(actor.parameters(),  lr=ACTOR_LR),
            "critic_opt":  optim.Adam(critic.parameters(), lr=CRITIC_LR),
            "buffer":      PPOBuffer(),
            "prev_dur":    float(MIN_GREEN),
            "prev_wt":     0.0,
            "last_switch": -MIN_GREEN,
            "pending":     None,
        }

    first_tls     = sorted_ids[0]
    log_steps, log_reward = [], []
    log_queue, log_wt     = [], []
    log_delay, log_trip   = [], []

    cumulative_reward = 0.0

    print("[INFO] Training started …\n")

    for step in range(TOTAL_STEPS):
        traci.simulationStep()
        update_trip_log(step)

        if step % DECISION_INTERVAL == 0:
            for TLS_ID, ag in agents.items():
                DETS      = ag["detectors"]
                NEIGHBORS = ag["neighbors"]

                if ag["pending"] is not None:
                    reward, new_wt = compute_reward(DETS, step, ag["prev_wt"])
                    ag["prev_wt"]  = new_wt
                    ag["buffer"].store(
                        ag["pending"]["state"],
                        ag["pending"]["action"],
                        reward,
                        ag["pending"]["log_prob"],
                        ag["pending"]["value"],
                    )
                    cumulative_reward += reward

                # ── Emergency Green Wave ───────
                emergency_active = apply_emergency_green_wave(
                    DETS, TLS_ID, NEIGHBORS
                )

                if not emergency_active:
                    # ── Normal action selection ──────────────────
                    state = get_state(TLS_ID, step, ag["prev_dur"], NEIGHBORS, DETS)
                    phase_action, delta_s, log_prob, value = select_action(
                        state, ag["actor"], ag["critic"]
                    )
                    ag["prev_dur"] = float(
                        np.clip(ag["prev_dur"] + delta_s, MIN_GREEN, MAX_GREEN)
                    )

                    current_phase = traci.trafficlight.getPhase(TLS_ID)
                    if (phase_action != current_phase
                            and (step - ag["last_switch"]) >= MIN_GREEN):
                        traci.trafficlight.setPhase(TLS_ID, phase_action)
                        ag["last_switch"] = step

                    ag["pending"] = {
                        "state":    state,
                        "action":   [phase_action, delta_s],
                        "log_prob": log_prob,
                        "value":    value,
                    }
                else:
                    ag["pending"] = None

        # ── PPO Update ───────────────────────────────────────
        if (step + 1) % UPDATE_EVERY == 0:
            for TLS_ID, ag in agents.items():
                if ag["buffer"].size() > 0:
                    ppo_update(
                        ag["buffer"], ag["actor"], ag["critic"],
                        ag["actor_opt"], ag["critic_opt"], ag["num_phases"]
                    )

        # ── Periodic save ────────────────────────────────────
        if (step + 1) % SAVE_EVERY == 0:
            for TLS_ID, ag in agents.items():
                safe_name = TLS_ID.replace("/", "_").replace(" ", "_")[:40]
                torch.save(ag["actor"].state_dict(),
                           os.path.join(MODEL_SAVE_PATH, f"actor_{safe_name}_{step+1}.pt"))
                torch.save(ag["critic"].state_dict(),
                           os.path.join(MODEL_SAVE_PATH, f"critic_{safe_name}_{step+1}.pt"))
            print(f"  [SAVE] Models saved at step {step+1}")

        # ── Logging every 50 steps ───────────────────────────
        if step % 50 == 0:
            ref_dets = agents[first_tls]["detectors"]
            print_stats(step, cumulative_reward, ref_dets, step)
            log_steps.append(step)
            log_reward.append(cumulative_reward)
            log_queue.append(
                sum(traci.lanearea.getJamLengthVehicle(d) for d in ref_dets)
            )
            log_wt.append(get_total_waiting_time(ref_dets) * MAX_WAIT)
            log_delay.append(get_mean_delay(ref_dets) * MAX_WAIT)
            log_trip.append(get_mean_trip_time(step) * MAX_WAIT)

    traci.close()

    # ── Final save ───────────────────────────────────────────
    for TLS_ID, ag in agents.items():
        safe_name = TLS_ID.replace("/", "_").replace(" ", "_")[:40]
        torch.save(ag["actor"].state_dict(),
                   os.path.join(MODEL_SAVE_PATH, f"actor_{safe_name}_final.pt"))
        torch.save(ag["critic"].state_dict(),
                   os.path.join(MODEL_SAVE_PATH, f"critic_{safe_name}_final.pt"))
    print("\n[INFO] Final models saved.")

    # ──statistics ─────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    if log_delay:
        print(f"  Mean Delay (all logged steps) : {np.mean(log_delay):>8.2f} s")
        print(f"  Mean Trip Time                : {np.mean(log_trip):>8.2f} s")
        print(f"  Mean Waiting Time             : {np.mean(log_wt):>8.2f} s")
        print(f"  Mean Queue Length             : {np.mean(log_queue):>8.1f} veh")
        print(f"  Final Cumulative Reward       : {cumulative_reward:>8.2f}")
    print("=" * 60)

    # ── Visualization (4 panels) ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO Traffic Control — Training Results", fontsize=14)

    axes[0, 0].plot(log_steps, log_reward, color="steelblue")
    axes[0, 0].set_title("Cumulative Reward")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(log_steps, log_queue, color="darkorange")
    axes[0, 1].set_title("Queue Length")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Vehicles")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(log_steps, log_wt,    color="firebrick",   label="Waiting Time")
    axes[1, 0].plot(log_steps, log_delay, color="purple",      label="Delay")
    axes[1, 0].set_title("Waiting Time & Delay")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(log_steps, log_trip, color="seagreen")
    axes[1, 1].set_title("Mean Trip Time")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Seconds")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = os.path.join(MODEL_SAVE_PATH, "training_results.png")
    plt.savefig(out_fig, dpi=150)
    plt.show()
    print(f"[INFO] Plot saved to {out_fig}")


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()