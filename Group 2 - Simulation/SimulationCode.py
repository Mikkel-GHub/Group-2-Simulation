import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
from scipy.spatial import cKDTree



# ==============================
# 1. Generate User Density Scenarios
# ==============================
np.random.seed(42)

# --- Helper functions ---
def generate_cluster(center, num_points, std_dev):
    x = np.random.normal(center[0], std_dev, num_points)
    y = np.random.normal(center[1], std_dev, num_points)
    return np.vstack((x, y)).T

def generate_poisson_background(area_size, intensity):
    num_points = np.random.poisson(intensity * area_size**2)
    x = np.random.uniform(0, area_size, num_points)
    y = np.random.uniform(0, area_size, num_points)
    return np.vstack((x, y)).T

def assign_demand(num_points):
    return np.random.lognormal(mean=0.5, sigma=0.75, size=num_points)

# --- Scenario definitions ---
scenarios = {}

# Urban scenario
urban_clusters = [generate_cluster((30,30), 800, 3),
                  generate_cluster((70,70), 900, 4),
                  generate_cluster((50,20), 700, 3),
                  generate_cluster((20,70), 600, 4)]
urban_background = generate_poisson_background(100, 0.1)
urban_points = np.vstack(urban_clusters + [urban_background])
scenarios["urban"] = pd.DataFrame({
    "scenario":"urban",
    "x_km": urban_points[:,0],
    "y_km": urban_points[:,1],
    "demand_weight": assign_demand(len(urban_points))
})

# Rural scenario
rural_clusters = [generate_cluster((20,20), 120, 6),
                  generate_cluster((70,30), 150, 7),
                  generate_cluster((50,80), 100, 8)]
rural_background = generate_poisson_background(100, 0.02)
rural_points = np.vstack(rural_clusters + [rural_background])
scenarios["rural"] = pd.DataFrame({
    "scenario":"rural",
    "x_km": rural_points[:,0],
    "y_km": rural_points[:,1],
    "demand_weight": assign_demand(len(rural_points))
})

# Remote scenario
remote_clusters = [generate_cluster((50,50), 40, 4),
                   generate_cluster((80,80), 30, 3)]
remote_background = generate_poisson_background(100, 0.005)
remote_points = np.vstack(remote_clusters + [remote_background])
scenarios["remote"] = pd.DataFrame({
    "scenario":"remote",
    "x_km": remote_points[:,0],
    "y_km": remote_points[:,1],
    "demand_weight": assign_demand(len(remote_points))
})

# Combine and save all scenarios
output_dir = "/mnt/data/user_density_scenarios"
os.makedirs(output_dir, exist_ok=True)
all_data = pd.concat(scenarios.values(), ignore_index=True)
csv_path = os.path.join(output_dir, "user_density_scenarios.csv")
all_data.to_csv(csv_path, index=False)

# Plot scenario distributions
for label, df in scenarios.items():
    plt.figure(figsize=(6,6))
    plt.scatter(df["x_km"], df["y_km"], s=df["demand_weight"]*2, alpha=0.4)
    plt.title(f"{label.capitalize()} Scenario")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


# ==============================
# 2. Characterise Spatial Distributions & Cluster Sizes
# ==============================

def characterise_clusters(df, eps, min_samples):
    coords = df[["x_km","y_km"]].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df = df.copy()
    df["cluster_id"] = clustering.labels_
    
    cluster_stats = []
    for cid in np.unique(clustering.labels_):
        if cid == -1:  # noise
            continue
        members = df[df["cluster_id"]==cid]
        cluster_stats.append({
            "cluster_id": cid,
            "num_users": len(members),
            "mean_demand": members["demand_weight"].mean(),
            "x_center": members["x_km"].mean(),
            "y_center": members["y_km"].mean(),
            "x_std": members["x_km"].std(),
            "y_std": members["y_km"].std()
        })
    return df, pd.DataFrame(cluster_stats)

results = {}
summary_tables = {}

# --- Scenario-specific DBSCAN parameters ---
dbscan_params = {
    "urban": {"eps": 2.0, "min_samples": 20},
    "rural": {"eps": 6.0, "min_samples": 8},
    "remote": {"eps": 8.0, "min_samples": 7},
}

# --- Run clustering with tuned parameters ---
for label, df in scenarios.items():
    params = dbscan_params[label]
    df_labeled, clusters = characterise_clusters(df, 
                                                 eps=params["eps"], 
                                                 min_samples=params["min_samples"])
    results[label] = {"labeled_df": df_labeled, "cluster_stats": clusters}
    
    if not clusters.empty:
        summary_tables[label] = {
            "scenario": label,
            "num_clusters": len(clusters),
            "avg_cluster_size": clusters["num_users"].mean(),
            "median_cluster_size": clusters["num_users"].median(),
            "largest_cluster_size": clusters["num_users"].max(),
            "smallest_cluster_size": clusters["num_users"].min()
        }
    else:
        summary_tables[label] = {"scenario": label, "num_clusters": 0,
                                 "avg_cluster_size": 0, "median_cluster_size": 0,
                                 "largest_cluster_size": 0, "smallest_cluster_size": 0}

summary_df = pd.DataFrame(summary_tables.values())
print("\n=== Cluster Size Summary ===")
print(summary_df)

# Plot clusters per scenario
for label, result in results.items():
    plt.figure(figsize=(6,6))
    plt.scatter(result["labeled_df"]["x_km"], result["labeled_df"]["y_km"],
                c=result["labeled_df"]["cluster_id"], cmap="tab10", s=10)
    plt.title(f"{label.capitalize()} Clusters (DBSCAN tuned)")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
    


# ==============================
# 3. Traffic Demand Simulation
# ==============================

# --- Define service profiles ---
service_profiles = {
    "IoT": {"bandwidth_range": (0.01, 0.1), "prob": {"urban": 0.4, "rural": 0.3, "remote": 0.5}},
    "Broadband": {"bandwidth_range": (1, 50), "prob": {"urban": 0.5, "rural": 0.6, "remote": 0.45}},
    "Critical": {"bandwidth_range": (5, 20), "prob": {"urban": 0.1, "rural": 0.1, "remote": 0.05}}
}

def assign_profiles(df, scenario):
    choices = list(service_profiles.keys())
    probs = [service_profiles[p]["prob"][scenario] for p in choices]
    df = df.copy()
    df["service_profile"] = np.random.choice(choices, size=len(df), p=probs)
    
    # Assign base bandwidth request
    bw = []
    for profile in df["service_profile"]:
        low, high = service_profiles[profile]["bandwidth_range"]
        bw.append(np.random.uniform(low, high))
    df["base_bandwidth_mbps"] = bw
    return df

# Apply profiles to each scenario
for label in scenarios:
    results[label]["labeled_df"] = assign_profiles(results[label]["labeled_df"], label)

# --- Time-of-day demand patterns (normalized multipliers) ---
time_curve = {
    "urban":   [0.6,0.5,0.4,0.5,0.7,0.9,1.0,1.2,1.1,1.0,0.9,0.8,  # 0–11h
                0.8,0.9,1.0,1.2,1.3,1.4,1.2,1.1,1.0,0.8,0.7,0.6], # 12–23h
    "rural":   [0.5,0.4,0.3,0.3,0.4,0.6,0.8,1.0,1.0,0.9,0.8,0.7,
                0.7,0.8,0.9,1.1,1.2,1.3,1.1,1.0,0.9,0.7,0.6,0.5],
    "remote":  [0.6,0.6,0.5,0.5,0.6,0.7,0.8,0.9,0.9,0.8,0.7,0.7,
                0.7,0.8,0.9,1.0,1.0,1.1,1.0,0.9,0.8,0.7,0.7,0.6]
}

# --- Simulate time-varying demand ---
def simulate_demand(df, scenario, hours=24):
    df = df.copy()
    demands = []
    for h in range(hours):
        multiplier = time_curve[scenario][h]
        hour_demand = df["base_bandwidth_mbps"] * multiplier
        demands.append(hour_demand.values)
    demand_matrix = np.array(demands)  # shape (hours, users)
    return demand_matrix

# Example: simulate for urban
urban_demand_matrix = simulate_demand(results["urban"]["labeled_df"], "urban")

# --- Plot average demand over time for each scenario ---
for scenario in scenarios:
    df = results[scenario]["labeled_df"]
    demand_matrix = simulate_demand(df, scenario)
    avg_demand = demand_matrix.mean(axis=1)
    plt.plot(range(24), avg_demand, label=scenario)

plt.title("Average User Demand vs Time of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Bandwidth (Mbps)")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot cumulative (total) traffic demand over time for each scenario ---
plt.figure(figsize=(8,5))
for scenario in scenarios:
    df = results[scenario]["labeled_df"]
    demand_matrix = simulate_demand(df, scenario)
    total_demand = demand_matrix.sum(axis=1)  # total Mbps at each hour
    plt.plot(range(24), total_demand, label=scenario)

plt.title("Cumulative Traffic Demand vs Time of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Total Bandwidth Demand (Mbps)")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Beam schemes implementation + comparison
# ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# --- Parameters (tune these) ---
N_beams = 16                      # number of beams for fixed grid
beam_capacity_mbps = 2000.0       # capacity per beam (Mbps)
handover_limit_per_beam = 50      # max users handovered into/out of a beam per hour
hours = 24

# --- Utilities ---
def jain_fairness(values):
    """Jain's fairness index for a 1D array of non-negative numbers."""
    v = np.array(values, dtype=float)
    if v.sum() == 0:
        return 1.0
    return (v.sum()**2) / (len(v) * (v**2).sum())

def proportional_serve(requests, capacity):
    """Given requests array, serve proportionally up to capacity."""
    total = requests.sum()
    if total <= capacity:
        served = requests.copy()
    else:
        served = requests * (capacity / total)
    return served

# --- Build fixed grid beam centers covering bounding box of data ---
all_users = pd.concat([results[s]["labeled_df"].reset_index(drop=True) for s in results])
x_min, x_max = all_users['x_km'].min(), all_users['x_km'].max()
y_min, y_max = all_users['y_km'].min(), all_users['y_km'].max()

# create grid
grid_n = int(np.ceil(np.sqrt(N_beams)))
xs = np.linspace(x_min, x_max, grid_n)
ys = np.linspace(y_min, y_max, grid_n)
grid_centers = np.array([(x,y) for x in xs for y in ys])[:N_beams]

# helper to assign user to nearest beam
def assign_to_beams(user_coords, beam_centers):
    tree = cKDTree(beam_centers)
    dists, idxs = tree.query(user_coords)
    return idxs, dists

# dynamic beamforming: choose beam centers = largest cluster centers per scenario aggregated
def get_dynamic_centers(results, N_beams):
    # gather cluster centers across all scenarios (or you could do per-scenario)
    centers = []
    for sc, info in results.items():
        cs = info['cluster_stats'] if 'cluster_stats' in info else pd.DataFrame()
        if not cs.empty:
            for _, row in cs.iterrows():
                centers.append((row['x_center'], row['y_center'], row['num_users']))
    if not centers:
        # fallback to fixed grid
        return grid_centers
    dfc = pd.DataFrame(centers, columns=['x','y','n'])
    # choose top-N by num_users; if fewer, pad with grid
    dfc = dfc.sort_values('n', ascending=False)
    chosen = dfc[['x','y']].values[:N_beams]
    if chosen.shape[0] < N_beams:
        needed = N_beams - chosen.shape[0]
        chosen = np.vstack([chosen, grid_centers[:needed]])
    return chosen

# demand-aware handovers: try to move users from overloaded beams to nearby beams with spare capacity
def run_demand_aware_handover(user_coords, base_requests, beam_centers, capacity, adjacency_radius=20.0):
    """
    Input for one hour:
    - user_coords: Nx2
    - base_requests: N (requested Mbps)
    - beam_centers: Bx2
    Returns served array of length N after attempting handovers.
    """
    user_beam_idxs, _ = assign_to_beams(user_coords, beam_centers)
    B = beam_centers.shape[0]
    users_idx_by_beam = [np.where(user_beam_idxs == b)[0] for b in range(B)]
    # initial served within beam by proportional sharing
    served = np.zeros_like(base_requests, dtype=float)
    beam_free_capacity = np.zeros(B, dtype=float)
    
    # initial allocation (per beam)
    for b in range(B):
        idxs = users_idx_by_beam[b]
        if len(idxs) == 0:
            beam_free_capacity[b] = capacity
            continue
        req = base_requests[idxs]
        served_in_beam = proportional_serve(req, capacity)
        served[idxs] = served_in_beam
        beam_free_capacity[b] = capacity - served_in_beam.sum()
    
    # attempt handovers: overloaded beams (free capacity < 0) vs donors with positive free capacity
    # build adjacency using distance between beam centers
    bc_tree = cKDTree(beam_centers)
    neighbors = [bc_tree.query_ball_point(beam_centers[b], adjacency_radius) for b in range(B)]
    
    # For each overloaded beam, try to move some users to neighboring beams with spare capacity
    for b in range(B):
        if beam_free_capacity[b] >= 0:
            continue
        # overloaded: identify users in this beam sorted by descending request (try handing larger ones first)
        idxs = users_idx_by_beam[b]
        if len(idxs) == 0:
            continue
        # unmet in beam
        unmet = base_requests[idxs] - served[idxs]
        if unmet.sum() <= 0:
            continue
        # candidate neighbor beams with spare capacity
        neighs = neighbors[b]
        # sort neighbors by most free capacity
        neighs = sorted(neighs, key=lambda nb: -beam_free_capacity[nb])
        # iterate users trying to move them
        moved_count = 0
        for uid in idxs[np.argsort(-unmet)]:  # largest unmet first
            if moved_count >= handover_limit_per_beam:
                break
            req = base_requests[uid]
            # check neighbors to find one that can accommodate (partial or full)
            for nb in neighs:
                if nb == b:
                    continue
                if beam_free_capacity[nb] <= 0:
                    continue
                # transfer user to nb if spare capacity exists
                transfer_amount = min(req, beam_free_capacity[nb])
                if transfer_amount <= 0:
                    continue
                # update served and beam capacities
                served[uid] = transfer_amount  # note: we now serve only the amount available at new beam
                beam_free_capacity[nb] -= transfer_amount
                moved_count += 1
                break
        # after attempts, continue
    return served

# --- Simulation runner for a scheme ---
def simulate_scheme(scheme_name, beam_centers, results, hours=24, capacity=beam_capacity_mbps):
    metrics = { 'total_requested': np.zeros(hours), 'total_served': np.zeros(hours),
                'blocking_fraction': np.zeros(hours), 'fairness': np.zeros(hours) }
    per_user_served_fraction = {}  # store per-user served proportions (for fairness)
    
    # For each scenario, get its user order and demand matrix
    # We'll concatenate scenario users and track indices so beams cover all users
    users_all = []
    user_scenarios = []
    for sc in results:
        df = results[sc]['labeled_df'].reset_index(drop=True)
        users_all.append(df)
        user_scenarios += [sc]*len(df)
    users_all = pd.concat(users_all, ignore_index=True)
    user_coords = users_all[['x_km','y_km']].values
    # generate demand matrix per user using simulate_demand; expecting result shape (24, N_users)
    demand_by_user = []
    idx = 0
    for sc in results:
        df = results[sc]['labeled_df'].reset_index(drop=True)
        mat = simulate_demand(df, sc)  # hours x n_users_in_sc
        demand_by_user.append(mat)
    demand_matrix = np.hstack(demand_by_user)  # hours x total_users
    
    total_users = demand_matrix.shape[1]
    # per-user cumulative served / requested
    cum_served = np.zeros(total_users)
    cum_requested = demand_matrix.sum(axis=0)
    
    for h in range(hours):
        requests = demand_matrix[h, :].copy()
        total_req = requests.sum()
        # scheme-specific allocation
        if scheme_name == 'fixed-beam':
            # assign to beams by nearest beam center
            user_beam_idxs, _ = assign_to_beams(user_coords, beam_centers)
            served = np.zeros_like(requests)
            B = beam_centers.shape[0]
            for b in range(B):
                idxs = np.where(user_beam_idxs == b)[0]
                if len(idxs) == 0:
                    continue
                served[idxs] = proportional_serve(requests[idxs], capacity)
        elif scheme_name == 'dynamic-beamforming':
            # here beam_centers are dynamic (constructed externally). Same allocation method as fixed but centers chosen differently.
            user_beam_idxs, _ = assign_to_beams(user_coords, beam_centers)
            served = np.zeros_like(requests)
            B = beam_centers.shape[0]
            for b in range(B):
                idxs = np.where(user_beam_idxs == b)[0]
                if len(idxs) == 0:
                    continue
                served[idxs] = proportional_serve(requests[idxs], capacity)
        elif scheme_name == 'demand-aware-handovers':
            # attempt handovers when overloaded
            served = run_demand_aware_handover(user_coords, requests, beam_centers, capacity, adjacency_radius= ( (x_max-x_min + y_max-y_min)/8) )
        else:
            raise ValueError("Unknown scheme")
        
        # metrics
        metrics['total_requested'][h] = total_req
        metrics['total_served'][h] = served.sum()
        unmet = total_req - served.sum()
        metrics['blocking_fraction'][h] = unmet / total_req if total_req>0 else 0.0
        
        # fairness: compute per-user served fraction (served/request) ignoring users with zero request
        ratios = np.zeros_like(requests)
        nonzero = requests > 0
        ratios[nonzero] = served[nonzero] / requests[nonzero]
        metrics['fairness'][h] = jain_fairness(ratios)
        cum_served += served
    
    # aggregate results
    agg = {
        'scheme': scheme_name,
        'total_requested': metrics['total_requested'].sum(),
        'total_served': metrics['total_served'].sum(),
        'avg_blocking_fraction': metrics['blocking_fraction'].mean(),
        'avg_fairness': metrics['fairness'].mean(),
        'per_hour': metrics
    }
    return agg

# ----------------------------
# Run simulations for all three schemes
# ----------------------------
# 1) Fixed-beam: grid_centers
fixed_centers = grid_centers

# 2) Dynamic beamforming: choose top-N cluster centers across all scenarios
dynamic_centers = get_dynamic_centers(results, N_beams)

# 3) Demand-aware handovers: use fixed grid as base, but algorithm allows handovers
handovers_centers = grid_centers.copy()

schemes = {
    'fixed-beam': fixed_centers,
    'dynamic-beamforming': dynamic_centers,
    'demand-aware-handovers': handovers_centers
}

agg_results = []
for name, centers in schemes.items():
    print(f"Simulating scheme: {name}")
    res = simulate_scheme(name, centers, results, hours=hours, capacity=beam_capacity_mbps)
    print(f"  Total requested: {res['total_requested']:.1f} Mbps, Total served: {res['total_served']:.1f} Mbps")
    print(f"  Avg blocking fraction: {res['avg_blocking_fraction']:.3f}, Avg fairness: {res['avg_fairness']:.3f}\n")
    agg_results.append(res)

# Summarize in DataFrame
summary_rows = []
for r in agg_results:
    summary_rows.append({
        'scheme': r['scheme'],
        'total_requested_Mbps': r['total_requested'],
        'total_served_Mbps': r['total_served'],
        'pct_served': 100.0 * r['total_served'] / r['total_requested'] if r['total_requested']>0 else 0,
        'avg_blocking_fraction': r['avg_blocking_fraction'],
        'avg_fairness': r['avg_fairness']
    })
summary_df = pd.DataFrame(summary_rows).set_index('scheme')
print("\n=== Scheme Comparison ===")
print(summary_df)

# Plot comparison: percent served and fairness
plt.figure(figsize=(13,9))
plt.subplot(1,2,1)
summary_df['pct_served'].plot(kind='bar', rot=0)
plt.title('Percent of Total Demand Served')
plt.ylabel('% served')

plt.subplot(1,2,2)
summary_df['avg_fairness'].plot(kind='bar', rot=0)
plt.title('Average Served-Fraction Fairness (Jain)')
plt.ylim(0,1.05)
plt.tight_layout()
plt.show()


# --- RF / link model helper functions ---
def deg2rad(d): return d * np.pi / 180.0

def antenna_gain_from_beamwidth_deg(hpbw_deg):
    """
    Approximate antenna gain (dBi) from half-power beamwidth in degrees.
    Empirical approximation: G(dBi) ≈ 10*log10(32400 / theta^2)
    (useful engineering approximation for parabolic-like patterns).
    """
    theta = np.maximum(hpbw_deg, 0.1)
    g = 10.0 * np.log10(32400.0 / (theta**2))
    return g

def fspl_dB(range_m, frequency_hz):
    """Free-space path loss in dB for range (m) and frequency (Hz)."""
    c = 3e8
    lam = c / frequency_hz
    return 20*np.log10(4*np.pi*range_m / lam)

def noise_power_dBW(bandwidth_hz, noise_temp_k=500.0):
    """Thermal noise power in dBW over bandwidth (Hz). noise_temp_k includes system noise figure."""
    k = 1.380649e-23
    noise_watts = k * noise_temp_k * bandwidth_hz
    return 10*np.log10(noise_watts) if noise_watts > 0 else -1e9

def snr_dB(eirp_dBW, g_rx_dBi, fspl_dB, other_losses_dB, noise_dBW):
    """SNR in dB at receiver."""
    return eirp_dBW + g_rx_dBi - fspl_dB - other_losses_dB - noise_dBW

def spectral_efficiency_from_snr_db(snr_db, cap_bps_hz=6.0):
    """Shannon spectral efficiency (bps/Hz) capped to a practical maximum."""
    snr_lin = 10**(snr_db/10.0)
    se = np.log2(1.0 + snr_lin)
    # cap to a reasonable maximum (implementation/modcod limits)
    return np.minimum(se, cap_bps_hz)

def capacity_from_params(beam_hpbw_deg,
                         sat_tx_power_w=100.0,
                         freq_ghz=20.0,
                         beam_bandwidth_hz=200e6,
                         sat_tx_losses_dB=2.0,
                         user_rx_gain_dBi=30.0,
                         sys_noise_temp_k=500.0,
                         slant_range_km=600.0,
                         other_losses_dB=2.0,
                         se_cap_bps_hz=6.0):
    """
    Computes approximate per-beam capacity (Mbps) using simple link-budget -> Shannon mapping.
    Returns: capacity_mbps, snr_db, se_bps_hz
    """
    freq_hz = freq_ghz * 1e9
    range_m = slant_range_km * 1000.0
    # TX antenna gain from beamwidth
    g_tx_dBi = antenna_gain_from_beamwidth_deg(beam_hpbw_deg)
    # EIRP in dBW
    pt_dBW = 10*np.log10(sat_tx_power_w) if sat_tx_power_w > 0 else -1e9
    eirp_dBW = pt_dBW + g_tx_dBi - sat_tx_losses_dB
    # FSPL
    fspl = fspl_dB(range_m, freq_hz)
    # Noise
    noise_dBW = noise_power_dBW(beam_bandwidth_hz, sys_noise_temp_k)
    # SNR (dB)
    snr = snr_dB(eirp_dBW, user_rx_gain_dBi, fspl, other_losses_dB, noise_dBW)
    se = spectral_efficiency_from_snr_db(snr, cap_bps_hz=se_cap_bps_hz)
    capacity_bps = se * beam_bandwidth_hz
    capacity_mbps = capacity_bps / 1e6
    return capacity_mbps, snr, se

# --- Example baseline parameters (tune these) ---
beam_hpbw_deg_by_scheme = {
    # assume dynamic beams are narrower (steered), fixed beams wider
    "fixed-beam": 6.0,        # degrees HPBW
    "dynamic-beamforming": 3.0,
    "demand-aware-handovers": 6.0
}
sat_tx_power_w = 100.0          # per-beam transmit power (W)
freq_ghz = 20.0                 # Ka-band example
beam_bandwidth_hz = 200e6       # Hz per beam
sys_noise_temp_k = 500.0
user_gain_by_scenario = {"urban": 25.0, "rural": 30.0, "remote": 35.0}  # dBi

slant_range_km = 600.0
other_losses_dB = 2.0
sat_tx_losses_dB = 2.0
se_cap_bps_hz = 6.0

# compute per-scheme beam capacities (single scalar per-beam)
beam_capacity_by_scheme = {}
for scheme, hp in beam_hpbw_deg_by_scheme.items():
    # choose a representative user gain (use average across scenarios)
    avg_user_gain = np.mean(list(user_gain_by_scenario.values()))
    cap_mbps, snr_db, se_bps_hz = capacity_from_params(
        beam_hpbw_deg=hp,
        sat_tx_power_w=sat_tx_power_w,
        freq_ghz=freq_ghz,
        beam_bandwidth_hz=beam_bandwidth_hz,
        sat_tx_losses_dB=sat_tx_losses_dB,
        user_rx_gain_dBi=avg_user_gain,
        sys_noise_temp_k=sys_noise_temp_k,
        slant_range_km=slant_range_km,
        other_losses_dB=other_losses_dB,
        se_cap_bps_hz=se_cap_bps_hz
    )
    beam_capacity_by_scheme[scheme] = {
        "capacity_mbps": cap_mbps,
        "snr_db": snr_db,
        "se_bps_hz": se_bps_hz,
        "hp_bw_deg": hp
    }

print("Per-beam capacity estimates (Mbps):")
for s, v in beam_capacity_by_scheme.items():
    print(f"  {s}: {v['capacity_mbps']:.1f} Mbps (SNR={v['snr_db']:.1f} dB, SE={v['se_bps_hz']:.2f} bps/Hz)")

# --- Integrate beam capacity into simulation runner ---
# We'll update simulate_scheme to accept per-beam capacity scalar (so just pass scalar capacity)
# (If you previously defined simulate_scheme to use a fixed capacity variable, you can call it with capacity=...)

# --- Congestion sweep ---
# Sweep axes:
#  - offered_load_scale: multiply all user base bandwidths by this factor to create congested scenarios
#  - beam_count_values: vary number of beams (reduces coverage granularity and total network capacity)
offered_load_scales = [0.5, 1.0, 1.5, 2.0, 3.0]   # 1.0 is baseline offered load
beam_count_values = [8, 16, 32]

# We'll run schemes: fixed-beam, dynamic-beamforming, demand-aware-handovers
schemes_list = ['fixed-beam', 'dynamic-beamforming', 'demand-aware-handovers']

# reuse get_dynamic_centers, grid_centers, etc. from previous cell if available; else build a simple grid
# get user bounding box:
all_users_df = pd.concat([results[s]["labeled_df"].reset_index(drop=True) for s in results])
x_min, x_max = all_users_df['x_km'].min(), all_users_df['x_km'].max()
y_min, y_max = all_users_df['y_km'].min(), all_users_df['y_km'].max()

def make_grid_centers(N_beams, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max):
    grid_n = int(np.ceil(np.sqrt(N_beams)))
    xs = np.linspace(x_min, x_max, grid_n)
    ys = np.linspace(y_min, y_max, grid_n)
    grid_centers = np.array([(x,y) for x in xs for y in ys])[:N_beams]
    return grid_centers

# Prepare results storage
rows = []
for N_beams in beam_count_values:
    fixed_centers = make_grid_centers(N_beams)
    dynamic_centers = get_dynamic_centers(results, N_beams) if 'get_dynamic_centers' in globals() else fixed_centers.copy()
    handover_centers = fixed_centers.copy()
    centers_dict = {
        'fixed-beam': fixed_centers,
        'dynamic-beamforming': dynamic_centers,
        'demand-aware-handovers': handover_centers
    }
    for scale in offered_load_scales:
        # scale base bandwidths in-place temporarily (we'll multiply demand matrices inside simulate routine instead)
        # We'll create a wrapper to scale demands when calling simulate_scheme
        def simulate_scheme_scaled(name, centers, scale_factor, beam_capacity_mbps):
            # wrapper: temporarily scale base bandwidths in results by multiplying base_bandwidth_mbps
            # create a deep copy of results[sc]['labeled_df'] for simulation so we don't modify original
            results_copy = {}
            for sc in results:
                df = results[sc]['labeled_df'].copy().reset_index(drop=True)
                df['base_bandwidth_mbps'] = df['base_bandwidth_mbps'] * scale_factor
                results_copy[sc] = {'labeled_df': df, 'cluster_stats': results[sc].get('cluster_stats', pd.DataFrame())}
            return simulate_scheme(name, centers, results_copy, hours=24, capacity=beam_capacity_mbps)
        
        for scheme_name in schemes_list:
            # pick beam capacity from beam model using scheme hp bw
            cap_mbps = beam_capacity_by_scheme[scheme_name]['capacity_mbps']
            # NOTE: if you want total system capacity to scale with number of beams, then total capacity = cap_mbps * N_beams
            # simulate
            agg = simulate_scheme_scaled(scheme_name, centers_dict[scheme_name], scale, cap_mbps)
            rows.append({
                'N_beams': N_beams,
                'offered_scale': scale,
                'scheme': scheme_name,
                'total_requested_Mbps': agg['total_requested'],
                'total_served_Mbps': agg['total_served'],
                'pct_served': 100.0 * agg['total_served'] / agg['total_requested'] if agg['total_requested']>0 else 0.0,
                'avg_blocking_fraction': agg['avg_blocking_fraction'],
                'avg_fairness': agg['avg_fairness']
            })

# Build DataFrame
congestion_df = pd.DataFrame(rows)

# Pivot / plotting
print("\n=== Congestion Sweep Summary (sample) ===")
display(congestion_df.head(12))

# Plot percent served vs offered load for each scheme and beam count
plt.figure(figsize=(10,6))
for scheme_name in schemes_list:
    for N_beams in beam_count_values:
        sub = congestion_df[(congestion_df['scheme']==scheme_name) & (congestion_df['N_beams']==N_beams)]
        plt.plot(sub['offered_scale'], sub['pct_served'], marker='o', label=f"{scheme_name}, beams={N_beams}")
plt.xlabel("Offered load scale factor")
plt.ylabel("Percent of demand served (%)")
plt.title("Percent served vs Offered Load — schemes and beam counts")
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
plt.show()

# Plot blocking fraction heatmap (avg blocking) as function of N_beams and offered scale for each scheme
for scheme_name in schemes_list:
    pivot = congestion_df[congestion_df['scheme']==scheme_name].pivot(index='N_beams', columns='offered_scale', values='avg_blocking_fraction')
    plt.figure(figsize=(6,3))
    plt.title(f"Avg Blocking Fraction — {scheme_name}")
    plt.imshow(pivot.values, aspect='auto', origin='lower', cmap='Reds', vmin=0, vmax=1)
    plt.colorbar(label='Blocking fraction')
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    plt.xlabel('Offered load scale')
    plt.ylabel('N_beams')
    plt.show()

# Show final summary table aggregated by scheme (average over sweep)
summary_by_scheme = congestion_df.groupby('scheme').agg({
    'pct_served':'mean',
    'avg_blocking_fraction':'mean',
    'avg_fairness':'mean'
}).reset_index()
print("\n=== Aggregate summary by scheme (averaged over sweep) ===")
print(summary_by_scheme)



