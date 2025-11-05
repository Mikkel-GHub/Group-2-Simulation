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


