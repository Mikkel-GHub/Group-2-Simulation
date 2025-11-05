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
    
    
