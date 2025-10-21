import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# =============================================================================#
# Quality of Service (QoS)                                                     #
# =============================================================================#

# =============================================================================
# COMPUTATION
# =============================================================================

# =====================
# Blocking Probability
# =====================

def erlang_b(traffic_erlangs: float, servers: int) -> float:
    """
    Calculate blocking probability using Erlang-B formula
    What's the chance all satellite lines are busy?
    """
    
import math

def erlang_b(traffic_erlangs: float, servers: int) -> float:
    """
    Calculate blocking probability using Erlang-B formula
    Uses logarithmic form for numerical stability with large numbers
    """
    # =========================================================================
    # STEP 1: Handle edge cases
    # =========================================================================
    if servers == 0 or traffic_erlangs < 0:
        return 1.0  # No servers or negative traffic = 100% blocking
    
    # =========================================================================
    # STEP 2: Use logarithms to avoid massive numbers
    # =========================================================================
    try:
        # ---------------------------------------------------------------------
        # STEP 2A: Calculate numerator in log space
        # Original: numerator = (traffic_erlangs^servers) / servers!
        # Log form: log(numerator) = servers*log(traffic) - log(servers!)
        # ---------------------------------------------------------------------
        log_numerator = servers * math.log(traffic_erlangs) - math.lgamma(servers + 1)
        # math.lgamma(n+1) = log(n!) but more numerically stable than math.log(math.factorial(n))
        
        # ---------------------------------------------------------------------
        # STEP 2B: Calculate denominator in log space  
        # Original: denominator = Σ(traffic^i / i!) for i=0 to servers
        # Log form: We need to sum terms: log(traffic^i / i!) = i*log(traffic) - log(i!)
        # ---------------------------------------------------------------------
        # Start with the first term (i=0): traffic^0 / 0! = 1/1 = 1 → log(1) = 0
        log_denominator = 0.0
        
        # Sum all terms in the series using log-sum-exp trick
        for i in range(1, servers + 1):  # Start from i=1 since we already have i=0
            # Calculate log of each term: log(traffic^i / i!)
            log_term = i * math.log(traffic_erlangs) - math.lgamma(i + 1)
            
            # -----------------------------------------------------------------
            # STEP 2C: Log-sum-exp trick to add terms in log space
            # We want: log(exp(log_denominator) + exp(log_term))
            # Formula: log(a + b) = log(a) + log(1 + exp(log(b) - log(a)))
            # Where: a = current sum, b = new term
            # -----------------------------------------------------------------
            if log_term > log_denominator:
                # If new term is larger, swap roles to maintain numerical stability
                log_denominator = log_term + math.log1p(math.exp(log_denominator - log_term))
            else:
                # Standard case: add new term to existing sum
                log_denominator = log_denominator + math.log1p(math.exp(log_term - log_denominator))
        
        # ---------------------------------------------------------------------
        # STEP 3: Convert back from log space to probability
        # Original: blocking_prob = numerator / denominator
        # Log form: log(blocking_prob) = log(numerator) - log(denominator)
        # Then: blocking_prob = exp(log(numerator) - log(denominator))
        # ---------------------------------------------------------------------
        log_blocking_prob = log_numerator - log_denominator
        blocking_prob = math.exp(log_blocking_prob)
        
        # Ensure result is valid probability
        return max(0.0, min(1.0, blocking_prob))
        
    except (OverflowError, ValueError):
        # =====================================================================
        # STEP 4: Fallback for extreme cases where logs still fail
        # =====================================================================
        if traffic_erlangs >= servers:
            return 1.0  # More traffic than servers = system overloaded
        else:
            return 0.0  # Less traffic than servers = no blocking
    """ 
    if servers > 1000:  # Cap server count
        servers = 1000
        
    if servers == 0 or traffic_erlangs < 0:
        return 1.0  # No servers or negative traffic = 100% blocking
    
    numerator = (traffic_erlangs ** servers) / math.factorial(servers)
    denominator = sum((traffic_erlangs ** i) / math.factorial(i) for i in range(servers + 1))
    
    return numerator / denominator if denominator > 0 else 1.0
    """

# ==================
# Throughput per user
# ==================

def calculate_throughput(total_capacity: float, active_users: int, 
                       blocking_prob: float) -> float:
    """Calculate average throughput per user WITH safety checks"""
    # Handle edge cases
    if total_capacity <= 0 or active_users <= 0:
        return 0.0
    
    # Clamp blocking probability to valid range
    blocking_prob = max(0, min(1, blocking_prob))
    
    # Available capacity after accounting for blocked connections
    effective_capacity = total_capacity * (1 - blocking_prob)
    return effective_capacity / active_users

# ===============
# Fairness Index
# ===============

def jains_fairness_index(throughputs: List[float]) -> float:
    """Calculate Jain's fairness index for throughput distribution"""
    if not throughputs or len(throughputs) == 0:
        return 0.0
    
    numerator = sum(throughputs) ** 2
    denominator = len(throughputs) * sum(t ** 2 for t in throughputs)
    
    return numerator / denominator if denominator > 0 else 0.0

class LEOSatelliteSimulator:
    def __init__(self):
        self.results = []
    
    def simulate_scenario(self, cluster_density: float, constellation_size: int, 
                         revisit_interval: float, total_capacity: float = 1000) -> Dict:
        """
        Simulate one satellite network scenario
        
        Args:
            cluster_density: Users per square km (more users = more busy signals)
            constellation_size: Number of satellites (more satellites = more lines available)  
            revisit_interval: Minutes between satellite passes (shorter = faster service)
            total_capacity: Total system capacity in Mbps
        """
        
        # Input validation
        if cluster_density < 0 or constellation_size < 0 or revisit_interval <= 0:
            return self._get_error_result("Invalid input parameters")
        
        # Model parameters (customize these based on your scenario)
        BASE_ARRIVAL_RATE = 0.1   # Requests per user per minute
        BEAMS_PER_SATELLITE = 10   # Channels each satellite can serve
        COVERAGE_FACTOR = 0.3      # Fraction of satellites covering area at once
        
        # Calculate traffic parameters
        arrival_rate = cluster_density * BASE_ARRIVAL_RATE
        service_rate = 1.0 / revisit_interval
        effective_servers = int(constellation_size * COVERAGE_FACTOR * BEAMS_PER_SATELLITE)
        
        # Offered traffic in Erlangs
        traffic_intensity = arrival_rate / service_rate
        
        # Calculate core metrics
        blocking_prob = erlang_b(traffic_intensity, effective_servers)
        
        # Estimate active users with safety check
        active_users = max(0, cluster_density * (1 - blocking_prob))
        
        throughput_per_user = calculate_throughput(
            total_capacity, active_users, blocking_prob
        )
        
        # Simulate user throughputs for fairness calculation
        user_throughputs = []
        if active_users > 0:
            user_throughputs = np.random.normal(
                throughput_per_user, 
                throughput_per_user * 0.3,  # 30% variation
                size=max(1, int(active_users))  # Ensure at least 1 user
            )
            user_throughputs = np.maximum(user_throughputs, 0)  # No negative throughput
        
        fairness_index = jains_fairness_index(user_throughputs.tolist())
        
        return {
            'cluster_density': cluster_density,
            'constellation_size': constellation_size,
            'revisit_interval': revisit_interval,
            'blocking_probability': blocking_prob,
            'throughput_per_user': throughput_per_user,
            'fairness_index': fairness_index,
            'traffic_intensity': traffic_intensity,
            'effective_servers': effective_servers,
            'active_users': active_users
        }
    
    def _get_error_result(self, error_msg: str) -> Dict:
        """Return a result dictionary indicating error"""
        return {
            'cluster_density': 0,
            'constellation_size': 0,
            'revisit_interval': 0,
            'blocking_probability': 1.0,  # 100% blocking = system failure
            'throughput_per_user': 0.0,
            'fairness_index': 0.0,
            'traffic_intensity': 0,
            'effective_servers': 0,
            'active_users': 0,
            'error': error_msg
        }

# =============================================================================
# ANALYSIS PERFORMANCE VARIABILITY
# =============================================================================

def run_comprehensive_analysis():
    """Run analysis across all parameter combinations"""
    simulator = LEOSatelliteSimulator()
    
    # ================
    # Cluster Density
    # ================
    cluster_densities = [1, 10, 50, 100, 200]    # users/km²
    
    # ==================
    # Constellation Size  
    # ==================
    constellation_sizes = [10, 50, 100, 200] # satellites
    
    # =================
    # Revisit Interval
    # =================
    revisit_intervals = [1, 5, 10, 30]           # minutes
    
    results = []
    
    print("Running satellite performance analysis...")
    # Analyze all parameter combinations
    for density in cluster_densities:
        for satellites in constellation_sizes:
            for interval in revisit_intervals:
                scenario_result = simulator.simulate_scenario(
                    density, satellites, interval
                )
                results.append(scenario_result)
                print(f"Density: {density}, Sats: {satellites}, Interval: {interval}min -> "
                      f"Blocking: {scenario_result['blocking_probability']:.3f}, "
                      f"Throughput: {scenario_result['throughput_per_user']:.1f} Mbps, "
                      f"Fairness: {scenario_result['fairness_index']:.3f}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    return df, simulator

def analyze_performance_trends(results_df):
    """Analyze how performance varies with each parameter"""
    
    print("\n" + "="*50)
    print("PERFORMANCE VARIABILITY ANALYSIS")
    print("="*50)
    
    # ================
    # Cluster Density
    # ================
    print("\n1. CLUSTER DENSITY IMPACT:")
    high_density = results_df[results_df['cluster_density'] == 200].iloc[0]
    low_density = results_df[results_df['cluster_density'] == 10].iloc[0]
    
    print(f"   High density (200 users/km²):")
    print(f"   - Blocking: {high_density['blocking_probability']:.3f} (HIGH)")
    print(f"   - Throughput: {high_density['throughput_per_user']:.1f} Mbps (LOW)")
    print(f"   - Fairness: {high_density['fairness_index']:.3f}")
    
    print(f"   Low density (10 users/km²):")
    print(f"   - Blocking: {low_density['blocking_probability']:.3f} (LOW)") 
    print(f"   - Throughput: {low_density['throughput_per_user']:.1f} Mbps (HIGH)")
    print(f"   - Fairness: {low_density['fairness_index']:.3f}")
    
    # ==================
    # Constellation Size  
    # ==================
    print("\n2. CONSTELLATION SIZE IMPACT:")
    small_constellation = results_df[results_df['constellation_size'] == 100].iloc[0]
    large_constellation = results_df[results_df['constellation_size'] == 2000].iloc[0]
    
    print(f"   Small constellation (100 sats):")
    print(f"   - Blocking: {small_constellation['blocking_probability']:.3f} (HIGH)")
    print(f"   Large constellation (2000 sats):")
    print(f"   - Blocking: {large_constellation['blocking_probability']:.3f} (LOW)")
    
    # =================
    # Revisit Interval
    # =================
    print("\n3. REVISIT INTERVAL IMPACT:")
    slow_revisit = results_df[results_df['revisit_interval'] == 30].iloc[0]
    fast_revisit = results_df[results_df['revisit_interval'] == 1].iloc[0]
    
    print(f"   Slow revisit (30 min):")
    print(f"   - Blocking: {slow_revisit['blocking_probability']:.3f} (HIGH)")
    print(f"   Fast revisit (1 min):") 
    print(f"   - Blocking: {fast_revisit['blocking_probability']:.3f} (LOW)")

def plot_performance_tradeoffs(results_df):
    """Create visualization plots for analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ================
    # Cluster Density
    # ================
    for constellation in results_df['constellation_size'].unique():
        subset = results_df[results_df['constellation_size'] == constellation]
        axes[0,0].plot(subset['cluster_density'], subset['blocking_probability'], 
                      label=f'{constellation} sats', marker='o')
    axes[0,0].set_xlabel('Cluster Density (users/km²)')
    axes[0,0].set_ylabel('Blocking Probability')
    axes[0,0].set_title('Cluster Density Impact: More Users = More Busy Signals')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # =================
    # Revisit Interval
    # =================
    for density in [10, 100, 200]:
        subset = results_df[results_df['cluster_density'] == density]
        axes[0,1].plot(subset['revisit_interval'], subset['throughput_per_user'],
                      label=f'{density} users/km²', marker='s')
    axes[0,1].set_xlabel('Revisit Interval (minutes)')
    axes[0,1].set_ylabel('Throughput per User (Mbps)')
    axes[0,1].set_title('Revisit Interval Impact: Faster Service = Better Throughput')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # ==================
    # Constellation Size  
    # ==================
    for density in [10, 100, 200]:
        subset = results_df[results_df['cluster_density'] == density]
        axes[1,0].plot(subset['constellation_size'], subset['fairness_index'],
                      label=f'{density} users/km²', marker='^')
    axes[1,0].set_xlabel('Constellation Size')
    axes[1,0].set_ylabel("Jain's Fairness Index")
    axes[1,0].set_title('Constellation Size Impact: More Satellites = Fairer Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Combined impact heatmap
    pivot_data = results_df.pivot_table(
        values='blocking_probability',
        index='cluster_density',
        columns='constellation_size'
    )
    sns.heatmap(pivot_data, ax=axes[1,1], cmap='YlOrRd', annot=True, fmt='.3f')
    axes[1,1].set_title('Combined Impact: Density vs Constellation Size')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run complete analysis
    results_df, simulator = run_comprehensive_analysis()
    
    # Analyze performance variability
    analyze_performance_trends(results_df)
    
    # Visualize results
    plot_performance_tradeoffs(results_df)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print("Key Findings:")
    print("• CLUSTER DENSITY: Blocking probability INCREASES, throughput DECREASES")
    print("• CONSTELLATION SIZE: More satellites REDUCES blocking, IMPROVES fairness") 
    print("• REVISIT INTERVAL: Shorter intervals IMPROVE all metrics")