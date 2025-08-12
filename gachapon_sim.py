import random
import math
from typing import List, Dict, Any, Set, TypedDict
from scipy.stats import ttest_1samp

# --- 1. CONFIGURATION & SETUP ---

# --- Simulation Settings ---
NUM_SIMULATIONS = 10000 # Reduced for quicker testing, can be increased

# --- Gachapon Machine Contents ---
ITEMS: List[str] = ["Cat Keychain", "Dog Keychain", "Rabbit Figurine", "Hamster Sticker", "Rare Gold Cat"]
CAPSULES_PER_ITEM: int = 50
NUM_ITEM_TYPES: int = len(ITEMS)
TOTAL_CAPSULES: int = NUM_ITEM_TYPES * CAPSULES_PER_ITEM

# --- Customer Behavior Models ---
ITEM_POPULARITY: Dict[str, float] = {
    "Cat Keychain": 0.00,
    "Dog Keychain": 0.00,
    "Rabbit Figurine": 0.00,
    "Hamster Sticker": 0.00,
    "Rare Gold Cat": 1.0,  # High demand for the rare item
}

MAX_PULLS_PER_ITEM: Dict[str, Dict[int, float]] = {
    "Rare Gold Cat": {10000000: 1.0}, # Effectively infinite patience
    "Default": {10000000: 1.0}
}


# Define clear data structures for type hinting
class CustomerOutcome(TypedDict):
    desired_item: str
    got_item: bool
    pulls_taken: int

# --- CHANGE 1: The result of a simulation is now cleaner ---
# It no longer contains the massive state_history list.
class SimulationResult(TypedDict):
    snapshots: Dict[str, Dict[str, int]]
    customer_outcomes: List[CustomerOutcome]
    depletion_points: Dict[str, int]


# --- 2. SINGLE SIMULATION LOGIC (REWRITTEN) ---

def run_single_simulation() -> SimulationResult:
    """
    Runs one full simulation, capturing unbiased snapshots at the exact
    moment checkpoints are crossed.
    """
    gachapon_contents: Dict[str, int] = {item: CAPSULES_PER_ITEM for item in ITEMS}
    customer_outcomes: List[CustomerOutcome] = []
    depleted_pull_number: Dict[str, int] = {}
    total_pull_counter = 0

    # --- Unbiased Snapshot Logic ---
    # Define the levels and data structures for this specific run
    snapshots_found: Dict[str, Dict[str, int]] = {}
    snapshot_levels = {
        '100%': TOTAL_CAPSULES,
        '75%': int(TOTAL_CAPSULES * 0.75),
        '50%': int(TOTAL_CAPSULES * 0.50),
        '25%': int(TOTAL_CAPSULES * 0.25)
    }
    processed_levels: Set[str] = set()

    # Immediately capture the 100% state
    snapshots_found['100%'] = gachapon_contents.copy()
    processed_levels.add('100%')


    while sum(gachapon_contents.values()) > 0:
        desired_item = "Rare Gold Cat" # Simplified based on 100% popularity
        pull_distribution = MAX_PULLS_PER_ITEM["Rare Gold Cat"]
        max_pulls = list(pull_distribution.keys())[0]

        pulls_this_turn = 0
        got_item = False

        for _ in range(max_pulls):
            if sum(gachapon_contents.values()) == 0:
                break

            pulls_this_turn += 1
            total_pull_counter += 1

            capsule_pool = [item for item, count in gachapon_contents.items() for _ in range(count)]
            pulled_item = random.choice(capsule_pool)
            gachapon_contents[pulled_item] -= 1

            # --- CHANGE 2: The snapshot is taken HERE, after every single pull ---
            # This decouples the measurement from the turn-ending event.
            remaining_capsules = sum(gachapon_contents.values())
            for level_name, level_value in snapshot_levels.items():
                if remaining_capsules <= level_value and level_name not in processed_levels:
                    snapshots_found[level_name] = gachapon_contents.copy()
                    processed_levels.add(level_name)

            if gachapon_contents[pulled_item] == 0 and pulled_item not in depleted_pull_number:
                depleted_pull_number[pulled_item] = total_pull_counter

            if pulled_item == desired_item:
                got_item = True
                break

        customer_outcomes.append({"desired_item": desired_item, "got_item": got_item, "pulls_taken": pulls_this_turn})

    return {"snapshots": snapshots_found, "customer_outcomes": customer_outcomes,
            "depletion_points": depleted_pull_number}


# --- 3. AGGREGATION LOGIC (SIMPLIFIED) ---

class SimulationAggregator:
    """A class to aggregate results from multiple simulation runs."""

    def __init__(self, items: List[str]):
        self.items = items
        self.num_runs = 0
        self.snapshot_levels = ['100%', '75%', '50%', '25%']

        self.agg_snapshots = {level: {item: 0 for item in self.items} for level in self.snapshot_levels}
        self.agg_depletion_pulls = {item: 0 for item in self.items}
        self.agg_depletion_counts = {item: 0 for item in self.items}

        self.snapshot_rate_distributions = {
            level: {item: [] for item in self.items} for level in self.snapshot_levels
        }

    def add_result(self, result: SimulationResult):
        """Processes a single simulation result and adds it to the aggregate totals."""
        self.num_runs += 1
        self._aggregate_snapshots(result["snapshots"])
        self._aggregate_depletion_points(result["depletion_points"])

    # --- CHANGE 3: Snapshot aggregation is now much simpler ---
    def _aggregate_snapshots(self, snapshots_this_run: Dict[str, Dict[str, int]]):
        for level_name, state in snapshots_this_run.items():
            if level_name in self.snapshot_levels:
                for item in self.items:
                    self.agg_snapshots[level_name][item] += state.get(item, 0)

                remaining_capsules = sum(state.values())
                if remaining_capsules > 0:
                    for item in self.items:
                        rate_this_run = state.get(item, 0) / remaining_capsules
                        self.snapshot_rate_distributions[level_name][item].append(rate_this_run)

    def _aggregate_depletion_points(self, depletion_points: Dict[str, int]):
        for item, pull_num in depletion_points.items():
            self.agg_depletion_pulls[item] += pull_num
            self.agg_depletion_counts[item] += 1

    def calculate_final_report(self) -> Dict[str, Any]:
        """Calculates final averages and rates from all aggregated data."""
        if self.num_runs == 0: return {}
        return {
            "snapshots": {
                level: {item: count / self.num_runs for item, count in items.items()}
                for level, items in self.agg_snapshots.items()
            },
            "rate_distributions": self.snapshot_rate_distributions
        }

# --- 4. REPORTING ---

def _print_snapshots(report_data: Dict[str, Any]):
    print("\n--- Part 1: Machine State at Depletion Snapshots (Unbiased) ---")
    snapshots = report_data["snapshots"]
    for level_name in ['100%', '75%', '50%', '25%']:
        avg_counts = snapshots.get(level_name, {})
        if not avg_counts: continue
        total_avg_capsules = sum(avg_counts.values())
        print(f"\n  When machine is ~{level_name} FULL (Avg. {total_avg_capsules:.2f} capsules):")
        sorted_items = sorted(avg_counts.items(), key=lambda item_tuple: item_tuple[1], reverse=True)
        for item, avg_count in sorted_items:
            rate = (avg_count / total_avg_capsules) if total_avg_capsules > 0 else 0
            print(f"    - {item:<18}: {avg_count:>5.2f} avg. units | Rate: {rate:.2%}")

def _print_significance_analysis(report_data: Dict[str, Any]):
    print("\n\n--- Part 2: Statistical Significance Analysis ---")
    level_to_test = '25%'
    item_to_test = 'Rare Gold Cat'
    baseline_rate = 1 / NUM_ITEM_TYPES

    print(f"\n  Hypothesis Test for '{item_to_test}' at '{level_to_test}' Fullness:")
    observed_rates = report_data["rate_distributions"][level_to_test][item_to_test]
    if len(observed_rates) < 2:
        print("    Not enough data to perform significance test.")
        return

    t_statistic, p_value = ttest_1samp(a=observed_rates, popmean=baseline_rate)
    observed_mean = sum(observed_rates) / len(observed_rates)

    print(f"    - Null Hypothesis (H₀): The true average rate is equal to the baseline of {baseline_rate:.2%}.")
    print(f"    - Observed Mean Rate: {observed_mean:.4%}")
    print(f"    - p-value: {p_value:.4g}")
    alpha = 0.05
    if p_value < alpha:
        print(f"    - Conclusion: Since p < {alpha}, we reject the null hypothesis.")
    else:
        print(f"    - Conclusion: Since p >= {alpha}, we fail to reject the null hypothesis.")
        print("      The difference from the baseline is NOT statistically significant.")


def print_advanced_report(final_report_data: Dict[str, Any]):
    """Prints the final, formatted report to the console."""
    print("\n\n" + "=" * 70)
    print(f"    COMPREHENSIVE GACHAPON ANALYSIS ({NUM_SIMULATIONS} SIMULATIONS)")
    print("=" * 70)
    _print_snapshots(final_report_data)
    _print_significance_analysis(final_report_data)
    print("\n\n" + "=" * 70 + "\n--- END OF REPORT ---\n" + "=" * 70)


# --- 5. MAIN EXECUTION ---

def main():
    print(f"--- Running {NUM_SIMULATIONS} Simulations (Corrected Measurement) ---")
    aggregator = SimulationAggregator(items=ITEMS)

    # Use tqdm for a proper progress bar
    from tqdm import tqdm
    for _ in tqdm(range(NUM_SIMULATIONS), desc="Simulating"):
        single_run_result = run_single_simulation()
        aggregator.add_result(single_run_result)

    print("--- All simulations complete. Finalizing report. ---")
    final_results = aggregator.calculate_final_report()
    print_advanced_report(final_results)


if __name__ == "__main__":
    main()
