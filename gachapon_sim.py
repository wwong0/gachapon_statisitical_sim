import random
import math
from typing import List, Dict, Any, Set, TypedDict
from scipy.stats import ttest_1samp  # <-- Added import for statistical testing

# --- 1. CONFIGURATION & SETUP ---

# --- Simulation Settings ---
NUM_SIMULATIONS = 100000

# --- Gachapon Machine Contents ---
ITEMS: List[str] = ["Cat Keychain", "Dog Keychain", "Rabbit Figurine", "Hamster Sticker", "Rare Gold Cat"]
CAPSULES_PER_ITEM: int = 10
NUM_ITEM_TYPES: int = len(ITEMS)
TOTAL_CAPSULES: int = NUM_ITEM_TYPES * CAPSULES_PER_ITEM

# --- Customer Behavior Models ---
ITEM_POPULARITY: Dict[str, float] = {
    "Cat Keychain": 0.10,
    "Dog Keychain": 0.10,
    "Rabbit Figurine": 0.10,
    "Hamster Sticker": 0.10,
    "Rare Gold Cat": 0.60,  # High demand for the rare item
}

# ITEM-SPECIFIC PULL BEHAVIOR:
# Customers seeking the "Rare Gold Cat" have a distribution skewed towards more pulls.
# All other items use the "Default" low-patience distribution.
MAX_PULLS_PER_ITEM: Dict[str, Dict[int, float]] = {
    "Rare Gold Cat": {
        1: 0.10,  # Less likely to stop after one pull
        2: 0.15,
        3: 0.20,
        5: 0.25,  # Most likely to pull up to 5 times
        10: 0.20,
        20: 0.10  # 10% are "whales"
    },
    "Default": {
        1: 0.50,  # Most likely to stop after one pull
        2: 0.30,
        3: 0.15,
        5: 0.04,
        10: 0.01,
        20: 0.00  # No whales for common items
    }
}


# Define clear data structures for type hinting
class CustomerOutcome(TypedDict):
    desired_item: str
    got_item: bool
    pulls_taken: int


class SimulationResult(TypedDict):
    state_history: List[Dict[str, int]]
    customer_outcomes: List[CustomerOutcome]
    depletion_points: Dict[str, int]


# --- 2. SINGLE SIMULATION LOGIC ---

def run_single_simulation() -> SimulationResult:
    """Runs one full simulation from a full machine to an empty one."""
    gachapon_contents: Dict[str, int] = {item: CAPSULES_PER_ITEM for item in ITEMS}
    state_history: List[Dict[str, int]] = [gachapon_contents.copy()]
    customer_outcomes: List[CustomerOutcome] = []
    depleted_pull_number: Dict[str, int] = {}
    total_pull_counter = 0

    while sum(gachapon_contents.values()) > 0:
        desired_item = random.choices(population=list(ITEM_POPULARITY.keys()), weights=list(ITEM_POPULARITY.values()))[
            0]

        # Select the correct pull distribution based on the desired item
        pull_distribution = MAX_PULLS_PER_ITEM.get(desired_item, MAX_PULLS_PER_ITEM["Default"])

        max_pulls = random.choices(population=list(pull_distribution.keys()), weights=list(pull_distribution.values()))[
            0]

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
            if gachapon_contents[pulled_item] == 0 and pulled_item not in depleted_pull_number:
                depleted_pull_number[pulled_item] = total_pull_counter

            if pulled_item == desired_item:
                got_item = True
                break

        customer_outcomes.append({"desired_item": desired_item, "got_item": got_item, "pulls_taken": pulls_this_turn})
        state_history.append(gachapon_contents.copy())

    return {"state_history": state_history, "customer_outcomes": customer_outcomes,
            "depletion_points": depleted_pull_number}


# --- 3. AGGREGATION LOGIC ---

class SimulationAggregator:
    """A class to aggregate results from multiple simulation runs."""

    def __init__(self, items: List[str], max_pull_limit: int):
        self.items = items
        self.num_runs = 0
        self.snapshot_levels = {'100%': TOTAL_CAPSULES,
                                '75%': int(TOTAL_CAPSULES * 0.75),
                                '50%': int(TOTAL_CAPSULES * 0.50),
                                '25%': int(TOTAL_CAPSULES * 0.25),
                                '0%': 0}

        # Data structures for accumulating averages
        self.agg_snapshots = {level: {item: 0 for item in self.items} for level in self.snapshot_levels}
        self.agg_success_by_item = {item: {'success': 0, 'failure': 0} for item in self.items}
        self.agg_success_by_pull_count = {item: {i: 0 for i in range(1, max_pull_limit + 1)} for item in self.items}
        self.agg_depletion_pulls = {item: 0 for item in self.items}
        self.agg_depletion_counts = {item: 0 for item in self.items}
        self.agg_pulls_successful = 0
        self.agg_pulls_failed = 0
        self.total_successful_customers = 0
        self.total_failed_customers = 0
        self.agg_lifecycle_rates = {item: 0.0 for item in self.items}
        self.total_lifecycle_steps = 0

        # [NEW] Store the individual rate from each run for significance testing
        self.snapshot_rate_distributions = {
            level: {item: [] for item in self.items} for level in self.snapshot_levels
        }

    def add_result(self, result: SimulationResult):
        """Processes a single simulation result and adds it to the aggregate totals."""
        self.num_runs += 1
        self._aggregate_snapshots(result["state_history"])
        self._aggregate_customer_outcomes(result["customer_outcomes"])
        self._aggregate_depletion_points(result["depletion_points"])
        self._aggregate_lifecycle_rates(result["state_history"])

    def _aggregate_snapshots(self, state_history: List[Dict[str, int]]):
        processed_snapshots: Set[str] = set()
        # Iterate backwards to find the *first* time the state drops below a threshold
        for state in reversed(state_history):
            remaining_capsules = sum(state.values())
            for level_name, level_value in self.snapshot_levels.items():
                if remaining_capsules >= level_value and level_name not in processed_snapshots:
                    for item in self.items:
                        self.agg_snapshots[level_name][item] += state.get(item, 0)

                        # [NEW] Calculate and store the rate for this specific run
                        if remaining_capsules > 0:
                            rate_this_run = state.get(item, 0) / remaining_capsules
                            self.snapshot_rate_distributions[level_name][item].append(rate_this_run)
                        else:
                            self.snapshot_rate_distributions[level_name][item].append(0)

                    processed_snapshots.add(level_name)

    def _aggregate_customer_outcomes(self, customer_outcomes: List[CustomerOutcome]):
        for outcome in customer_outcomes:
            desired, pulls, got_item = outcome["desired_item"], outcome["pulls_taken"], outcome["got_item"]
            if got_item:
                self.agg_success_by_item[desired]['success'] += 1
                self.agg_pulls_successful += pulls
                self.total_successful_customers += 1
                if pulls in self.agg_success_by_pull_count[desired]:
                    self.agg_success_by_pull_count[desired][pulls] += 1
            else:
                self.agg_success_by_item[desired]['failure'] += 1
                self.agg_pulls_failed += pulls
                self.total_failed_customers += 1

    def _aggregate_depletion_points(self, depletion_points: Dict[str, int]):
        for item, pull_num in depletion_points.items():
            self.agg_depletion_pulls[item] += pull_num
            self.agg_depletion_counts[item] += 1

    def _aggregate_lifecycle_rates(self, state_history: List[Dict[str, int]]):
        """Aggregates the pull rate for each item at every step of the simulation."""
        for state in state_history:
            total_remaining = sum(state.values())
            if total_remaining > 0:
                self.total_lifecycle_steps += 1
                for item in self.items:
                    rate = state.get(item, 0) / total_remaining
                    self.agg_lifecycle_rates[item] += rate

    def calculate_final_report(self) -> Dict[str, Any]:
        """Calculates final averages and rates from all aggregated data."""
        if self.num_runs == 0:
            return {}

        final_report = {
            "snapshots": {
                level: {item: count / self.num_runs for item, count in items.items()}
                for level, items in self.agg_snapshots.items()
            },
            "overall_lifecycle_rates": {
                item: self.agg_lifecycle_rates[
                          item] / self.total_lifecycle_steps if self.total_lifecycle_steps > 0 else 0
                for item in self.items
            },
            "success_rate_by_item": {
                item: data['success'] / (data['success'] + data['failure']) if (data['success'] + data[
                    'failure']) > 0 else 0
                for item, data in self.agg_success_by_item.items()
            },
            "avg_pulls_to_depletion": {
                item: self.agg_depletion_pulls[item] / self.agg_depletion_counts[item] if self.agg_depletion_counts[
                                                                                              item] > 0 else float(
                    'inf')
                for item in self.items
            },
            "avg_pulls_successful": self.agg_pulls_successful / self.total_successful_customers if self.total_successful_customers > 0 else 0,
            "avg_pulls_failed": self.agg_pulls_failed / self.total_failed_customers if self.total_failed_customers > 0 else 0,
            "total_successes_per_item": {item: data['success'] for item, data in self.agg_success_by_item.items()},
            "success_by_pull_count": self.agg_success_by_pull_count,
            # [NEW] Pass the raw distributions through for testing
            "rate_distributions": self.snapshot_rate_distributions
        }
        return final_report


# --- 4. REPORTING ---

def _print_snapshots(report_data: Dict[str, Any]):
    print("\n--- Part 1: Machine State at Depletion Snapshots ---")
    print("Shows the average item counts and pull rates at different fullness levels.")
    snapshots = report_data["snapshots"]
    for level_name in ['100%', '75%', '50%', '25%']:
        avg_counts = snapshots[level_name]
        total_avg_capsules = sum(avg_counts.values())
        print(f"\n  When machine is ~{level_name} FULL (Avg. {total_avg_capsules:.2f} capsules):")
        sorted_items = sorted(avg_counts.items(), key=lambda item_tuple: item_tuple[1], reverse=True)
        for item, avg_count in sorted_items:
            rate = (avg_count / total_avg_capsules) if total_avg_capsules > 0 else 0
            print(f"    - {item:<18}: {avg_count:>5.2f} avg. units | Rate: {rate:.2%}")


def _print_overall_rates(report_data: Dict[str, Any]):
    """Prints the overall lifecycle average rates."""
    print("\n\n--- Part 2: Overall Lifecycle Average Rates ---")
    print("This is the average chance of getting an item on any single pull, at any time.")
    sorted_rates = sorted(report_data["overall_lifecycle_rates"].items(), key=lambda item_tuple: item_tuple[1],
                          reverse=True)
    for item, rate in sorted_rates:
        print(f"  - {item:<18}: {rate:.2%}")


def _print_lifecycle(report_data: Dict[str, Any]):
    print("\n\n--- Part 3: Item Lifecycle & Performance ---")
    print("\nAverage Total Pulls Until An Item Sells Out:")
    sorted_depletion = sorted(report_data["avg_pulls_to_depletion"].items(), key=lambda item_tuple: item_tuple[1])
    for item, avg_pulls in sorted_depletion:
        print(f"  - {item:<18}: {avg_pulls:>6.2f} pulls")


def _print_customer_experience(report_data: Dict[str, Any]):
    print("\n\n--- Part 4: Customer Experience & Success Rates ---")
    print("\nOverall Success Rate for Customers Seeking a Specific Item:")
    sorted_success = sorted(report_data["success_rate_by_item"].items(), key=lambda item_tuple: item_tuple[1],
                            reverse=True)
    for item, rate in sorted_success:
        print(f"  - Seeking '{item:<18}': {rate:>7.2%} chance of success")

    print("\nAverage Number of Pulls Customers Make per Visit:")
    print(f"  - Successful Customers: {report_data['avg_pulls_successful']:.2f} pulls on average to get their item.")
    print(f"  - Unsuccessful Customers: {report_data['avg_pulls_failed']:.2f} pulls on average before giving up.")


def _print_success_deep_dive(report_data: Dict[str, Any]):
    print("\n\n--- Part 5: Deep Dive - How Success Happens ---")
    print("For customers who SUCCEEDED, this shows how many pulls it took them.")
    for item in ITEMS:
        print(f"\n  Analysis for '{item}':")
        total_successes = report_data["total_successes_per_item"][item]
        if total_successes == 0:
            print("    No successful customers recorded for this item.")
            continue

        pull_counts = report_data["success_by_pull_count"][item]
        sorted_pulls = sorted(pull_counts.items())
        cumulative_pct = 0
        for pull_num, count in sorted_pulls:
            if count > 0:
                pct = count / total_successes
                cumulative_pct += pct
                print(f"    - On pull #{pull_num:<2}: {pct:>6.2%} of successes (Cumulative: {cumulative_pct:>7.2%})")


def _print_significance_analysis(report_data: Dict[str, Any]):
    """[NEW] Performs and prints a t-test analysis."""
    print("\n\n--- Part 6: Statistical Significance Analysis ---")
    print("This tests if an observed rate is statistically different from the initial baseline.")

    # We will test the 'Rare Gold Cat' rate at the 25% fullness level,
    # as this is where its rate is likely to be most distorted by demand.
    level_to_test = '25%'
    item_to_test = 'Rare Gold Cat'
    # The baseline rate is the initial physical chance of pulling the item (1 out of 5 types)
    baseline_rate = 1 / NUM_ITEM_TYPES

    print(f"\n  Hypothesis Test for '{item_to_test}' at '{level_to_test}' Fullness:")

    observed_rates = report_data["rate_distributions"][level_to_test][item_to_test]

    if not observed_rates or len(observed_rates) < 2:
        print("    Not enough data to perform significance test.")
        return

    # Perform the one-sample t-test against the initial population mean (baseline rate)
    t_statistic, p_value = ttest_1samp(a=observed_rates, popmean=baseline_rate)
    observed_mean = sum(observed_rates) / len(observed_rates)

    print(f"    - Null Hypothesis (H₀): The true average rate is equal to the baseline of {baseline_rate:.2%}.")
    print(f"    - Observed Mean Rate: {observed_mean:.4%}")
    print(f"    - p-value: {p_value:.4g}")

    # Interpret the result using a standard alpha level
    alpha = 0.05
    if p_value < alpha:
        print(f"    - Conclusion: Since p < {alpha}, we reject the null hypothesis.")
        print(f"      The observed rate for '{item_to_test}' is **statistically significant**.")
    else:
        print(f"    - Conclusion: Since p >= {alpha}, we fail to reject the null hypothesis.")
        print("      The difference from the baseline is not statistically significant.")


def print_advanced_report(final_report_data: Dict[str, Any]):
    """Prints the final, formatted report to the console."""
    print("\n\n" + "=" * 70)
    print(f"    COMPREHENSIVE GACHAPON ANALYSIS ({NUM_SIMULATIONS} SIMULATIONS)")
    print("=" * 70)

    _print_snapshots(final_report_data)
    _print_overall_rates(final_report_data)
    _print_lifecycle(final_report_data)
    _print_customer_experience(final_report_data)
    _print_success_deep_dive(final_report_data)
    _print_significance_analysis(final_report_data)  # <-- New report section added

    print("\n\n" + "=" * 70)
    print("--- END OF REPORT ---")
    print("=" * 70)


# --- 5. MAIN EXECUTION ---

def main():
    """Main function to run the simulation and print the report."""
    # --- Validation ---
    if not math.isclose(sum(ITEM_POPULARITY.values()), 1.0):
        print(f"Error: ITEM_POPULARITY weights must sum to 1.0, but they sum to {sum(ITEM_POPULARITY.values())}")
        return

    max_pull_limit = 0
    for item_name, dist in MAX_PULLS_PER_ITEM.items():
        if not math.isclose(sum(dist.values()), 1.0):
            print(f"Error: Distribution for '{item_name}' must sum to 1.0, but sums to {sum(dist.values())}")
            return
        # Find the absolute maximum number of pulls possible across all distributions
        max_pull_limit = max(max_pull_limit, max(dist.keys()))

    # --- Orchestration ---
    print(f"--- Running {NUM_SIMULATIONS} Simulations with Item-Specific Pull Behavior ---")
    aggregator = SimulationAggregator(items=ITEMS, max_pull_limit=max_pull_limit)

    for i in range(NUM_SIMULATIONS):
        if (i + 1) % (NUM_SIMULATIONS // 10 or 1) == 0:
            progress = (i + 1) / NUM_SIMULATIONS
            print(f"  Progress: {i + 1}/{NUM_SIMULATIONS} ({progress:.0%})")

        single_run_result = run_single_simulation()
        aggregator.add_result(single_run_result)

    print("--- All simulations complete. Finalizing report. ---")
    final_results = aggregator.calculate_final_report()
    print_advanced_report(final_results)


if __name__ == "__main__":
    main()
