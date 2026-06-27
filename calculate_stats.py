import json
import numpy as np
from typing import Dict, List
import sys


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_statistics(data: Dict, metric: str = 'score') -> Dict:
    """
    Calculate mean, std, and best for each task and method
    
    Args:
        data: JSON data containing seeds and results
        metric: 'score' or 'cost'
    
    Returns:
        Dictionary with statistics for each task and method
    """
    tasks = data['tasks']
    methods = data['methods']
    seeds = data['seeds']
    
    stats = {}
    
    for task in tasks:
        stats[task] = {}
        
        for method in methods:
            # Collect values from all seeds
            values = []
            
            for seed in seeds:
                for result in seed['results']:
                    if result['task'] == task:
                        values.append(result[method][metric])
                        break
            
            # Calculate statistics
            values = np.array(values)
            stats[task][method] = {
                'mean': round(float(np.mean(values)), 4),
                'std': round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,  # Sample std
                'best': round(float(np.max(values)), 4)  # Assuming maximize
            }
    
    return stats


def print_table_header(tasks: List[str], metric_name: str = "Score"):
    """Print table header"""
    # Print table title
    print(f"\n{'='*120}")
    
    # Print column headers
    header = f"{'Strategy':<15}"
    for task in tasks:
        header += f"{task:>25}"
    print(header)
    
    # Print sub-headers (Avg and Best)
    subheader = " " * 15
    for _ in tasks:
        subheader += f"{'Avg':>12} {'Best':>13}"
    print(subheader)
    print("-" * 120)


def print_model_results(model_name: str, stats_score: Dict, stats_cost: Dict, methods: List[str], tasks: List[str]):
    """Print results for one model (backbone)"""
    print(f"\nBackbone: {model_name}")
    print("-" * 120)
    
    for method in methods:
        # Score row
        score_row = f"{method:<15}"
        for task in tasks:
            mean_score = stats_score[task][method]['mean']
            std_score = stats_score[task][method]['std']
            best_score = stats_score[task][method]['best']
            
            score_row += f"{mean_score:>8.4f} ± {std_score:<6.4f} {best_score:>8.4f}"
        print(score_row)
        
        # Cost row
        cost_row = f"{'(Cost)':<15}"
        for task in tasks:
            mean_cost = stats_cost[task][method]['mean']
            std_cost = stats_cost[task][method]['std']
            best_cost = stats_cost[task][method]['best']
            
            cost_row += f"{mean_cost:>8.4f} ± {std_cost:<6.4f} {best_cost:>8.4f}"
        print(cost_row)
        print()


def print_simple_table(model_name: str, stats_score: Dict, methods: List[str], tasks: List[str]):
    """Print simplified table showing only scores"""
    print(f"\n{'='*100}")
    print(f"Model: {model_name}")
    print(f"{'='*100}")
    
    # Header
    header = f"{'Strategy':<15}"
    for task in tasks:
        header += f"{task + ' (↑)':>20}"
    print(header)
    
    subheader = " " * 15
    for _ in tasks:
        subheader += f"{'Avg':>10} {'Best':>10}"
    print(subheader)
    print("-" * 100)
    
    # Data rows
    for method in methods:
        row = f"{method:<15}"
        for task in tasks:
            mean_score = stats_score[task][method]['mean']
            std_score = stats_score[task][method]['std']
            best_score = stats_score[task][method]['best']
            
            avg_str = f"{mean_score:.4f} ± {std_score:.3f}" if std_score > 0 else f"{mean_score:.4f} ±"
            row += f"{avg_str:>17} {best_score:>10.4f}"
        print(row)
    
    print("=" * 100)


def main():
    """Main function"""
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = input("Enter path to JSON file: ").strip()
    
    # Load data
    print(f"\nLoading data from: {filepath}")
    data = load_json(filepath)
    
    # Display basic info
    print(f"\nDataset: {data['dataset']}")
    print(f"Model: {data['model']}")
    print(f"Total seeds: {data['summary_statistics']['total_runs']}")
    print(f"Total tasks: {data['summary_statistics']['total_tasks']}")
    print(f"Methods: {', '.join(data['methods'])}")
    
    # Calculate statistics for score
    stats_score = calculate_statistics(data, metric='score')
    
    # Calculate statistics for cost
    stats_cost = calculate_statistics(data, metric='cost')
    
    # Print table
    print_simple_table(
        model_name=data['model'],
        stats_score=stats_score,
        methods=data['methods'],
        tasks=data['tasks']
    )
    
    # Print detailed statistics
    print(f"\n{'='*100}")
    print("DETAILED STATISTICS (Score / Cost)")
    print(f"{'='*100}\n")
    
    for task in data['tasks']:
        print(f"\n{'─'*80}")
        print(f"Task: {task}")
        print(f"{'─'*80}")
        print(f"{'Method':<15} {'Score (mean±std)':<25} {'Best Score':<15} {'Cost (mean±std)':<25} {'Best Cost':<15}")
        print("-" * 80)
        
        for method in data['methods']:
            score_mean = stats_score[task][method]['mean']
            score_std = stats_score[task][method]['std']
            score_best = stats_score[task][method]['best']
            
            cost_mean = stats_cost[task][method]['mean']
            cost_std = stats_cost[task][method]['std']
            cost_best = stats_cost[task][method]['best']
            
            score_str = f"{score_mean:.4f} ± {score_std:.4f}"
            cost_str = f"{cost_mean:.4f} ± {cost_std:.4f}"
            
            print(f"{method:<15} {score_str:<25} {score_best:<15.4f} {cost_str:<25} {cost_best:<15.4f}")
    
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()
