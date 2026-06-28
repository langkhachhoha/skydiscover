import json
import numpy as np
from typing import Dict, List, Tuple
import sys


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_statistics(data: Dict, metric: str = 'score') -> Dict:
    """Calculate mean, std, and best for each task and method"""
    tasks = data['tasks']
    methods = data['methods']
    seeds = data['seeds']
    
    stats = {}
    
    for task in tasks:
        stats[task] = {}
        
        for method in methods:
            values = []
            
            for seed in seeds:
                for result in seed['results']:
                    if result['task'] == task:
                        values.append(result[method][metric])
                        break
            
            values = np.array(values)
            stats[task][method] = {
                'mean': round(float(np.mean(values)), 4),
                'std': round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
                'best': round(float(np.max(values)), 4)
            }
    
    return stats


def compare_all_models():
    """Compare results across all models and datasets"""
    
    files = {
        "KIMI K2 - Math": "math_kimi.json",
        "GPT-5 - Math": "math_gpt.json",
        "KIMI K2 - System": "system_kimi.json",
        "GPT-5 - System": "system_gpt.json"
    }
    
    print("\n" + "="*120)
    print("COMPREHENSIVE COMPARISON: All Models & Datasets")
    print("="*120 + "\n")
    
    all_data = {}
    
    # Load all data
    for label, filepath in files.items():
        try:
            data = load_json(filepath)
            stats_score = calculate_statistics(data, metric='score')
            stats_cost = calculate_statistics(data, metric='cost')
            all_data[label] = {
                'data': data,
                'stats_score': stats_score,
                'stats_cost': stats_cost
            }
            print(f"✓ Loaded: {label} ({filepath})")
        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
    
    print("\n" + "="*120)
    
    # Compare by dataset
    print("\n" + "━"*120)
    print("COMPARISON BY DATASET")
    print("━"*120)
    
    # Mathematical Discovery Task Comparison
    if "KIMI K2 - Math" in all_data and "GPT-5 - Math" in all_data:
        print("\n📊 MATHEMATICAL DISCOVERY TASK: KIMI K2 vs GPT-5")
        print("-"*120)
        compare_two_models(
            all_data["KIMI K2 - Math"],
            all_data["GPT-5 - Math"],
            "KIMI K2",
            "GPT-5"
        )
    
    # System Engineering Task Comparison
    if "KIMI K2 - System" in all_data and "GPT-5 - System" in all_data:
        print("\n📊 SYSTEM ENGINEERING TASK: KIMI K2 vs GPT-5")
        print("-"*120)
        compare_two_models(
            all_data["KIMI K2 - System"],
            all_data["GPT-5 - System"],
            "KIMI K2",
            "GPT-5"
        )
    
    # Compare by model
    print("\n" + "━"*120)
    print("COMPARISON BY MODEL")
    print("━"*120)
    
    # KIMI K2: Math vs System
    if "KIMI K2 - Math" in all_data and "KIMI K2 - System" in all_data:
        print("\n🤖 KIMI K2: Mathematical Discovery vs System Engineering")
        print("-"*120)
        compare_datasets(
            all_data["KIMI K2 - Math"],
            all_data["KIMI K2 - System"],
            "KIMI K2"
        )
    
    # GPT-5: Math vs System
    if "GPT-5 - Math" in all_data and "GPT-5 - System" in all_data:
        print("\n🤖 GPT-5: Mathematical Discovery vs System Engineering")
        print("-"*120)
        compare_datasets(
            all_data["GPT-5 - Math"],
            all_data["GPT-5 - System"],
            "GPT-5"
        )
    
    # Method-wise performance summary
    print("\n" + "━"*120)
    print("METHOD PERFORMANCE SUMMARY")
    print("━"*120)
    summarize_methods(all_data)
    
    print("\n" + "="*120 + "\n")


def compare_two_models(data1: Dict, data2: Dict, model1_name: str, model2_name: str):
    """Compare two models on the same dataset"""
    
    tasks = data1['data']['tasks']
    methods = data1['data']['methods']
    
    print(f"\n{'Method':<15} {'Metric':<10} {model1_name:>20} {model2_name:>20} {'Winner':<15}")
    print("-"*90)
    
    for method in methods:
        # Calculate average score across all tasks
        scores1 = [data1['stats_score'][task][method]['mean'] for task in tasks]
        scores2 = [data2['stats_score'][task][method]['mean'] for task in tasks]
        
        costs1 = [data1['stats_cost'][task][method]['mean'] for task in tasks]
        costs2 = [data2['stats_cost'][task][method]['mean'] for task in tasks]
        
        avg_score1 = np.mean(scores1)
        avg_score2 = np.mean(scores2)
        avg_cost1 = np.mean(costs1)
        avg_cost2 = np.mean(costs2)
        
        # Determine winner
        score_winner = model1_name if avg_score1 > avg_score2 else model2_name
        cost_winner = model1_name if avg_cost1 < avg_cost2 else model2_name
        
        print(f"{method:<15} {'Score':<10} {avg_score1:>20.4f} {avg_score2:>20.4f} {score_winner:<15}")
        print(f"{'':<15} {'Cost':<10} {avg_cost1:>20.4f} {avg_cost2:>20.4f} {cost_winner:<15}")
        print()


def compare_datasets(data1: Dict, data2: Dict, model_name: str):
    """Compare two datasets for the same model"""
    
    methods = data1['data']['methods']
    
    print(f"\nModel: {model_name}")
    print(f"{'Method':<15} {'Dataset':<20} {'Avg Score':>15} {'Avg Cost':>15}")
    print("-"*70)
    
    for method in methods:
        tasks1 = data1['data']['tasks']
        tasks2 = data2['data']['tasks']
        
        scores1 = [data1['stats_score'][task][method]['mean'] for task in tasks1]
        scores2 = [data2['stats_score'][task][method]['mean'] for task in tasks2]
        
        costs1 = [data1['stats_cost'][task][method]['mean'] for task in tasks1]
        costs2 = [data2['stats_cost'][task][method]['mean'] for task in tasks2]
        
        avg_score1 = np.mean(scores1)
        avg_score2 = np.mean(scores2)
        avg_cost1 = np.mean(costs1)
        avg_cost2 = np.mean(costs2)
        
        print(f"{method:<15} {'Math Discovery':<20} {avg_score1:>15.4f} {avg_cost1:>15.4f}")
        print(f"{'':<15} {'System Engineering':<20} {avg_score2:>15.4f} {avg_cost2:>15.4f}")
        print()


def summarize_methods(all_data: Dict):
    """Summarize method performance across all datasets"""
    
    methods = ['OpenEvolve', 'GEPA', 'AdaEvolve', 'EvoX', 'SpecEvo']
    
    print(f"\n{'Method':<15} {'Total Wins (Score)':<20} {'Total Wins (Cost)':<20} {'Overall Rank':<15}")
    print("-"*70)
    
    method_stats = {method: {'score_wins': 0, 'cost_wins': 0} for method in methods}
    
    # Count wins for each method
    for label, data_dict in all_data.items():
        tasks = data_dict['data']['tasks']
        
        for task in tasks:
            # Find best score
            best_score = max([
                data_dict['stats_score'][task][method]['best'] 
                for method in methods
            ])
            
            # Find best cost (lowest)
            best_cost = min([
                data_dict['stats_cost'][task][method]['mean'] 
                for method in methods
            ])
            
            # Award wins
            for method in methods:
                if data_dict['stats_score'][task][method]['best'] == best_score:
                    method_stats[method]['score_wins'] += 1
                if data_dict['stats_cost'][task][method]['mean'] == best_cost:
                    method_stats[method]['cost_wins'] += 1
    
    # Rank methods by total wins
    ranked = sorted(
        method_stats.items(), 
        key=lambda x: x[1]['score_wins'] + x[1]['cost_wins'], 
        reverse=True
    )
    
    for rank, (method, stats) in enumerate(ranked, 1):
        print(f"{method:<15} {stats['score_wins']:<20} {stats['cost_wins']:<20} {'#' + str(rank):<15}")
    
    print("\nNote: Wins are counted based on best score and lowest average cost per task.")


def main():
    """Main function"""
    compare_all_models()


if __name__ == "__main__":
    main()
