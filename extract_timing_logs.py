#!/usr/bin/env python3
"""
Elastic EP Scale Down Timing Log Extractor

This script extracts timing information from vLLM elastic EP scale down logs
and groups the same timing measurements across multiple workers.

Usage:
    python extract_timing_logs.py <log_file>
    cat log_file | python extract_timing_logs.py
    python extract_timing_logs.py --help
"""

import argparse
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional
import json


class TimingExtractor:
    """Extract and analyze timing information from vLLM logs."""
    
    def __init__(self):
        # Define timing patterns for all known timing categories
        self.timing_patterns = [
            # Level 1: Top-level Elastic EP Scale Down
            r'\[Elastic EP Scale Down Timing\] Total: ([\d.]+)ms',
            
            # Level 2: Engine Core Reinitialization  
            r'\[Engine Core Reinit Timing\] Cleanup & Shutdown: ([\d.]+)ms',
            r'\[Engine Core Reinit Timing\] Config Update: ([\d.]+)ms',
            r'\[Engine Core Reinit Timing\] Model Executor Reinit: ([\d.]+)ms',
            
            # Level 3: GPU Worker Reinitialization
            r'\[Elastic EP Scale Down Timing\] EPLB Rearrangement: ([\d.]+)ms',
            r'\[GPU Worker Reinit Timing\] Cleanup Dist Env: ([\d.]+)ms',
            r'\[GPU Worker Reinit Timing\] Reconfigure Parallel Config: ([\d.]+)ms',
            r'\[GPU Worker Reinit Timing\] Init Worker Distributed Env: ([\d.]+)ms',
            r'\[GPU Worker Reinit Timing\] Reconfigure MoE: ([\d.]+)ms',
            r'\[GPU Worker Reinit Timing\] Total: ([\d.]+)ms',
            
            # Level 4: Worker Distributed Environment Init
            r'\[Worker Distributed Init Timing\] Batch Invariance Init: ([\d.]+)ms',
            r'\[Worker Distributed Init Timing\] Custom All-Reduce Config: ([\d.]+)ms',
            r'\[Worker Distributed Init Timing\] Distributed Environment Init: ([\d.]+)ms',
            r'\[Worker Distributed Init Timing\] Model Parallel Init: ([\d.]+)ms',
            r'\[Worker Distributed Init Timing\] EC Transfer Init: ([\d.]+)ms',
            r'\[Worker Distributed Init Timing\] Total: ([\d.]+)ms',
            
            # Level 5: Model Parallel Groups Creation
            r'\[Model Parallel Init Timing\] TP Group Creation: ([\d.]+)ms',
            r'\[Model Parallel Init Timing\] DCP Group Creation: ([\d.]+)ms',
            r'\[Model Parallel Init Timing\] PP Group Creation: ([\d.]+)ms',
            r'\[Model Parallel Init Timing\] DP Group Creation: ([\d.]+)ms',
            r'\[Model Parallel Init Timing\] EP Group Creation: ([\d.]+)ms',
            r'\[Model Parallel Init Timing\] Total Group Creation: ([\d.]+)ms',
            
            # Level 6: EPLB Expert Rebalancing
            r'\[EPLB Timing\] Load Information Preprocessing: ([\d.]+)ms',
            r'\[EPLB Timing\] Expert Rebalance Algorithm: ([\d.]+)ms',
            r'\[EPLB Timing\] Expert Weights Transfer: ([\d.]+)ms',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [(re.compile(pattern), pattern) for pattern in self.timing_patterns]
        
        # Storage for extracted timings
        self.timings: Dict[str, List[float]] = defaultdict(list)
        
    def extract_timing_name(self, pattern: str) -> str:
        """Extract timing name from regex pattern."""
        # Extract the timing description between brackets and colon
        match = re.search(r'\[(.*?)\].*?:\s*\(', pattern)
        if match:
            return match.group(1)
        return pattern
    
    def process_line(self, line: str) -> None:
        """Process a single log line to extract timing information."""
        for compiled_pattern, original_pattern in self.compiled_patterns:
            match = compiled_pattern.search(line)
            if match:
                timing_value = float(match.group(1))
                timing_name = self.extract_timing_name(original_pattern)
                self.timings[timing_name].append(timing_value)
                break  # Only match the first pattern to avoid duplicates
    
    def process_log(self, log_source) -> None:
        """Process log from file or stdin."""
        for line in log_source:
            line = line.strip()
            if line:
                self.process_line(line)
    
    def calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of timing values."""
        if not values:
            return {}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'total': sum(values)
        }
    
    def print_hierarchical_report(self) -> None:
        """Print timing results in hierarchical format."""
        print("🌲 Elastic EP Scale Down Timing Analysis")
        print("=" * 80)
        
        # Define hierarchy groups
        hierarchy = {
            "📊 Level 1 - Top Level Scale Down": [
                "Elastic EP Scale Down Timing] Total"
            ],
            "📊 Level 2 - Engine Core Reinitialization": [
                "Engine Core Reinit Timing] Cleanup & Shutdown",
                "Engine Core Reinit Timing] Config Update", 
                "Engine Core Reinit Timing] Model Executor Reinit"
            ],
            "📊 Level 3 - GPU Worker Reinitialization": [
                "Elastic EP Scale Down Timing] EPLB Rearrangement",
                "GPU Worker Reinit Timing] Cleanup Dist Env",
                "GPU Worker Reinit Timing] Reconfigure Parallel Config",
                "GPU Worker Reinit Timing] Init Worker Distributed Env",
                "GPU Worker Reinit Timing] Reconfigure MoE",
                "GPU Worker Reinit Timing] Total"
            ],
            "📊 Level 4 - Worker Distributed Environment Init": [
                "Worker Distributed Init Timing] Batch Invariance Init",
                "Worker Distributed Init Timing] Custom All-Reduce Config",
                "Worker Distributed Init Timing] Distributed Environment Init",
                "Worker Distributed Init Timing] Model Parallel Init",
                "Worker Distributed Init Timing] EC Transfer Init",
                "Worker Distributed Init Timing] Total"
            ],
            "📊 Level 5 - Model Parallel Groups Creation": [
                "Model Parallel Init Timing] TP Group Creation",
                "Model Parallel Init Timing] DCP Group Creation",
                "Model Parallel Init Timing] PP Group Creation",
                "Model Parallel Init Timing] DP Group Creation",
                "Model Parallel Init Timing] EP Group Creation",
                "Model Parallel Init Timing] Total Group Creation"
            ],
            "📊 Level 6 - EPLB Expert Rebalancing": [
                "EPLB Timing] Load Information Preprocessing",
                "EPLB Timing] Expert Rebalance Algorithm",
                "EPLB Timing] Expert Weights Transfer"
            ]
        }
        
        for level_name, timing_keys in hierarchy.items():
            print(f"\n{level_name}")
            print("-" * 60)
            
            level_found = False
            for timing_key in timing_keys:
                if timing_key in self.timings:
                    level_found = True
                    values = self.timings[timing_key]
                    stats = self.calculate_stats(values)
                    
                    print(f"  {timing_key}")
                    if stats['count'] == 1:
                        print(f"    Value: {stats['avg']:.2f}ms")
                    else:
                        print(f"    Workers: {stats['count']}")
                        print(f"    Min: {stats['min']:.2f}ms | Max: {stats['max']:.2f}ms | Avg: {stats['avg']:.2f}ms")
                        print(f"    Total: {stats['total']:.2f}ms")
            
            if not level_found:
                print("  No timing data found for this level")
    
    def print_summary_table(self) -> None:
        """Print a summary table of all timings."""
        print("\n📈 Summary Table")
        print("=" * 100)
        print(f"{'Timing Category':<50} {'Workers':<8} {'Min(ms)':<10} {'Max(ms)':<10} {'Avg(ms)':<10} {'Total(ms)':<12}")
        print("-" * 100)
        
        # Sort by timing category for better readability
        for timing_name in sorted(self.timings.keys()):
            values = self.timings[timing_name]
            stats = self.calculate_stats(values)
            
            if stats['count'] == 1:
                print(f"{timing_name:<50} {stats['count']:<8} {stats['avg']:<10.2f} {stats['avg']:<10.2f} {stats['avg']:<10.2f} {stats['avg']:<12.2f}")
            else:
                print(f"{timing_name:<50} {stats['count']:<8} {stats['min']:<10.2f} {stats['max']:<10.2f} {stats['avg']:<10.2f} {stats['total']:<12.2f}")
    
    def export_json(self, output_file: str) -> None:
        """Export timing data to JSON format."""
        output_data = {}
        for timing_name, values in self.timings.items():
            stats = self.calculate_stats(values)
            output_data[timing_name] = {
                'values': values,
                'statistics': stats
            }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Timing data exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract timing information from vLLM elastic EP scale down logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process log file
    python extract_timing_logs.py server.log
    
    # Process from stdin
    cat server.log | python extract_timing_logs.py
    
    # Export to JSON
    python extract_timing_logs.py server.log --json output.json
    
    # Show only summary table
    python extract_timing_logs.py server.log --table-only
        """
    )
    
    parser.add_argument('log_file', nargs='?', help='Log file to process (use stdin if not provided)')
    parser.add_argument('--json', metavar='FILE', help='Export results to JSON file')
    parser.add_argument('--table-only', action='store_true', help='Show only summary table')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = TimingExtractor()
    
    # Process log source
    try:
        if args.log_file:
            with open(args.log_file, 'r') as f:
                extractor.process_log(f)
            print(f"📁 Processed log file: {args.log_file}")
        else:
            extractor.process_log(sys.stdin)
            print("📁 Processed log from stdin")
    except FileNotFoundError:
        print(f"❌ Error: Log file '{args.log_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error processing log: {e}")
        sys.exit(1)
    
    # Check if any timing data was found
    if not extractor.timings:
        print("⚠️  No timing data found in the log")
        sys.exit(0)
    
    # Print results
    if args.table_only:
        extractor.print_summary_table()
    else:
        extractor.print_hierarchical_report()
        extractor.print_summary_table()
    
    # Export to JSON if requested
    if args.json:
        extractor.export_json(args.json)
    
    print(f"\n🎯 Total timing categories found: {len(extractor.timings)}")
    total_measurements = sum(len(values) for values in extractor.timings.values())
    print(f"📊 Total timing measurements: {total_measurements}")


if __name__ == "__main__":
    main()
