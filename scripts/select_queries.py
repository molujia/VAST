"""Regenerate the final seed-42 HDBSCAN query plan from external inputs."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/"src"),str(ROOT)]

def main():
    from fixed_active_learning.config import load_fixed_config
    from fixed_active_learning.pipeline import run_fixed_active_learning
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--feature-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--verify-reference",action="store_true")
    args=parser.parse_args()
    plan=run_fixed_active_learning(load_fixed_config(args.config,data_root_override=args.feature_root),seed=42,output_path=args.output,verify_against_reference=args.verify_reference)
    print(plan["plan_id"])

if __name__=="__main__": main()
