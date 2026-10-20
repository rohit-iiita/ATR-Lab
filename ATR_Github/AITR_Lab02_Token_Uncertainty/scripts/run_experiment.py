import argparse
from pathlib import Path
from src.score_halueval import run_experiment

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--n_pairs',type=int,default=500)
    p.add_argument('--model_name',type=str,default='distilgpt2')
    p.add_argument('--output_dir',type=str,default='results/main')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max_length',type=int,default=512)
    p.add_argument('--batch_size',type=int,default=1)
    a=p.parse_args()
    run_experiment(a.n_pairs,a.model_name,Path(a.output_dir),a.seed,a.max_length,a.batch_size)
if __name__=='__main__': main()
