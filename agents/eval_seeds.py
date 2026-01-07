import click
from datetime import datetime
import numpy as np
from pathlib import Path
import importlib


def get_agent(logdir: str, nworkers: int = 32):
    basename = Path(logdir).name
    module = basename.split('-', 1)[0]
    m = importlib.import_module(module)
    c = m.config()
    c.nworkers = nworkers
    ag = m.AGENT(c)
    return ag


@click.command()
@click.option('--logdir')
@click.option('--l', default=1000)
@click.option('--r', default=1100)
@click.option('--n', default=32)
@click.option('--nworkers', default=32)
def eval_seeds(logdir: str, l: int, r: int, n: int, nworkers: int) -> None:
    actual_nworkers = max(n, nworkers)
    ag = get_agent(logdir, nworkers=actual_nworkers)
    logdir = Path(logdir)
    ag.load(logdir.joinpath('rainy-agent.pth').as_posix())
    rewards = []
    start = datetime.now()
    total_seeds = r - l
    for idx, i in enumerate(range(l, r), 1):
        seed_start = datetime.now()
        ag.config.seed = i
        res = ag.eval_parallel(n=n)
        rewards.append([r.reward for r in res])
        elapsed = (datetime.now() - start).total_seconds()
        seed_elapsed = (datetime.now() - seed_start).total_seconds()
        avg_time_per_seed = elapsed / idx
        remaining_seeds = total_seeds - idx
        estimated_remaining = avg_time_per_seed * remaining_seeds
        print(f'[{idx}/{total_seeds}] seed={i}: {len(res)} episodes, '
              f'{seed_elapsed:.1f}s, elapsed={elapsed/60:.1f}min, '
              f'est. remaining={estimated_remaining/60:.1f}min', flush=True)
    ag.close()
    r = np.array(rewards)
    total_elapsed = (datetime.now() - start).total_seconds()
    print(f'\nCompleted: {total_seeds} seeds, {total_elapsed/60:.1f} minutes')
    print(f'reward sum: {r.sum()}')
    np.save('{}/eval_seeds{}.npy'.format(logdir, n), r)


if __name__ == '__main__':
    eval_seeds()
