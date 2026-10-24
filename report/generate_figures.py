"""Regenerate Banking77 experiment figures from exported summary/history CSVs.

Usage:
    python report/generate_figures.py --data /path/to/report/figures

The input directory must contain experiment_summary.csv and training_history.csv.
Outputs are written to the same directory as PNG and PDF files.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def save(fig, directory: Path, stem: str):
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(directory / f'{stem}.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    out = args.data
    summary = pd.read_csv(out / 'experiment_summary.csv')
    history = pd.read_csv(out / 'training_history.csv')

    required_summary = {'model', 'dimension', 'test_macro_f1', 'test_accuracy'}
    required_history = {'model', 'dimension', 'epoch'}
    if not required_summary.issubset(summary.columns):
        raise ValueError(f'Missing summary columns: {required_summary - set(summary.columns)}')
    if not required_history.issubset(history.columns):
        raise ValueError(f'Missing history columns: {required_history - set(history.columns)}')

    summary = summary.sort_values(['model', 'dimension'])
    for metric, title in [('test_macro_f1', 'Test Macro-F1'), ('test_accuracy', 'Test Accuracy')]:
        fig, ax = plt.subplots(figsize=(9, 5))
        names = [f'{r.model}\n{int(r.dimension)}' for r in summary.itertuples()]
        ax.bar(names, summary[metric])
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.tick_params(axis='x', labelsize=9)
        save(fig, out, metric)

    metrics = [
        ('validation_macro_f1', 'Validation Macro-F1'),
        ('validation_accuracy', 'Validation Accuracy'),
        ('training_loss', 'Training Loss'),
        ('validation_loss', 'Validation Loss'),
    ]
    for model, part in history.groupby('model'):
        slug = 'llama' if 'llama' in model.lower() else 'phi4' if 'phi' in model.lower() else model.lower().replace(' ', '_')
        for metric, label in metrics:
            if metric not in history.columns:
                raise ValueError(f'Missing training history column: {metric}')
            fig, ax = plt.subplots(figsize=(8, 5))
            for dimension, group in part.groupby('dimension'):
                group = group.sort_values('epoch')
                ax.plot(group['epoch'], group[metric], marker='o', label=f'{dimension} dims')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(label)
            ax.set_title(f'{model} — {label}')
            ax.legend()
            ax.grid(alpha=0.25)
            save(fig, out, f'{slug}_{metric}')

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, part in summary.groupby('model'):
        ax.plot(part['dimension'], part['test_macro_f1'], marker='o', label=model)
    ax.set_xlabel('Embedding dimension')
    ax.set_ylabel('Test Macro-F1')
    ax.set_title('Embedding Width vs Test Macro-F1')
    ax.set_xticks(sorted(summary['dimension'].unique()))
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, out, 'dimension_reduction')
    print(f'Generated 11 charts (PNG + PDF) in {out}')


if __name__ == '__main__':
    main()
