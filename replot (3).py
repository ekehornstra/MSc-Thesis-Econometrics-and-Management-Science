"""Redraw single figures from the stored inputs, without rerunning any notebook.

    python replot.py --list                  list the figures that can be redrawn
    python replot.py mcs_survivors           redraw every figure whose key contains this text
    python replot.py --all                   redraw everything
    python replot.py NAME --out SOME_FOLDER  write to another folder (default: the folder of the original run)

Set THESIS_SMOKE=1 or THESIS_ARTIFACT_DIR to point at other artifacts. Edit the drawing code in plotting_*.py,
figure_recipes.py, spatial_aggregation.py or synthetic_report.py, then run this; each figure takes seconds.
"""
import argparse
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('THESIS_ALLOW_UNCALIBRATED_IMPORT', '1')

import figure_replay  # noqa: E402


def main(argv=None):
    """Parse the command line and redraw the matching figures.

    Parameters:
    argv (list): Command-line arguments. Default None (sys.argv).

    Returns:
    int: 0 on success, 1 if a figure fails or nothing matches.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='*', help='text that must occur in the figure key')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    keys = figure_replay.list_figures()
    if a.list or (not a.names and not a.all):
        print('\n'.join(keys) if keys else 'no stored figure calls yet: run the notebooks first')
        return 0
    todo = keys if a.all else [k for k in keys if any(n in k for n in a.names)]
    if not todo:
        print(f"nothing matches {a.names}; available:\n" + '\n'.join(keys))
        return 1
    failed = []
    for k in todo:
        try:
            print(f"{k} -> {figure_replay.replay(k, images_dir=a.out)}")
        except Exception as exc:
            failed.append(k); print(f"{k} FAILED: {exc!r}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
