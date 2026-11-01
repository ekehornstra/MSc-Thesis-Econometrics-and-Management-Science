"""Save the inputs of every figure so that one figure can be redrawn in seconds, without rerunning a notebook.

A drawing function decorated with `recorded` stores its call (function, arguments, output folder) as the artifact
`figcall_<figure>` each time it runs. `replay` (or `python replot.py <figure>`) loads that call and runs the drawing function
again. The drawing functions only draw: everything slow (fits, bootstraps) happens before the call and is part of the
arguments, so a replay takes seconds.

Figures that draw on a matplotlib Axes supplied by the caller are not recorded (an Axes cannot be stored); their callers
wrap the whole figure in a drawing function of their own (see figure_recipes.py).
"""
import functools
import importlib
import inspect
import re
import warnings

PREFIX = 'figcall_'
_NAME_PARAMS = ('filename', 'fname', 'path', 'filename_stem', 'var_name', 'name', 'label', 'tag')


def _is_mpl_object(x):
    """True for matplotlib Figure and Axes objects, which cannot be stored.

    Parameters:
    x (object): Any object.

    Returns:
    bool: Whether x is a matplotlib Figure, Axes, GridSpec or SubFigure.
    """
    mod = type(x).__module__
    return mod.startswith('matplotlib') and type(x).__name__ in ('Axes', 'AxesSubplot', 'Figure', 'GridSpec', 'SubFigure')


def _safe(text):
    """Make a string usable as part of an artifact name.

    Parameters:
    text (object): Text to convert.

    Returns:
    str: The text with every run of other characters replaced by an underscore.
    """
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text)).strip('_.')


def figure_key(func, args, kwargs):
    """Artifact key of one call: the output file name of the call, or else the function name plus its name-like arguments.

    Parameters:
    func (callable): The drawing function.
    args (tuple): Positional arguments of the call.
    kwargs (dict): Keyword arguments of the call.

    Returns:
    str: Key without prefix: the output file name without extension (for example 'mcs_survivors'), or the function name plus
        its name-like arguments when the call has no file name argument.
    """
    parts, by_file = [], False
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        vals = bound.arguments
        for p in _NAME_PARAMS:
            v = vals.get(p)
            if isinstance(v, str) and v:
                if p in ('filename', 'fname', 'path'):
                    v = re.sub(r'\.(png|pdf)$', '', v.replace('\\', '/').split('/')[-1])
                    parts, by_file = [v], True
                    break
                parts.append(v)
    except (TypeError, ValueError):
        pass
    return _safe('__'.join(parts if by_file else [func.__name__] + parts))


def recorded(func):
    """Decorator: store every call of a drawing function as an artifact, then run it.

    Parameters:
    func (callable): A module-level function that draws a figure from its arguments alone.

    Returns:
    callable: The wrapped function. It has the attribute `.raw` (the undecorated function) used by `replay`.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Record the call, then draw the figure.

        Parameters:
        *args: Positional arguments of the drawing function.
        **kwargs: Keyword arguments of the drawing function.

        Returns:
        The return value of the drawing function.
        """
        try:
            _save_call(func, args, kwargs)
        except Exception as exc:                       # recording must never stop a figure from being drawn
            warnings.warn(f"could not record the call of {func.__name__}: {exc!r}", RuntimeWarning, stacklevel=2)
        return func(*args, **kwargs)
    wrapper.raw = func
    wrapper.is_recorded = True
    return wrapper


def _save_call(func, args, kwargs):
    """Store the call as artifact figcall_<key>; skipped when an argument is a matplotlib Axes or Figure.

    Parameters:
    func (callable): The recorded drawing function.
    args (tuple): Positional arguments of the call.
    kwargs (dict): Keyword arguments of the call.
    """
    if any(_is_mpl_object(a) for a in list(args) + list(kwargs.values())):
        return
    import plotting_core
    import run_support
    key = figure_key(func, args, kwargs)
    run_support.save_artifact(PREFIX + key, call=dict(module=func.__module__, func=func.__name__, args=tuple(args),
                                                      kwargs=dict(kwargs), images_dir=plotting_core.IMAGES_DIR))


def list_figures():
    """Keys of all stored figure calls.

    Returns:
    list: Sorted figure keys (without the prefix).
    """
    import run_support
    return sorted(p.name[len(PREFIX):-len('_objects.pkl')] for p in run_support.ARTIFACT_DIR.glob(PREFIX + '*_objects.pkl'))


def replay(key, images_dir=None, show=False):
    """Redraw one figure from its stored call.

    Parameters:
    key (str): Figure key as listed by list_figures().
    images_dir (str or None): Output folder. Default None (the folder used when the figure was first drawn).
    show (bool): Call plt.show() afterwards. Default False.

    Returns:
    str: The folder the figure was written to.

    Raises:
    FileNotFoundError: If no call is stored under this key.
    """
    import matplotlib.pyplot as plt
    import plotting_core
    import run_support
    art = run_support.load_artifact(PREFIX + key)
    if 'call' not in art:
        raise FileNotFoundError(f"no stored call for {key!r}; available: {list_figures()}")
    c = art['call']
    out = images_dir or c['images_dir']
    import os
    os.makedirs(out, exist_ok=True)
    plotting_core.IMAGES_DIR = out
    func = getattr(importlib.import_module(c['module']), c['func'])
    func = getattr(func, 'raw', func)
    func(*c['args'], **c['kwargs'])
    if show:
        plt.show()
    plt.close('all')
    return out
