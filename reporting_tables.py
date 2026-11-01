"""reporting_tables.py

Reporting: the LaTeX and CSV table writers.

Every table is written twice, as <name>.tex and <name>.csv with the same rows, into LATEX_DIR, the 'Tables <notebook>' folder
set by configure. The writers (df_to_latex, write_raw_table, lb_diagnostics_table, write_table) never display anything;
show_table displays a table once and writes it. The LaTeX files need \\usepackage{booktabs} and \\usepackage[table]{xcolor}
(the [table] option makes \\cellcolor work) in the preamble.
"""
import os
import re

import numpy as np
import pandas as pd
from IPython.display import Markdown, display


# ── LaTeX and CSV writers ───────────────────────────────────────────────────────────────────────────
LATEX_DIR = os.path.join(os.path.dirname(__file__), 'LatexTables')

def configure(window_label: str):
    """Point LATEX_DIR at the 'Tables <window_label>' folder of a notebook and create it.

    The folder gets the suffix ' SMOKE' when THESIS_SMOKE is '1', so test output never mixes with real output.

    Parameters:
    window_label (str): Label identifying the calling notebook, used to build the folder name.
    """
    global LATEX_DIR
    suffix = ' SMOKE' if os.environ.get('THESIS_SMOKE', '0') == '1' else ''     # smoke output never mixes with real output
    LATEX_DIR = os.path.join(os.path.dirname(__file__), f'Tables {window_label}{suffix}')
    os.makedirs(LATEX_DIR, exist_ok=True)


def lb_diagnostics_table(diag_df, filename='lb_diagnostics_full.tex'):
    """Save the Ljung-Box diagnostics table as LaTeX and CSV.

    The p-values are shaded in three tiers (red below 0.01, orange from 0.01 to 0.05, green above 0.05) with \\cellcolor. The tiers
    are a visual aid; the 0.05 level is the threshold of every model-selection decision. The same wrapper as df_to_latex is
    used (booktabs, [htbp], caption above, \\small). The table is not displayed.

    Parameters:
    diag_df (pd.DataFrame): Columns 'Variable', 'Lag' and one or more columns starting with 'LB(' that hold p-values.
    filename (str): Output file name relative to LATEX_DIR. Default 'lb_diagnostics_full.tex'.

    Returns:
    str: The LaTeX table (also written to LATEX_DIR/filename, with the CSV of the same name).
    """
    pval_cols = [c for c in diag_df.columns if c.startswith('LB(')]
    what = {'LB(z) final': 'the final (AR + GARCH) residuals $z$', 'LB(z^2) final': 'their squares $z^2$'}
    described = ' and '.join(what.get(c, c) for c in pval_cols)
    caption = (rf"Ljung--Box test for remaining serial correlation: $p$-values for {described} of wind speed and precipitation at lags 8, 16 and 24 "
               r"(three-hourly steps). Under white noise the $p$-values are uniform; small values mean remaining autocorrelation. "
               r"Green: $p>0.05$, orange: $0.01<p\leq0.05$, red: $p\leq0.01$. The shading is a visual aid only; the 0.05 level is the "
               r"threshold used for every model-selection decision.")
    head = " & ".join(["Variable", "Lag"] + [{'LB(z) final': r'$p$-value, $z$', 'LB(z^2) final': r'$p$-value, $z^2$'}.get(c, _esc_latex(c)) for c in pval_cols]) + r" \\"
    body_rows = []
    for _, row in diag_df.iterrows():
        cells = [str(row['Variable']), str(row['Lag'])]
        for c in pval_cols:
            v = row[c]
            shade = 'red!20' if v < 0.01 else ('orange!25' if v < 0.05 else 'green!20')
            cells.append(rf"\cellcolor{{{shade}}}{v:.4f}")
        body_rows.append(" & ".join(cells) + r" \\")
    latex_str = _table_wrapper(rf"\caption{{{caption}}}", "tab:lb_diagnostics_full", "ll" + "r" * len(pval_cols), head,
                               "\n".join(body_rows)).rstrip("\n")
    with open(os.path.join(LATEX_DIR, filename), 'w') as f:
        f.write(latex_str)
    diag_df.to_csv(os.path.join(LATEX_DIR, os.path.splitext(filename)[0] + '.csv'), index=False)
    return latex_str


_LATEX_SPECIAL = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#', '$': r'\$',
                  '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
                  '\u00b0': r'$^{\circ}$'}


def _esc_latex(s):
    """LaTeX escaping of model names and cell text.

    Escapes the ten LaTeX special characters (backslash, & % _ # $ { } ~ ^) in one pass, and writes the degree sign of
    'Gumbel (180°)' as $^{\\circ}$, so the table compiles without relying on the input encoding.

    Parameters:
    s (object): Value to escape; it is converted to str first.

    Returns:
    str: LaTeX-safe text.
    """
    return ''.join(_LATEX_SPECIAL.get(ch, ch) for ch in str(s))


# ── fold-design tables (four-fold design): generic and MCS-specific writers ────────────────────────
def _fmt_cell(v, digits=4):
    """Format one table cell: yes/no for booleans, integers as such, floats with `digits` decimals ('--' if not finite), text escaped.

    Parameters:
    v (object): Cell value.
    digits (int): Decimals of a float. Default 4.

    Returns:
    str: The LaTeX-ready text.
    """
    if isinstance(v, (bool, np.bool_)):
        return 'yes' if v else 'no'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return '--' if not np.isfinite(v) else f'{v:.{digits}f}'
    return _esc_latex(v)


# Cell shading shared with the Ljung-Box table: green for "yes", red for "no". Needs \usepackage[table]{xcolor}.
_YES_SHADE, _NO_SHADE = 'green!20', 'red!20'


def _table_wrapper(cap, label, fmt, head, body, size=r'\small'):
    """The booktabs table environment shared by df_to_latex and lb_diagnostics_table.

    Parameters:
    cap (str): The complete \\caption command.
    label (str): LaTeX label.
    fmt (str): LaTeX column format.
    head (str): Header row, ending in the row separator.
    body (str): Body rows, one per line, each ending in the row separator.
    size (str): Size command written after \\centering. Default \\small.

    Returns:
    str: The LaTeX text, ending in a newline.
    """
    return (r'\begin{table}[htbp]\centering' + size + '\n'
            + cap + rf'\label{{{label}}}' '\n' rf'\begin{{tabular}}{{{fmt}}}' '\n'
            r'\toprule' '\n' + head + '\n' r'\midrule' '\n' + body + '\n' r'\bottomrule' '\n'
            r'\end{tabular}\end{table}' '\n')


_FIT_OPEN = r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%'
_TAB_RE = re.compile(r'(\\begin\{tabular\}.*?\\end\{tabular\})', re.S)


def fit_table_text(tex):
    """Make every tabular inside a table environment shrink to the text width when it is wider, and never enlarge it.

    The tabular is wrapped in graphicx's resizebox with the width min(natural width, text width), so a table that fits is left
    untouched and a wider one is scaled down uniformly (text, rules and column gaps together). Longtables cannot be scaled and
    are left alone; tables already wrapped are not wrapped twice.

    Parameters:
    tex (str): LaTeX text of one table file.

    Returns:
    str: The LaTeX text with the wrapper.
    """
    if r'\begin{longtable}' in tex or _FIT_OPEN in tex or r'\begin{table}' not in tex:
        return tex
    return _TAB_RE.sub(lambda m: _FIT_OPEN + '\n' + m.group(1) + '}', tex)


def fit_all_tables(latex_dir=None):
    """Apply fit_table_text to every .tex file of the table folder (idempotent). Call it after a notebook has written its tables.

    Parameters:
    latex_dir (str or None): Table folder. Default None, which uses LATEX_DIR.

    Returns:
    list: Names of the files that were changed.
    """
    d = latex_dir or LATEX_DIR
    changed = []
    for name in sorted(os.listdir(d)):
        if not name.endswith('.tex'):
            continue
        path = os.path.join(d, name)
        with open(path) as f:
            old = f.read()
        new = fit_table_text(old)
        if new != old:
            with open(path, 'w') as f:
                f.write(new)
            changed.append(name)
    return changed


def df_to_latex(df, filename, caption, label, longtable=False, digits=4, index=True,
                col_format=None, small=True, verbose=False, headers=None, shade_bool=False,
                compact=False, short_caption=None, raw_text=False, csv_data=None):
    """Write a DataFrame as a booktabs table, or as a longtable for the 76-row appendix tables.

    The index is written as the first column when index=True (named by the index name, or blank). The LaTeX string is saved to
    LATEX_DIR/filename and the DataFrame, with the same rows, to the CSV file of the same stem. A '%' in the caption is escaped, because an unescaped one comments out the closing brace.

    Parameters:
    df (pd.DataFrame): Table.
    filename (str): File name in LATEX_DIR.
    caption (str): Caption (LaTeX text).
    label (str): LaTeX label.
    longtable (bool): Write a longtable instead of a table. Default False.
    digits (int): Decimals of float cells. Default 4.
    index (bool): Write the index as the first column. Default True.
    col_format (str or None): LaTeX column format. Default None, which is 'l' followed by 'r' for each other column.
    small (bool): Use \\small text. Default True.
    verbose (bool): Print the LaTeX text. Default False.
    headers (list or None): Column headers written as raw LaTeX (not escaped), one per column including the index. Default None.
    shade_bool (bool): Shade boolean cells green ('yes') or red ('no'), the colours of the Ljung-Box table. Default False.
    compact (bool): Footnotesize text and a tighter column separator, for wide tables. Default False.
    short_caption (str or None): Entry for the list of tables. Default None.
    raw_text (bool): Write string cells as given (already LaTeX, for example '0.26 [0.22, 0.29]') instead of escaping them.
        Default False.
    csv_data (pd.DataFrame or None): Numbers written to the CSV file instead of df, for a table whose cells are formatted text.
        Default None, which writes df.

    Returns:
    str: The LaTeX text.

    Raises:
    ValueError: If the number of headers differs from the number of columns.
    """
    d = df.reset_index() if index else df.copy()
    cols = [_esc_latex(c) for c in d.columns] if headers is None else list(headers)
    if len(cols) != d.shape[1]:
        raise ValueError(f'{len(cols)} headers for {d.shape[1]} columns')
    fmt = col_format or ('l' + 'r' * (len(cols) - 1))

    def cell(v):
        """One formatted cell of the body (raw text, or shaded boolean, or the standard formatting).

        Parameters:
        v (object): Cell value.

        Returns:
        str: The LaTeX text of the cell.
        """
        if raw_text and isinstance(v, str):
            return v
        txt = _fmt_cell(v, digits)
        if shade_bool and isinstance(v, (bool, np.bool_)):
            return rf'\cellcolor{{{_YES_SHADE if v else _NO_SHADE}}}{txt}'
        return txt

    body = '\n'.join(' & '.join(cell(v) for v in row) + r' \\' for row in d.itertuples(index=False))
    head = ' & '.join(cols) + r' \\'
    caption = re.sub(r'(?<!\\)%', r'\\%', caption)          # an unescaped % comments out the closing brace of the caption
    if short_caption: short_caption = re.sub(r'(?<!\\)%', r'\\%', short_caption)
    cap = rf'\caption[{short_caption}]{{{caption}}}' if short_caption else rf'\caption{{{caption}}}'
    if longtable:
        out = (rf'\begin{{longtable}}{{{fmt}}}' '\n' + cap + rf'\label{{{label}}}\\' '\n'
               r'\toprule' '\n' + head + '\n' r'\midrule\endfirsthead' '\n' r'\toprule' '\n' + head + '\n'
               r'\midrule\endhead' '\n' r'\bottomrule\endfoot' '\n' + body + '\n' r'\end{longtable}' '\n')
        if compact:
            out = r'\begingroup\footnotesize\setlength{\tabcolsep}{3pt}' '\n' + out + r'\endgroup' '\n'
    else:
        size = '\\footnotesize\\setlength{\\tabcolsep}{3pt}' if compact else ('\\small' if small else '')
        out = fit_table_text(_table_wrapper(cap, label, fmt, head, body, size))
    with open(os.path.join(LATEX_DIR, filename), 'w') as f:
        f.write(out)
    (df if csv_data is None else csv_data).to_csv(os.path.join(LATEX_DIR, os.path.splitext(filename)[0] + '.csv'), index=index)
    if verbose:
        print(out)
    return out


def write_raw_table(filename, caption, label, fmt, head, body, data=None, size=r'\small', short_caption=None, longtable=False):
    """Write a table whose header and body are already LaTeX (multi-row headers, bold cells, signed numbers).

    df_to_latex covers tables with one header row; this writer takes the finished header and body text, so the tables
    with grouped column headings can be written in the same booktabs layout, and the tabular is fitted to the text width
    in the same way.

    Parameters:
    filename (str): File name in LATEX_DIR.
    caption (str): Caption (LaTeX text); an unescaped % is escaped.
    label (str): LaTeX label.
    fmt (str): LaTeX column format.
    head (str): Header lines, each ending in the row separator (and any \\cmidrule lines).
    body (str): Body lines, each ending in the row separator.
    data (pd.DataFrame or None): The numbers behind the table, written to the CSV file of the same stem. Default None.
    size (str): Size command after \\centering. Default \\small.
    short_caption (str or None): Entry for the list of tables. Default None.
    longtable (bool): Write a longtable (for tables that break across pages). Default False.

    Returns:
    str: The LaTeX text.
    """
    caption = re.sub(r'(?<!\\)%', r'\\%', caption)
    cap = rf'\caption[{short_caption}]{{{caption}}}' if short_caption else rf'\caption{{{caption}}}'
    if longtable:
        first_row = head.strip().split('\n')[-1]
        out = (r'\begingroup' + size + '\n' rf'\begin{{longtable}}{{{fmt}}}' '\n' + cap + rf'\label{{{label}}}\\' '\n'
               r'\toprule' '\n' + head + '\n' r'\midrule\endfirsthead' '\n' r'\toprule' '\n' + first_row + '\n'
               r'\midrule\endhead' '\n' r'\bottomrule\endfoot' '\n' + body + '\n' r'\end{longtable}' '\n' r'\endgroup' '\n')
    else:
        out = fit_table_text(_table_wrapper(cap, label, fmt, head, body, size))
    with open(os.path.join(LATEX_DIR, filename), 'w') as f:
        f.write(out)
    if data is not None:
        pd.DataFrame(data).to_csv(os.path.join(LATEX_DIR, os.path.splitext(filename)[0] + '.csv'), index=False)
    return out


def tex_p(p):
    """A p-value for a table: '$<0.0001$' below 0.0001, four decimals below 0.001, three decimals otherwise; '--' if missing.

    Parameters:
    p (float): p-value.

    Returns:
    str: LaTeX text.
    """
    if p is None or not np.isfinite(p):
        return '--'
    if p < 1e-4:
        return r'$<0.0001$'
    return f'{p:.4f}' if p < 1e-3 else f'{p:.3f}'


def tex_num(x, digits=2):
    """A number for a table with a typeset minus sign ('$-$0.01'); '--' if missing.

    Parameters:
    x (float): Number.
    digits (int): Decimals. Default 2.

    Returns:
    str: LaTeX text.
    """
    if x is None or not np.isfinite(x):
        return '--'
    t = f'{abs(x):.{digits}f}'
    return (r'$-$' + t) if x < 0 and float(t) != 0.0 else t


def tex_family(f):
    """Display name of a copula family key with the Student-t written as math.

    Parameters:
    f (str): Family key or display name.

    Returns:
    str: LaTeX text.
    """
    import copula_families as CF
    d = CF.FAMILY_DISPLAY_NAMES.get(CF.resolve(f).key, CF.resolve(f).display)
    return r'Student-$t$' if d in ('t', 'Student-t', 'Student') else _esc_latex(d)


def tex_bold_min(values, fmt):
    """Format a list of numbers with the lowest one in bold (ties all bold; missing values as '--').

    Parameters:
    values (list): Numbers.
    fmt (callable): Number-to-text function.

    Returns:
    list: Cell texts.
    """
    v = np.asarray(values, float)
    best = np.nanmin(v) if np.isfinite(v).any() else None
    return ['--' if not np.isfinite(x) else (r'\textbf{' + fmt(x) + '}' if x == best else fmt(x)) for x in v]


def glrt_primary_table(glrt, filename='glrt_primary.tex', label='tab:glrt_primary', holm_label='tab:glrt_holm'):
    """Table of the primary-family GLRT of each outer fold (statistic, bandwidth, asymptotic and bootstrap p-values).

    Parameters:
    glrt (pd.DataFrame): Output of acar_glrt.run_glrt_real for all folds with a 'fold' column.
    filename (str): File name in LATEX_DIR. Default 'glrt_primary.tex'.
    label (str): LaTeX label of this table.
    holm_label (str): LaTeX label of the Holm table named in the caption.

    Returns:
    str: The LaTeX text.
    """
    prim = glrt[glrt.role == 'primary']
    B = int(prim['B'].iloc[0])
    head = (r'Fold & Family & $\lambda$ & $h$ & $|\mathcal X|/h$ & $p_{\text{asy}}$ & $p_{\text{boot}}$ \\')
    body = '\n'.join(f"{r.fold} & {tex_family(r.family)} & {r.lam:.2f} & {r.h:.2f} & {r.abs_X_over_h:.1f} & "
                     f"{tex_p(r.p_asymptotic_reference)} & {tex_p(r.p_param_boot)} " + r'\\' for r in prim.itertuples())
    cap = (r'Test of a constant calibration function for the primary family of each outer fold (training rows). '
           r'$\lambda$: GLRT statistic; $h$: cross-validated bandwidth; $|\mathcal X|/h$: covariate range over bandwidth '
           r'(the asymptotic reference requires at least 5). $p_{\text{asy}}$: asymptotic reference (not used); '
           rf'$p_{{\text{{boot}}}}$: calibrated $p$-value from the parametric bootstrap ({B} replicates), which ignores the '
           r'serial dependence of the data. Holm-adjusted values and the conclusions at 5\% and 10\% are in '
           rf'Table~\ref{{{holm_label}}}.')
    return write_raw_table(filename, cap, label, 'llrrrrr', head, body, data=prim)


def glrt_holm_table(table, B, filename='glrt_primary_holm.tex', label='tab:glrt_holm'):
    """Table of the Holm-adjusted bootstrap p-values of the primary-family GLRT across the four outer folds.

    Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65-70.

    Parameters:
    table (pd.DataFrame): Indexed by fold, with the columns 'Raw p', 'Holm-adjusted p', 'Reject at 5%' and 'Reject at 10%'.
    B (int): Number of bootstrap replicates behind the raw p-values, named in the caption.
    filename (str): File name in LATEX_DIR. Default 'glrt_primary_holm.tex'.
    label (str): LaTeX label. Default 'tab:glrt_holm'.

    Returns:
    str: The LaTeX text.
    """
    head = r'Fold & Raw $p$ & Holm-adjusted $p$ & Reject at 5\% & Reject at 10\% \\'
    body = '\n'.join(f"{f} & {r['Raw p']:.3f} & {r['Holm-adjusted p']:.3f} & {r['Reject at 5%']} & {r['Reject at 10%']} " + r'\\'
                     for f, r in table.iterrows())
    cap = ("Holm-adjusted $p$-values of the test of a constant calibration function across the four outer folds "
           f"(primary family of each fold, parametric bootstrap with {B} replicates), with the decision at the 5\\% and 10\\% level. "
           "The Holm adjustment is valid under any dependence between the tests, so it holds although the folds overlap in time.")
    return write_raw_table(filename, cap, label, 'lrrll', head, body, data=table.reset_index(names='fold'))


def tex_wd(w):
    """A weight decay as LaTeX math text: 0 or 10^{-k}.

    Parameters:
    w (float): Weight decay.

    Returns:
    str: For example '$0$' or '$10^{-3}$'.
    """
    if w == 0:
        return '$0$'
    return rf'$10^{{{int(round(np.log10(w)))}}}$'


def wd_used_sentence(wd_codine, wd_affine, wd_spline):
    """Sentence naming the three weight decays used in the study.

    Parameters:
    wd_codine (float): Weight decay of the C-CODINE networks.
    wd_affine (float): Weight decay of the C-CopFlow networks with affine coupling.
    wd_spline (float): Weight decay of the C-CopFlow networks with spline coupling.

    Returns:
    str: The sentence.
    """
    return (rf'The values used in the study, chosen by the synthetic calibration, are {tex_wd(wd_codine)} for C-CODINE, '
            rf'{tex_wd(wd_affine)} for C-CopFlow with affine coupling and {tex_wd(wd_spline)} for C-CopFlow with spline coupling.')


def wd_sensitivity_table(wd, filename='wd_sensitivity.tex', label='tab:wd_sensitivity', used=None):
    """Table of the weight-decay sensitivity check (lowest validation CVPE by network and weight decay, lowest per row in bold).

    Parameters:
    wd (pd.DataFrame): Second output of fold_pipeline.select_wd (columns model, wd, best_val_cvpe).
    filename (str): File name in LATEX_DIR. Default 'wd_sensitivity.tex'.
    label (str): LaTeX label.
    used (str or None): Sentence naming the values used in the study (see wd_used_sentence). Default None leaves it out.

    Returns:
    str: The LaTeX text.
    """
    piv = wd.pivot_table(index='model', columns='wd', values='best_val_cvpe', aggfunc='first', sort=False)
    grid = sorted(piv.columns)
    head = (rf'& \multicolumn{{{len(grid)}}}{{c}}{{Weight decay}} \\' '\n'
            rf'\cmidrule(lr){{2-{len(grid) + 1}}}' '\n'
            'Model & ' + ' & '.join(tex_wd(w) for w in grid) + r' \\')
    rows = []
    for m, r in piv.iterrows():
        cells = tex_bold_min([r[w] for w in grid], lambda x: f'{x:.4f}' if np.isfinite(x) else '--')
        rows.append(_esc_latex(m) + ' & ' + ' & '.join(cells) + r' \\')
    cap = (r'Weight-decay sensitivity: lowest validation CVPE reached during training on the training half of fold A1, one seed, '
           r'by weight decay (lowest per row in bold). ' + ((used + ' ') if used else '') +
           r'A robustness diagnostic only: it does not select the values used.')
    return write_raw_table(filename, cap, label, 'l' + 'r' * len(grid), head, '\n'.join(rows), data=piv.reset_index())


MCS_CAPTION_ALPHA = (r'Inclusion is reported at $\alpha=0.10$ and $\alpha=0.05$, which are the levels the '
                     r'conclusions rest on. $\alpha=0.25$ is a sensitivity level only: the MCS procedure '
                     r'eliminates a model when its $p$-value falls below $\alpha$, so a larger $\alpha$ '
                     r'removes more models on weaker evidence and is read as weak evidence, not as a '
                     r'stricter test.')


def mcs_fold_design_table(main, block, filename='mcs_main_table.tex', longtable=True, top=None,
                          label='tab:mcs_main', ci=None):
    """The pooled Model Confidence Set table of the four-fold design: CVPE, p-value and inclusion at the three levels alpha.

    The group column is left out, because every model name starts with its group. The model column wraps, the three inclusion
    columns are narrow with two-line headers, and inclusion is shaded green (in the set) or red (eliminated), as in the
    Ljung-Box table.

    Parameters:
    main (pd.DataFrame): The main_table of notebook 05, indexed by model, already sorted by CVPE, with the columns group, CVPE,
        p-value, 'in MCS (alpha=0.10)', 'in MCS (alpha=0.05)' and 'in MCS (alpha=0.25, sensitivity)'.
    block (int): Mean block length of the stationary bootstrap, named in the caption.
    filename (str): File name in LATEX_DIR. Default 'mcs_main_table.tex'.
    longtable (bool): Write a longtable. Default True.
    top (int or None): Keep only the first `top` rows. Default None, which keeps all.
    label (str): LaTeX label. Default 'tab:mcs_main'.
    ci (pd.DataFrame or None): Stationary-bootstrap intervals of the CVPE, indexed by model, with the columns ci_lo and ci_hi. When
        given, a column '95% interval' follows the CVPE and the CSV file holds ci_lo and ci_hi as numbers. Default None.

    Returns:
    str: The LaTeX text.
    """
    t = main.copy() if top is None else main.head(top).copy()
    t.index.name = 'model'
    keep = ['CVPE', 'p-value', 'in MCS (alpha=0.10)', 'in MCS (alpha=0.05)', 'in MCS (alpha=0.25, sensitivity)']
    t = t[keep]
    if ci is not None:
        iv = ci.reindex(t.index)
        data = t.join(iv[['ci_lo', 'ci_hi']])
        t.insert(1, '95% interval', [f'[{lo:.5f}, {hi:.5f}]' for lo, hi in zip(iv.ci_lo, iv.ci_hi)])
        t['p-value'] = t['p-value'].map(lambda p: f'{p:.4f}')          # the CVPE has five decimals, the p-value four
        n_out = len(main) - len(t)
        rest = (f' (the other {n_out} are out at every level)'
                if top is not None and not main.iloc[top:][keep[2:]].astype(bool).to_numpy().any() else '')
        cap = (rf'Model Confidence Set on the pooled per-step losses for the {len(t)} admissible models with the lowest CVPE{rest}; '
               rf'loss averaged over the four blocked folds, and over three seeds for neural models. Intervals are 95\% stationary-bootstrap '
               rf"intervals of a single model's CVPE (block length {block}). " + MCS_CAPTION_ALPHA + r' Green: model in the set; red: model eliminated.')
        centred = r'>{\centering\arraybackslash}p{1.0cm}'
        return df_to_latex(t, filename, cap, label, longtable=longtable, csv_data=data, digits=5,
                           col_format=r'>{\raggedright\arraybackslash}p{0.38\textwidth}rcr' + centred * 3,
                           headers=['model', 'CVPE', r'95\% interval', r'$p$-value', r'MCS $\alpha{=}0.10$', r'MCS $\alpha{=}0.05$',
                                    r'MCS $\alpha{=}0.25$'], shade_bool=True, compact=True,
                           short_caption=f'Model Confidence Set, {len(t)} lowest CVPE')
    headers = ['model', 'CVPE', r'$p$-value', r'MCS $\alpha{=}0.10$', r'MCS $\alpha{=}0.05$', r'MCS $\alpha{=}0.25$ (sens.)']
    cap = (rf'Model Confidence Set on the pooled per-step losses (loss averaged over the four blocked folds, '
           rf'over three seeds for neural models), stationary-bootstrap with mean block length {block}. ' + MCS_CAPTION_ALPHA +
           r' Green: model in the set; red: model eliminated.')
    centred = r'>{\centering\arraybackslash}p{1.1cm}'
    col_format = r'>{\raggedright\arraybackslash}p{0.42\textwidth}rr' + centred * 3
    return df_to_latex(t, filename, cap, label, longtable=longtable, col_format=col_format, headers=headers,
                       shade_bool=True, compact=True, short_caption='Model Confidence Set, admissible candidates')


def mcs_group_summary_table(main, gate_table, filename='mcs_group_summary.tex', label='tab:mcs_group_summary'):
    """Main-text summary per model group: candidates, models in the pool after the gates, best model by CVPE and MCS survivors.

    Parameters:
    main (pd.DataFrame): The main_table of notebook 05 (columns group, CVPE and the three inclusion columns).
    gate_table (pd.DataFrame): Gate table with the columns group and in_pool, one row per candidate.
    filename (str): File name in LATEX_DIR. Default 'mcs_group_summary.tex'.
    label (str): LaTeX label. Default 'tab:mcs_group_summary'.

    Returns:
    str: The LaTeX text.
    """
    a10, a05, a25 = 'in MCS (alpha=0.10)', 'in MCS (alpha=0.05)', 'in MCS (alpha=0.25, sensitivity)'
    rows = []
    for g, sub in gate_table.groupby('group'):
        m = main[main.group == g]
        best = m.CVPE.idxmin() if len(m) else None
        rows.append({'group': g, 'candidates': len(sub), 'in pool': int(sub.in_pool.sum()),
                     'best model': best if best else '--', 'best CVPE': float(m.CVPE.min()) if len(m) else float('nan'),
                     r'MCS 0.10': int(m[a10].sum()) if len(m) else 0, r'MCS 0.05': int(m[a05].sum()) if len(m) else 0,
                     r'MCS 0.25': int(m[a25].sum()) if len(m) else 0})
    t = pd.DataFrame(rows).set_index('group')
    cap = (r'Candidates, gate survivors, best model by CVPE and MCS survivors per model group. ' + MCS_CAPTION_ALPHA)
    return df_to_latex(t, filename, cap, label, col_format='llrp{4.2cm}rrrr')


# ── display-only status tables of the notebooks ──────────────────────────────────────────────────────


# ── generic helpers ────────────────────────────────────────────────────────────────────────────────
def _show(obj, title=None, legend=None, rows=None):
    """Display an object in the notebook with an optional bold title, row limit and legend.

    Parameters:
    obj (object): DataFrame or anything display() accepts.
    title (str or None): Title shown above. Default None.
    legend (dict or None): Column or term -> meaning, printed below. Default None.
    rows (int or None): Show only the first `rows` rows of a DataFrame and say how many are left out. Default None.
    """
    if title:
        display(Markdown(f"**{title}**"))
    if isinstance(obj, pd.DataFrame) and rows is not None and len(obj) > rows:
        extra = len(obj) - rows
        obj = obj.head(rows)
        display(obj)
        print(f"... {extra} more rows (full table: see the saved CSV / the variable)")
    else:
        display(obj)
    if legend:
        print("Legend: " + " | ".join(f"{k} = {v}" for k, v in legend.items()))

def write_table(df, name, caption=None, index=True, digits=4, label=None, **kwargs):
    """Write a table as LaTeX (name.tex) and CSV (name.csv) in LATEX_DIR, without displaying it.

    A Series becomes a one-column frame, MultiIndex columns are flattened, and tables of more than 40 rows are written as
    longtables. The LaTeX label is tab:<name> unless given.

    Parameters:
    df (pd.DataFrame or pd.Series): Table.
    name (str): File stem.
    caption (str or None): Caption. Default None, which uses the name.
    index (bool): Write the index. Default True.
    digits (int): Decimals written. Default 4.
    label (str or None): LaTeX label. Default None, which is tab:<name>.
    **kwargs: Further arguments of df_to_latex (col_format, headers, shade_bool, compact, short_caption, raw_text, longtable).

    Returns:
    pd.DataFrame: The (flattened) table.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame(df.name if df.name is not None else 'value')
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(str(c) for c in col if str(c) != '') for col in df.columns]
    os.makedirs(LATEX_DIR, exist_ok=True)
    kwargs.setdefault('longtable', len(df) > 40)
    df_to_latex(df, f'{name}.tex', caption or name.replace('_', ' ').capitalize() + '.', label or f'tab:{name}',
                digits=digits, index=index, **kwargs)
    return df


def show_table(df, name, caption=None, index=True, digits=4, title=None, rows=None, label=None, **kwargs):
    """Display a table once in the notebook and write it as LaTeX (name.tex) and CSV (name.csv) in LATEX_DIR.

    Parameters:
    df (pd.DataFrame or pd.Series): Table.
    name (str): File stem.
    caption (str or None): Caption. Default None, which uses the name.
    index (bool): Write the index. Default True.
    digits (int or None): Decimals shown and written. Default 4; None shows the values unrounded.
    title (str or None): Title shown above the table. Default None, which uses the name.
    rows (int or None): Show only the first `rows` rows on screen. Default None.
    label (str or None): LaTeX label. Default None, which is tab:<name>.
    **kwargs: Further arguments of df_to_latex, see write_table.

    Returns:
    pd.DataFrame: The (flattened) table.
    """
    out = write_table(df, name, caption=caption, index=index, digits=(4 if digits is None else digits), label=label, **kwargs)
    _show(out.round(digits) if digits is not None else out, title=(name.replace('_', ' ') if title is None else title), rows=rows)
    return out
