"""Write a backtest's artefacts: a self-contained HTML report plus CSV/JSON.

The HTML has no external dependencies. Charts are drawn in inline SVG by a
small script from data embedded in the page, with a crosshair tooltip, a
light/dark palette and a table view behind every chart.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd

from app.backtest.metrics import MIN_DAYS_FOR_RATIOS


def _fmt(v, kind: str = "num") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, bool):
        return "yes" if v else "no"
    if kind == "pct":
        return f"{v:+.2f}%"
    if kind == "usd":
        return f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"
    if kind == "int":
        return f"{int(v):,}"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return html.escape(str(v))


def _table(df: pd.DataFrame, *, index: bool = True, cls: str = "") -> str:
    if df is None or df.empty:
        return '<p class="muted">None.</p>'
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in ([df.index.name or ""] if index else []) + list(df.columns))
    body = []
    for idx, row in df.astype(object).iterrows():
        cells = [f"<th scope='row'>{_fmt(idx)}</th>"] if index else []
        for v in row:
            num = isinstance(v, (int, float)) and not isinstance(v, bool)
            cells.append(f"<td class='{'num' if num else ''}'>{_fmt(v)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='tbl {cls}'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _kv(rows: list[tuple[str, str]]) -> str:
    return "<dl class='kv'>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>"


def _series_payload(equity: pd.Series, benchmarks: dict[str, pd.Series], dd: pd.Series) -> dict:
    """Downsample to <= ~1500 points so the page stays light."""
    step = max(1, len(equity) // 1500)
    idx = equity.index[::step]
    if idx[-1] != equity.index[-1]:
        idx = idx.append(equity.index[-1:])
    return {
        "t": [t.strftime("%Y-%m-%d %H:%M") for t in idx],
        "series": [{"name": "Strategy", "v": [round(float(x), 2) for x in equity.reindex(idx)]}]
        + [
            {"name": f"{k} buy & hold", "v": [round(float(x), 2) for x in s.reindex(idx).ffill()]}
            for k, s in benchmarks.items()
        ],
        "dd": [round(100 * float(x), 3) for x in dd.reindex(idx)],
    }


def write_report(
    out_dir: Path,
    title: str,
    result,
    stats: dict,
    benchmarks: dict[str, pd.Series],
    params,
    notes: list[str],
    diagnostics: list[dict],
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades = result.trades
    s = stats["summary"]

    trades.to_csv(out_dir / "trades.csv", index=False)
    result.equity.rename("equity").to_csv(out_dir / "equity_curve.csv", header=True)
    json_safe = {
        k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in s.items()
    }
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary": json_safe,
                "benchmarks": stats["benchmarks"],
                "params": {k: str(v) for k, v in asdict(params).items()} if is_dataclass(params) else {},
                "sizing": {k: v for k, v in asdict(result.sizing).items()},
                "costs": {k: v for k, v in asdict(result.costs).items()},
            },
            indent=2,
            default=str,
        )
    )

    payload = _series_payload(result.equity, benchmarks, stats["drawdown_series"])
    r_payload = (
        [
            {"id": int(r.trade_id), "label": f"{r.symbol} {r.direction[0].upper()} {pd.Timestamp(r.entry_time):%m-%d %H:%M}", "r": float(r.r_multiple), "pnl": float(r.net_pnl)}
            for r in trades.itertuples()
        ]
        if not trades.empty
        else []
    )

    kpis = [
        ("Total return", _fmt(s["total_return_pct"], "pct")),
        ("Ending equity", _fmt(s["ending_equity"], "usd")),
        ("Max drawdown", _fmt(s["max_drawdown_pct"], "pct")),
        ("Sharpe", _fmt(s.get("sharpe"))),
        ("Trades", _fmt(s.get("n_trades", 0), "int")),
        ("Win rate", "—" if not s.get("n_trades") else f"{s['win_rate_pct']:.1f}%"),
        ("Profit factor", _fmt(s.get("profit_factor"))),
        ("Avg R", _fmt(s.get("avg_r"))),
    ]
    kpi_html = "".join(f"<div class='kpi'><div class='kpi-l'>{k}</div><div class='kpi-v'>{v}</div></div>" for k, v in kpis)

    perf = _kv(
        [
            ("Period", f"{_fmt(s['start'])} → {_fmt(s['end'])} ET"),
            ("Trading days", _fmt(s["n_days"], "int")),
            ("Starting capital", _fmt(s["starting_capital"], "usd")),
            ("Ending equity", _fmt(s["ending_equity"], "usd")),
            ("Net profit", _fmt(s["net_profit"], "usd")),
            ("Total return", _fmt(s["total_return_pct"], "pct")),
            ("CAGR", _fmt(s["cagr_pct"], "pct") if s["n_days"] >= 252 else "n/a — needs at least a year of data"),
            ("Annualised volatility", _fmt(s["annual_volatility_pct"], "pct")),
            ("Sharpe (daily, rf=0)", _fmt(s["sharpe"])),
            ("Sortino", _fmt(s["sortino"])),
            ("Calmar", _fmt(s["calmar"]) if s["n_days"] >= 252 else "n/a — needs at least a year of data"),
            ("Max drawdown (intraday, marked to market)", f"{_fmt(s['max_drawdown_pct'], 'pct')} ({_fmt(s['max_drawdown_dollars'], 'usd')})"),
            ("Max drawdown peak → trough", f"{_fmt(s['max_drawdown_peak'])} → {_fmt(s['max_drawdown_trough'])}"),
            ("Recovered", _fmt(s["max_drawdown_recovery"]) if s["max_drawdown_recovery"] is not None else "not recovered by end of data"),
            ("Max drawdown (end-of-day)", _fmt(s["max_drawdown_eod_pct"], "pct")),
            ("Ulcer index", _fmt(s["ulcer_index"])),
            ("Best / worst day", f"{_fmt(s['best_day_pct'], 'pct')} / {_fmt(s['worst_day_pct'], 'pct')}"),
            ("Positive days", "—" if s["positive_days_pct"] is None else f"{s['positive_days_pct']:.0f}%"),
            ("Time in market", f"{s['exposure_pct']:.1f}% of regular-session minutes"),
        ]
    )
    if s.get("n_trades"):
        tstats = _kv(
            [
                ("Trades (long / short)", f"{s['n_trades']} ({s['n_long']} / {s['n_short']})"),
                ("Win rate", f"{s['win_rate_pct']:.1f}%"),
                ("Profit factor", _fmt(s["profit_factor"])),
                ("Expectancy per trade", _fmt(s["expectancy_dollars"], "usd")),
                ("Average / median R", f"{_fmt(s['avg_r'])} / {_fmt(s['median_r'])}"),
                ("SQN", _fmt(s["sqn"])),
                ("Average win / loss", f"{_fmt(s['avg_win'], 'usd')} / {_fmt(s['avg_loss'], 'usd')}"),
                ("Payoff ratio", _fmt(s["payoff_ratio"])),
                ("Largest win / loss", f"{_fmt(s['largest_win'], 'usd')} / {_fmt(s['largest_loss'], 'usd')}"),
                ("Max consecutive wins / losses", f"{s['max_consecutive_wins']} / {s['max_consecutive_losses']}"),
                ("Average hold", f"{s['avg_hold_minutes']:.0f} min"),
                ("Average MFE / MAE", f"{_fmt(s['avg_mfe_r'])}R / {_fmt(s['avg_mae_r'])}R"),
                ("Kelly fraction", _fmt(s["kelly_pct"], "pct")),
                ("Gross P&amp;L", _fmt(s["gross_pnl"], "usd")),
                ("Commissions", _fmt(s["commissions"], "usd")),
                ("Slippage (included in fills)", _fmt(s["slippage"], "usd")),
            ]
        )
    else:
        tstats = "<p class='muted'>No trades were taken.</p>"

    bench_df = pd.DataFrame(
        {"Strategy": {"total_return_pct": s["total_return_pct"], "ending_equity": s["ending_equity"], "max_drawdown_pct": s["max_drawdown_pct"], "sharpe": s["sharpe"], "annual_volatility_pct": s["annual_volatility_pct"], "correlation": None, "beta": None}}
        | stats["benchmarks"]
    ).T
    bench_df.columns = ["Total return %", "Ending equity", "Max DD %", "Sharpe", "Ann. vol %", "Corr. to strategy", "Beta of strategy"]

    dd_rows = pd.DataFrame(
        [
            {"Peak": p.peak_time, "Trough": p.trough_time, "Recovered": p.recovery_time, "Depth %": 100 * p.depth_pct, "Depth $": p.depth_dollars}
            for p in stats["drawdowns"]
        ]
    )

    trade_cols = [
        "trade_id", "symbol", "date", "direction", "alert_time", "entry_time", "entry_fill", "stop_price",
        "shares", "notional", "risk_dollars", "exits", "exit_reason", "hold_minutes", "net_pnl",
        "r_multiple", "return_on_equity_pct", "mfe_r", "mae_r", "consol_start", "consol_bars",
        "in_location", "drive_ok", "volume_ratio", "market_aligned", "beyond_key_level",
    ]
    tdf = trades[trade_cols].copy() if not trades.empty else pd.DataFrame()
    if not tdf.empty:
        tdf["entry_time"] = pd.to_datetime(tdf["entry_time"]).dt.strftime("%H:%M")
        tdf.columns = [
            "#", "Symbol", "Date", "Side", "Alert", "Entry", "Entry fill", "Stop", "Shares", "Notional",
            "Risk $", "Exits (shares@fill reason time)", "Final exit", "Hold (min)", "Net P&L", "R",
            "% of equity", "MFE R", "MAE R", "Consol. from", "Consol. bars", "Upper/lower ⅓",
            "Drive", "Vol. ratio*", "Market aligned", "Beyond PMH/PDH",
        ]

    diag_df = pd.DataFrame(diagnostics)
    if not diag_df.empty and "alert_time" in diag_df:
        diag_df["alert_time"] = pd.to_datetime(diag_df["alert_time"]).dt.strftime("%Y-%m-%d %H:%M")

    param_rows = [(html.escape(k), html.escape(str(v))) for k, v in asdict(params).items()]
    param_rows += [(f"sizing.{k}", html.escape(str(v))) for k, v in asdict(result.sizing).items()]
    param_rows += [(f"costs.{k}", html.escape(str(v))) for k, v in asdict(result.costs).items()]

    warn = ""
    if stats["few_days"]:
        warn = (
            f"<div class='warn' role='note'><strong>Small sample.</strong> This run covers "
            f"{s['n_days']} trading day(s) and {s.get('n_trades', 0)} trade(s). Sharpe, Sortino, "
            f"CAGR and yearly figures need at least {MIN_DAYS_FOR_RATIOS} days (ideally hundreds of trades) "
            f"to mean anything; read them as arithmetic, not evidence.</div>"
        )

    skipped = _table(result.skipped, index=False) if not result.skipped.empty else "<p class='muted'>None.</p>"
    notes_html = "".join(f"<li>{n}</li>" for n in notes)

    page = _TEMPLATE.format(
        title=html.escape(title),
        warn=warn,
        kpis=kpi_html,
        perf=perf,
        tstats=tstats,
        bench=_table(bench_df.astype(object)),
        yearly=_table(stats["yearly"].rename_axis("Year")),
        monthly=_table(stats["monthly"].rename_axis("Month")),
        dd=_table(dd_rows, index=False),
        by_symbol=_table(stats["by_symbol"]),
        by_direction=_table(stats["by_direction"]),
        by_exit=_table(stats["by_exit"]),
        trades=_table(tdf, index=False, cls="wide"),
        skipped=skipped,
        diag=_table(diag_df, index=False),
        params=_kv(param_rows),
        notes=notes_html,
        data=json.dumps(payload),
        rdata=json.dumps(r_payload),
    )
    path = out_dir / "report.html"
    path.write_text(page)
    return path


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{color-scheme:light;--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--rule:#e4e3df;--grid:#efeeea;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--warnbg:#fff6e0;--warnink:#6b4a00}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;--bg:#1a1a19;--panel:#222221;--ink:#ffffff;--ink2:#c3c2b7;--muted:#9a998f;--rule:#3a3a37;--grid:#2c2c2a;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--warnbg:#3a2f12;--warnink:#f5d88a}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#1a1a19;--panel:#222221;--ink:#ffffff;--ink2:#c3c2b7;--muted:#9a998f;--rule:#3a3a37;--grid:#2c2c2a;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--warnbg:#3a2f12;--warnink:#f5d88a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:32px 0 10px;padding-top:8px;border-top:1px solid var(--rule)}}
.muted{{color:var(--muted)}}
.warn{{background:var(--warnbg);color:var(--warnink);padding:10px 12px;border-radius:6px;margin:14px 0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fill,minmax(136px,1fr));gap:10px;margin:16px 0}}
.kpi{{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:10px 12px}}
.kpi-l{{color:var(--ink2);font-size:12px}} .kpi-v{{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:24px}} @media (max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
.kv{{display:grid;grid-template-columns:auto 1fr;gap:4px 16px;margin:0}} .kv dt{{color:var(--ink2)}} .kv dd{{margin:0;font-variant-numeric:tabular-nums}}
.tbl{{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--panel)}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:5px 9px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}}
thead th{{color:var(--ink2);font-weight:600;font-size:12px;position:sticky;top:0;background:var(--panel)}}
td.num{{text-align:right}} tbody th{{font-weight:500}}
.wide td,.wide th{{font-size:12px}}
.chart{{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:10px;position:relative}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:0 0 6px}}
.sw{{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}}
.tip{{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;white-space:nowrap}}
svg text{{fill:var(--ink2);font-size:11px}}
details summary{{cursor:pointer;color:var(--ink2);margin:6px 0}}
ul.notes li{{margin:4px 0}}
</style></head>
<body><main>
<h1>{title}</h1>
<p class="muted">Starting capital $10,000 unless stated. All times US/Eastern. Generated by <code>scripts/run_backtest.py</code>.</p>
{warn}
<div class="kpis">{kpis}</div>

<h2>Equity curve</h2>
<div class="chart" id="eqc"><div class="legend" id="eqleg"></div><svg id="eq" width="100%" height="300" role="img" aria-label="Equity curve of the strategy against buy and hold"></svg><div class="tip" id="eqtip"></div></div>
<h2>Drawdown</h2>
<div class="chart" id="ddc"><svg id="dd" width="100%" height="160" role="img" aria-label="Drawdown from running equity peak, percent"></svg><div class="tip" id="ddtip"></div></div>
<h2>R-multiple per trade</h2>
<div class="chart" id="rc"><svg id="rr" width="100%" height="200" role="img" aria-label="Net R multiple of each trade"></svg><div class="tip" id="rtip"></div></div>

<div class="grid2">
<section><h2>Performance</h2>{perf}</section>
<section><h2>Trade statistics</h2>{tstats}</section>
</div>

<h2>Strategy vs buy &amp; hold (same window, same $10,000)</h2>{bench}
<div class="grid2">
<section><h2>Returns by year (%)</h2>{yearly}</section>
<section><h2>Returns by month (%)</h2>{monthly}</section>
</div>
<h2>Worst drawdowns</h2>{dd}
<div class="grid2">
<section><h2>By symbol</h2>{by_symbol}</section>
<section><h2>By direction</h2>{by_direction}<h2>By exit reason</h2>{by_exit}</section>
</div>
<h2>All trades</h2>{trades}
<p class="muted">*Vol. ratio = trigger-bar volume ÷ prior bar. Shown for the cheat sheet's "30% more volume" factor only; it is not known until the bar closes, so it never decides a trade.</p>
<h2>Signals not taken (position sizing / buying power)</h2>{skipped}
<h2>Signal log</h2><p class="muted">Every alert or symbol-day and whether it produced a signal. A signal becomes a trade unless it appears in the table above.</p>{diag}
<h2>Assumptions &amp; caveats</h2><ul class="notes">{notes}</ul>
<details><summary>All parameters</summary>{params}</details>
</main>
<script>
const D={data}, R={rdata};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COLORS=['--s1','--s2','--s3'];
function money(v){{return (v<0?'-$':'$')+Math.abs(v).toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}})}}
function niceTicks(lo,hi,n){{const span=hi-lo||1;const step=Math.pow(10,Math.floor(Math.log10(span/n)));const m=[1,2,2.5,5,10].find(k=>span/(k*step)<=n)*step;const out=[];for(let v=Math.ceil(lo/m)*m;v<=hi+1e-9;v+=m)out.push(+v.toFixed(6));return out}}
function sessionTicks(){{const out=[];let prev='';D.t.forEach((t,i)=>{{const d=t.slice(0,10);if(d!==prev){{out.push([i,d.slice(5)]);prev=d}}}});const k=Math.ceil(out.length/8);return out.filter((_,i)=>i%k===0)}}
function line(svgId,tipId,series,fmt,opts){{
  const svg=document.getElementById(svgId),tip=document.getElementById(tipId);
  const W=svg.clientWidth||900,H=+svg.getAttribute('height'),L=72,Rm=12,T=10,B=24;
  const all=series.flatMap(s=>s.v).filter(v=>v!=null);
  let lo=Math.min(...all),hi=Math.max(...all);if(opts.zeroTop){{hi=0}};if(lo===hi){{lo-=1;hi+=1}}
  const pad=(hi-lo)*0.06;lo-=opts.zeroTop?pad:pad;hi+=opts.zeroTop?0:pad;
  const n=D.t.length,x=i=>L+(W-L-Rm)*(n<2?0:i/(n-1)),y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  let g='';
  niceTicks(lo,hi,5).forEach(v=>{{g+=`<line x1="${{L}}" x2="${{W-Rm}}" y1="${{y(v)}}" y2="${{y(v)}}" stroke="${{css('--grid')}}"/><text x="${{L-6}}" y="${{y(v)+4}}" text-anchor="end">${{(opts.axisFmt||fmt)(v)}}</text>`}});
  sessionTicks().forEach(([i,l])=>{{g+=`<line x1="${{x(i)}}" x2="${{x(i)}}" y1="${{T}}" y2="${{H-B}}" stroke="${{css('--grid')}}"/><text x="${{x(i)+3}}" y="${{H-6}}">${{l}}</text>`}});
  series.forEach((s,k)=>{{const c=css(opts.colors?opts.colors[k]:COLORS[k]);const pts=s.v.map((v,i)=>`${{x(i).toFixed(1)}},${{y(v).toFixed(1)}}`).join(' ');
    if(opts.area)g+=`<polygon points="${{x(0)}},${{y(0)}} ${{pts}} ${{x(n-1)}},${{y(0)}}" fill="${{c}}" opacity=".18"/>`;
    g+=`<polyline points="${{pts}}" fill="none" stroke="${{c}}" stroke-width="2" stroke-linejoin="round"/>`}});
  g+=`<line id="${{svgId}}x" y1="${{T}}" y2="${{H-B}}" stroke="${{css('--muted')}}" stroke-dasharray="3 3" visibility="hidden"/>`;
  g+=`<rect x="${{L}}" y="${{T}}" width="${{W-L-Rm}}" height="${{H-T-B}}" fill="transparent" id="${{svgId}}hit"/>`;
  svg.innerHTML=g;
  const hit=document.getElementById(svgId+'hit'),cx=document.getElementById(svgId+'x');
  hit.addEventListener('mousemove',e=>{{const r=svg.getBoundingClientRect();const i=Math.max(0,Math.min(n-1,Math.round((e.clientX-r.left-L)/(W-L-Rm)*(n-1))));
    cx.setAttribute('x1',x(i));cx.setAttribute('x2',x(i));cx.setAttribute('visibility','visible');
    tip.innerHTML=`<div>${{D.t[i]}} ET</div>`+series.map((s,k)=>`<div><span class="sw" style="background:${{css(opts.colors?opts.colors[k]:COLORS[k])}}"></span>${{s.name}}: <b>${{fmt(s.v[i])}}</b></div>`).join('');
    tip.style.display='block';const tx=x(i)+14;tip.style.left=(tx+tip.offsetWidth>W?x(i)-tip.offsetWidth-8:tx)+'px';tip.style.top='18px'}});
  hit.addEventListener('mouseleave',()=>{{tip.style.display='none';cx.setAttribute('visibility','hidden')}});
}}
function bars(){{
  const svg=document.getElementById('rr'),tip=document.getElementById('rtip');
  if(!R.length){{svg.outerHTML='<p class="muted">No trades.</p>';return}}
  const W=svg.clientWidth||900,H=+svg.getAttribute('height'),L=48,Rm=12,T=10,B=10;
  let lo=Math.min(0,...R.map(r=>r.r)),hi=Math.max(0,...R.map(r=>r.r));const pad=(hi-lo)*.08||1;lo-=pad;hi+=pad;
  const y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo)),bw=Math.max(4,Math.min(36,(W-L-Rm)/R.length-2)),step=(W-L-Rm)/R.length;
  let g='';niceTicks(lo,hi,4).forEach(v=>{{g+=`<line x1="${{L}}" x2="${{W-Rm}}" y1="${{y(v)}}" y2="${{y(v)}}" stroke="${{css('--grid')}}"/><text x="${{L-6}}" y="${{y(v)+4}}" text-anchor="end">${{v}}R</text>`}});
  R.forEach((r,i)=>{{const x0=L+i*step+(step-bw)/2,y0=y(Math.max(r.r,0)),h=Math.abs(y(r.r)-y(0));
    g+=`<rect x="${{x0}}" y="${{y0}}" width="${{bw}}" height="${{Math.max(h,1)}}" rx="2" fill="${{css('--s1')}}" opacity="${{r.r>=0?1:.45}}" data-i="${{i}}"/>`}});
  g+=`<line x1="${{L}}" x2="${{W-Rm}}" y1="${{y(0)}}" y2="${{y(0)}}" stroke="${{css('--ink2')}}"/>`;
  svg.innerHTML=g;
  svg.querySelectorAll('rect').forEach(el=>{{el.addEventListener('mousemove',e=>{{const r=R[+el.dataset.i];const b=svg.getBoundingClientRect();
    tip.innerHTML=`<div>#${{r.id}} ${{r.label}}</div><div><b>${{r.r.toFixed(2)}}R</b> · ${{money(r.pnl)}}</div>`;tip.style.display='block';
    tip.style.left=Math.min(e.clientX-b.left+12,W-tip.offsetWidth-4)+'px';tip.style.top='12px'}});el.addEventListener('mouseleave',()=>tip.style.display='none')}});
}}
function draw(){{
  document.getElementById('eqleg').innerHTML=D.series.map((s,k)=>`<span><span class="sw" style="background:${{css(COLORS[k])}}"></span>${{s.name}}</span>`).join('');
  line('eq','eqtip',D.series,money,{{axisFmt:v=>'$'+Math.round(v).toLocaleString()}});
  line('dd','ddtip',[{{name:'Drawdown',v:D.dd}}],v=>v.toFixed(2)+'%',{{zeroTop:true,area:true}});
  bars();
}}
draw();addEventListener('resize',draw);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
</script>
</body></html>
"""
