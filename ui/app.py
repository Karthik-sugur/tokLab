"""TokLab Streamlit UI — displays stored API results only; never computes metrics."""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from api_client import ApiUnavailable, TokLabClient, readable_http_error
from token_highlight import render_token_highlight_html

st.set_page_config(page_title="TokLab", layout="wide")

COLORS = {
    "C2": "#1f77b4",
    "C3": "#ff7f0e",
    "C4": "#2ca02c",
    "C5": "#d62728",
    "C1": "#9467bd",
    "C6": "#8c564b",
}


def client() -> TokLabClient:
    base = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")
    return TokLabClient(base)


def stop_on_unavailable(exc: ApiUnavailable) -> None:
    """Friendly API-down message; halt cleanly (no traceback in the browser)."""
    st.error(str(exc))
    st.stop()


def stop_on_http_error(exc: httpx.HTTPStatusError) -> None:
    """Show status code + API body on every page."""
    st.error(readable_http_error(exc))
    st.stop()


def fidelity_map(cli: TokLabClient, exp_id: str) -> dict[str, dict[str, Any]]:
    try:
        data = cli.fidelity(exp_id)
    except ApiUnavailable as exc:
        stop_on_unavailable(exc)
    except httpx.HTTPStatusError as exc:
        # Missing fidelity table → treat as no badges (not a hard stop).
        if exc.response.status_code == 409:
            return {}
        stop_on_http_error(exc)
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("rows") or []:
        cond = str(row.get("condition"))
        valid = bool(row.get("valid", True))
        status = str(row.get("status", "OK" if valid else "INVALID"))
        reason = status if valid else f"{status}: round-trip failures"
        out[cond] = {"valid": valid, "reason": reason, "row": row}
    return out


def grey_note(cond: str, fmap: dict[str, dict[str, Any]]) -> str | None:
    info = fmap.get(cond)
    if info is None:
        return None
    if not info["valid"]:
        return f"{cond} greyed out — {info['reason']}"
    return None


def page_preview(cli: TokLabClient) -> None:
    st.header("1 · Preview")
    text = st.text_area(
        "Snippet",
        value="def AcmeWidgetFactory(x):\n    return x\n",
        height=160,
    )
    conditions = st.multiselect(
        "Conditions",
        ["C2", "C3", "C4", "C5"],
        default=["C2", "C4", "C5"],
    )
    if st.button("Tokenize", type="primary"):
        try:
            data = cli.preview(text, conditions)
        except ApiUnavailable as exc:
            stop_on_unavailable(exc)
        except httpx.HTTPStatusError as exc:
            stop_on_http_error(exc)
        for block in data.get("conditions") or []:
            cond = block["condition"]
            st.subheader(f"{cond} — {block['n_tokens']} tokens")
            color = COLORS.get(cond, "#333")
            html_block = render_token_highlight_html(
                text,
                block.get("spans") or [],
                block.get("ids") or [],
                color,
            )
            # st.html avoids Streamlit markdown mangling backticks/*/_/#.
            st.html(html_block)


def page_comparison(cli: TokLabClient, exp_id: str) -> None:
    st.header("2 · Comparison (tokens/KB)")
    fmap = fidelity_map(cli, exp_id)
    try:
        data = cli.results(exp_id, table="token_efficiency_ci")
    except ApiUnavailable as exc:
        stop_on_unavailable(exc)
    except httpx.HTTPStatusError as exc:
        stop_on_http_error(exc)
    rows = data.get("rows") or []
    if not rows:
        st.warning("Not run — no chart.")
        return
    df = pd.DataFrame(rows)
    notes = []
    keep = []
    for _, r in df.iterrows():
        cond = str(r["condition"])
        note = grey_note(cond, fmap)
        if note:
            notes.append(note)
            continue
        keep.append(r)
    for n in notes:
        st.caption(n)
    if not keep:
        st.warning("All conditions greyed out / invalid.")
        return
    plot_df = pd.DataFrame(keep)
    fig = go.Figure()
    for _, r in plot_df.iterrows():
        fig.add_trace(
            go.Bar(
                name=str(r["condition"]),
                x=[str(r["condition"])],
                y=[float(r["estimate"])],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [float(r["ci_high"]) - float(r["estimate"])],
                    "arrayminus": [float(r["estimate"]) - float(r["ci_low"])],
                },
                marker_color=COLORS.get(str(r["condition"]), "#666"),
            )
        )
    fig.update_layout(yaxis_title="tokens / KB (lower better)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def page_factorial(cli: TokLabClient, exp_id: str) -> None:
    st.header("3 · 2×2 interaction")
    fmap = fidelity_map(cli, exp_id)
    for cond in ("C2", "C3", "C4", "C5"):
        note = grey_note(cond, fmap)
        if note:
            st.caption(note)
    try:
        data = cli.factorial(exp_id)
    except ApiUnavailable as exc:
        stop_on_unavailable(exc)
    except httpx.HTTPStatusError as exc:
        stop_on_http_error(exc)
    rows = data.get("rows") or []
    if not rows:
        st.warning("Not run — no chart.")
        return
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["k"],
            y=df["interaction"],
            mode="lines+markers",
            name="I",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": df["interaction_high"] - df["interaction"],
                "arrayminus": df["interaction"] - df["interaction_low"],
            },
        )
    )
    fig.update_layout(xaxis_title="K", yaxis_title="interaction I (tokens/KB)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)


def page_context(cli: TokLabClient, exp_id: str) -> None:
    st.header("4 · Context utilisation")
    fmap = fidelity_map(cli, exp_id)
    try:
        data = cli.results(exp_id, table="context_utilisation")
    except ApiUnavailable as exc:
        stop_on_unavailable(exc)
    except httpx.HTTPStatusError as exc:
        stop_on_http_error(exc)
    rows = data.get("rows") or []
    if not rows:
        st.warning("Not run — no chart.")
        return
    df = pd.DataFrame(rows)
    invalid = {c for c, info in fmap.items() if not info["valid"]}
    for c in invalid:
        st.caption(f"{c} greyed out — {fmap[c]['reason']}")
    df = df[~df["condition"].isin(invalid)] if "condition" in df.columns else df
    if df.empty:
        st.warning("Nothing valid to plot.")
        return
    fig = px.line(
        df,
        x="budget_tokens",
        y="bytes_fit",
        color="condition",
        markers=True,
        color_discrete_map=COLORS,
    )
    st.plotly_chart(fig, use_container_width=True)


def page_cost(cli: TokLabClient, exp_id: str) -> None:
    st.header("5 · Cost (latency / KV / vocab overhead)")
    try:
        sys = cli.results(exp_id, table="systems")
        sys_rows = sys.get("rows") or []
    except ApiUnavailable as exc:
        stop_on_unavailable(exc)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 409:
            sys_rows = []
            st.info("Systems latency: not run.")
        else:
            stop_on_http_error(exc)
            return
    if sys_rows:
        sdf = pd.DataFrame(sys_rows)
        fig = px.line(
            sdf,
            x="seq_len",
            y="latency_ms_median",
            color="condition",
            markers=True,
            color_discrete_map=COLORS,
        )
        fig.update_layout(yaxis_title="median latency (ms)")
        st.plotly_chart(fig, use_container_width=True)
    try:
        vc = cli.results(exp_id, table="vocab_cost")
        rows = vc.get("rows") or []
    except ApiUnavailable as exc:
        stop_on_unavailable(exc)
    except httpx.HTTPStatusError as exc:
        stop_on_http_error(exc)
    if not rows:
        st.warning("Vocab overhead not run — no table.")
        return
    st.subheader("Vocab / KV overhead (ESTIMATE)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    vdf = pd.DataFrame(rows)
    if "kv_kv_cache_bytes" in vdf.columns:
        fig2 = px.line(
            vdf,
            x="seq_len",
            y="kv_kv_cache_bytes",
            markers=True,
            title="KV cache bytes",
        )
        st.plotly_chart(fig2, use_container_width=True)


def page_quality(cli: TokLabClient, exp_id: str) -> None:
    st.header("6 · Quality (BPB)")
    fmap = fidelity_map(cli, exp_id)
    try:
        data = cli.results(exp_id, table="quality")
        rows = data.get("rows") or []
    except ApiUnavailable as exc:
        stop_on_unavailable(exc)
    except httpx.HTTPStatusError as exc:
        stop_on_http_error(exc)
    if not rows:
        st.warning("Quality not run — no chart.")
        return
    for row in rows:
        cond = str(row.get("condition"))
        gate = str(row.get("gate_verdict", ""))
        valid = fmap.get(cond, {}).get("valid", True)
        badge = f"`{gate}`" if gate else "`unknown`"
        if not valid or gate in {"UNTRAINED_NEW_ROWS", "INCOMPATIBLE"}:
            st.markdown(
                f"**{cond}** {badge} — greyed out "
                f"({fmap.get(cond, {}).get('reason', gate)})"
            )
        else:
            st.markdown(f"**{cond}** {badge} — BPB = {row.get('bpb')}")


def page_drift(cli: TokLabClient, exp_id: str) -> None:
    st.header("7 · Drift curve")
    try:
        data = cli.drift(exp_id)
    except ApiUnavailable as exc:
        stop_on_unavailable(exc)
    except httpx.HTTPStatusError as exc:
        stop_on_http_error(exc)
    rows = data.get("rows") or []
    if not rows:
        st.warning("Drift not run — no chart.")
        return
    df = pd.DataFrame(rows)
    fig = px.line(
        df,
        x="proportion",
        y="token_reduction",
        color="variant",
        markers=True,
        title="Token reduction vs baseline by drift proportion",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)


def page_report(cli: TokLabClient, exp_id: str) -> None:
    st.header("8 · Report download")
    try:
        md = cli.report_md(exp_id)
    except ApiUnavailable as exc:
        stop_on_unavailable(exc)
    except httpx.HTTPStatusError as exc:
        stop_on_http_error(exc)
    st.download_button(
        "Download report.md",
        data=md,
        file_name=f"toklab_report_{exp_id[:12]}.md",
        mime="text/markdown",
    )
    st.markdown(md)


def main() -> None:
    st.title("TokLab")
    st.caption("Display-only UI over stored experiment results (via API).")
    cli = client()
    # Sidebar probe once; do not stop the script here so Tokenize can
    # still demonstrate the same friendly error if the API drops later.
    try:
        cli.datasets()
        st.sidebar.success("API connected")
    except ApiUnavailable:
        st.sidebar.error("API not reachable")

    page = st.sidebar.radio(
        "Page",
        [
            "Preview",
            "Comparison",
            "2x2 interaction",
            "Context utilisation",
            "Cost",
            "Quality",
            "Drift",
            "Report",
        ],
    )
    exp_id = st.sidebar.text_input(
        "Experiment id (config hash)",
        value="",
        help="From runs/<hash>/ after `toklab run experiment`",
    )

    try:
        if page == "Preview":
            page_preview(cli)
            return
        if not exp_id.strip():
            st.info(
                "Enter an experiment id in the sidebar for stored-result pages."
            )
            return
        eid = exp_id.strip()
        try:
            meta = cli.experiment(eid)
        except ApiUnavailable as exc:
            stop_on_unavailable(exc)
        except httpx.HTTPStatusError as exc:
            stop_on_http_error(exc)
        st.sidebar.write(f"status: **{meta.get('status')}**")
        st.sidebar.json(meta.get("stages") or {})

        dispatch = {
            "Comparison": page_comparison,
            "2x2 interaction": page_factorial,
            "Context utilisation": page_context,
            "Cost": page_cost,
            "Quality": page_quality,
            "Drift": page_drift,
            "Report": page_report,
        }
        dispatch[page](cli, eid)
    except ApiUnavailable as exc:
        # Safety net: any missed page-level catch still dies cleanly.
        stop_on_unavailable(exc)
    except httpx.HTTPStatusError as exc:
        stop_on_http_error(exc)
    except httpx.ConnectError:
        # Last resort if a call bypassed the client wrapper somehow.
        stop_on_unavailable(ApiUnavailable(cli.base_url))
    except httpx.TimeoutException:
        stop_on_unavailable(ApiUnavailable(cli.base_url))


if __name__ == "__main__":
    main()
