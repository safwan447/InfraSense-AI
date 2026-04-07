a (%)", yaxis_title=None)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE — PROJECT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Project Registry":
    st.title("📋 Project Registry")
    projects = get_projects()

    c1,c2,c3,c4 = st.columns(4)
    rf    = c1.selectbox("Risk",   ["All","High","Medium","Low"])
    tf    = c2.selectbox("Type",   ["All"]+sorted(projects["project_type"].dropna().unique().tolist()))
    sf    = c3.selectbox("State",  ["All"]+sorted(projects["state"].dropna().unique().tolist()))
    df_src = c4.selectbox("Source",["All"]+sorted(projects["data_source"].dropna().unique().tolist()))

    fdf = projects.copy()
    if rf     != "All": fdf = fdf[fdf.risk_label==rf]
    if tf     != "All": fdf = fdf[fdf.project_type==tf]
    if sf     != "All": fdf = fdf[fdf.state==sf]
    if df_src != "All": fdf = fdf[fdf.data_source==df_src]

    st.caption(f"Showing {len(fdf)} of {len(projects)} projects")
    st.divider()

    for _, row in fdf.iterrows():
        c1,c2,c3,c4,c5 = st.columns([3,1.2,1,1.4,1.5])
        c1.markdown(f"**{row['project_name']}**  \n`{row.get('project_type','')}` · {row.get('state','')}")
        c2.markdown(f"**₹{int(row.get('sanctioned_budget_cr',0)):,} Cr**  \n<small>sanctioned</small>", unsafe_allow_html=True)
        c3.markdown(f"**{int(row.get('planned_duration_mo',0))} mo**  \n<small>duration</small>", unsafe_allow_html=True)
        c4.markdown(risk_badge(row["risk_label"]) + f"  \n<small><span class='src-badge'>{row.get('data_source','')}</span></small>", unsafe_allow_html=True)
        c5.progress(int(min(row.get("risk_score",30),100)), text=f"Score: {row.get('risk_score',0)}")
        st.divider()

    st.download_button("📥 Export filtered list", fdf.to_csv(index=False),
                       "filtered_projects.csv","text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE — WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 What-If Simulator":
    st.title("🔮 What-If Simulator")
    st.caption("Manually tune parameters, find the lowest-risk configuration instantly, or let the AI optimizer trace the gradient path automatically.")
    projects = get_projects()

    sel  = st.selectbox("Select project", projects["project_name"].tolist())
    base = projects[projects["project_name"] == sel].iloc[0]
    orig = base.get("risk_label", "Medium")

    # Clear stale gradient-descent results when the user switches project
    if st.session_state.get("gd_project") != sel:
        st.session_state.pop("gd_result", None)
        st.session_state.pop("gd_project", None)
        st.session_state.pop("lowest_risk_result", None)

    sim_tab, opt_tab = st.tabs(["🎛️ Manual simulator", "🤖 Gradient descent optimizer"])

    # ── TAB 1 — MANUAL SIMULATOR ──────────────────────────────────────────────
    with sim_tab:
        col_in, col_out = st.columns(2)

        with col_in:
            st.subheader("Adjust parameters")
            budget_util  = st.slider("Budget utilisation (%)",      20, 115, int(base.get("budget_utilised_pct",  70)),  key="s1")
            land_acq     = st.slider("Land acquisition (%)",        10, 100, int(base.get("land_acquisition_pct", 60)),  key="s2")
            env_clear    = st.slider("Environmental clearance (%)", 10, 100, int(base.get("env_clearance_pct",    70)),  key="s3")
            c_rating     = st.slider("Contractor rating",           1.0, 5.0, float(base.get("contractor_rating", 3.5)), key="s4", step=0.1)
            timeline_buf = st.slider("Timeline buffer (months)",    0,  24,  int(base.get("timeline_buffer_mo",   4)),   key="s5")
            funding_rel  = st.slider("Funding released (%)",        20, 100, int(base.get("funding_released_pct", 75)),  key="s6")
            rev_count    = st.slider("Revision count",              0,   5,  int(base.get("revision_count",       1)),   key="s7")

            st.divider()

            # ── ✨ LOWEST RISK BUTTON ─────────────────────────────────────────
            st.markdown("**🎯 Instant optimiser**")
            st.caption("Find the lowest-risk parameter combination for the current slider constraints in one click.")
            find_lowest = st.button(
                "🔍 Predict Lowest Risk Configuration",
                type="primary",
                use_container_width=True,
                key="btn_lowest_risk"
            )

            if find_lowest:
                base_vals_lr = {
                    "budget_utilised_pct":  float(budget_util),
                    "land_acquisition_pct": float(land_acq),
                    "env_clearance_pct":    float(env_clear),
                    "funding_released_pct": float(funding_rel),
                    "timeline_buffer_mo":   float(timeline_buf),
                    "contractor_rating":    float(c_rating),
                    "num_past_delays":      float(base.get("num_past_delays", 1)),
                    "elapsed_pct":          float(base.get("elapsed_pct", 50)),
                    "planned_duration_mo":  float(base.get("planned_duration_mo", 36)),
                    "revision_count":       float(rev_count),
                }
                with st.spinner("Running gradient descent to find lowest risk…"):
                    opt_p, loss_h, param_h = gradient_descent_optimize(
                        model, le, base_vals_lr, lr=2.0, iterations=150
                    )
                opt_full = {**base_vals_lr, **opt_p}
                opt_risk, opt_probs  = predict(model, le, opt_full)
                cur_risk, cur_probs  = predict(model, le, base_vals_lr)
                cur_domain = domain_risk_score(base_vals_lr)
                opt_domain = domain_risk_score(opt_full)
                st.session_state["lowest_risk_result"] = {
                    "opt_params":  opt_p,
                    "opt_risk":    opt_risk,
                    "opt_probs":   opt_probs,
                    "cur_risk":    cur_risk,
                    "cur_probs":   cur_probs,
                    "loss_hist":   loss_h,
                    "base_vals":   base_vals_lr,
                    "cur_domain":  cur_domain,
                    "opt_domain":  opt_domain,
                }

        # ── Current manual simulation output ──────────────────────────────────
        vals = {
            "budget_utilised_pct":  budget_util,
            "land_acquisition_pct": land_acq,
            "contractor_rating":    c_rating,
            "num_past_delays":      float(base.get("num_past_delays", 1)),
            "timeline_buffer_mo":   timeline_buf,
            "env_clearance_pct":    env_clear,
            "revision_count":       rev_count,
            "funding_released_pct": funding_rel,
            "elapsed_pct":          float(base.get("elapsed_pct", 50)),
            "planned_duration_mo":  float(base.get("planned_duration_mo", 36)),
        }
        new_risk, probs = predict(model, le, vals)

        with col_out:
            st.subheader("Live prediction")
            oc1, oc2 = st.columns(2)
            oc1.markdown(f"**Original**  \n{risk_badge(orig)}", unsafe_allow_html=True)
            oc2.markdown(f"**Simulated**  \n{risk_badge(new_risk)}", unsafe_allow_html=True)

            if orig != new_risk:
                if new_risk == "Low":    st.success("✅ Risk reduced with these changes!")
                elif new_risk == "High": st.error("⚠️ Risk increased with these parameters.")
                else:                    st.warning("🟡 Moderate — further improvements possible.")

            st.divider()
            st.markdown("**Probability breakdown**")
            for lvl in ["High", "Medium", "Low"]:
                v = probs.get(lvl, 0)
                st.markdown(f"`{lvl:6s}` {v:.1f}%")
                st.progress(int(v))

            st.divider()
            st.markdown("**Recommended actions**")
            recos = []
            if budget_util > 90:   recos.append("🔴 Request supplementary budget immediately")
            if land_acq    < 60:   recos.append("🔴 Escalate land acquisition to district collector")
            if c_rating    < 2.5:  recos.append("🟡 Issue contractor performance notice")
            if env_clear   < 50:   recos.append("🟡 Fast-track MoEF clearance via single-window")
            if timeline_buf < 3:   recos.append("🟡 Extend deadline or reduce current scope")
            if funding_rel  < 50:  recos.append("🟡 Follow up with Finance Ministry")
            if not recos:          recos.append("✅ No critical interventions required")
            for rec in recos:
                st.markdown(f"- {rec}")

        # ── Lowest Risk Result Panel (shown below full width) ─────────────────
        if "lowest_risk_result" in st.session_state:
            lr_res = st.session_state["lowest_risk_result"]
            st.divider()
            st.subheader("🎯 Lowest-Risk Configuration Found")
            st.markdown(
                '<span class="gd-badge">Gradient Descent · Optimal Parameters</span>',
                unsafe_allow_html=True
            )
            st.markdown("")

            lc1, lc2, lc3 = st.columns(3)
            lc1.markdown(f"**Current risk**  \n{risk_badge(lr_res['cur_risk'])}", unsafe_allow_html=True)
            lc2.markdown(f"**Optimal risk**  \n{risk_badge(lr_res['opt_risk'])}", unsafe_allow_html=True)
            # Use domain score reduction (smooth 0-100) — P(High) from classifier is always hard 0/1
            score_reduction = lr_res.get("cur_domain", 0) - lr_res.get("opt_domain", 0)
            lc3.metric("Risk score reduced by", f"{score_reduction:.1f} pts",
                       delta=f"-{score_reduction:.1f}", delta_color="inverse")

            # Convergence mini-chart — loss_hist is domain_risk_score values (0-100)
            loss_df = pd.DataFrame({
                "Iteration":    list(range(len(lr_res["loss_hist"]))),
                "Risk score": [round(v, 2) for v in lr_res["loss_hist"]]
            })
            fig_conv = px.area(loss_df, x="Iteration", y="Risk score",
                               color_discrete_sequence=["#2ecc71"])
            fig_conv.add_hline(
                y=lr_res["loss_hist"][-1], line_dash="dash", line_color="#27ae60",
                annotation_text=f"Converged at {lr_res['loss_hist'][-1]:.1f}"
            )
            fig_conv.update_layout(height=200, margin=dict(t=10, b=5),
                                   xaxis_title="Iteration", yaxis_title="Domain risk score (0–100)")
            st.plotly_chart(fig_conv, use_container_width=True)

            # Parameter change table
            PARAM_LABELS = {
                "land_acquisition_pct":  ("Land acquisition",  "%"),
                "env_clearance_pct":     ("Env. clearance",    "%"),
                "funding_released_pct":  ("Funding released",  "%"),
                "timeline_buffer_mo":    ("Timeline buffer",   " mo"),
                "contractor_rating":     ("Contractor rating", "/5"),
            }
            opt_rows = []
            for key, (label, unit) in PARAM_LABELS.items():
                original = lr_res["base_vals"].get(key, 0)
                optimal  = lr_res["opt_params"].get(key, original)
                change   = optimal - original
                opt_rows.append({
                    "Parameter": label,
                    "Current":   f"{original:.1f}{unit}",
                    "Optimal":   f"{optimal:.1f}{unit}",
                    "Change":    f"{'+' if change>=0 else ''}{change:.1f}{unit}",
                    "Action":    "↑ Increase" if change > 0.5 else ("↓ Decrease" if change < -0.5 else "→ No change"),
                })
            opt_param_df = pd.DataFrame(opt_rows)
            st.dataframe(opt_param_df, use_container_width=True, hide_index=True)

            # Action cards
            st.markdown("**📋 Optimal intervention plan**")
            for key, (label, unit) in PARAM_LABELS.items():
                original = lr_res["base_vals"].get(key, 0)
                optimal  = lr_res["opt_params"].get(key, original)
                change   = optimal - original
                if change > 0.5:
                    st.success(f"✅ **{label}**: Increase from {original:.1f}{unit} → {optimal:.1f}{unit}")
                elif change < -0.5:
                    st.warning(f"🟡 **{label}**: Reduce from {original:.1f}{unit} → {optimal:.1f}{unit}")
                else:
                    st.info(f"ℹ️ **{label}**: Already near-optimal ({original:.1f}{unit})")

    # ── TAB 2 — GRADIENT DESCENT (full trace) ────────────────────────────────
    with opt_tab:
        st.subheader("AI-powered parameter optimizer — full gradient trace")
        st.markdown("""
        Uses **numerical gradient descent** to find the combination of actionable parameters
        that minimises the probability of project delay/overrun.

        - **Fixed inputs** (elapsed time, past delays, planned duration, revision count, budget utilisation) are not changed
        - **Optimiser adjusts:** land acquisition, env clearance, funding released, timeline buffer, contractor rating
        - **Algorithm:** finite-difference gradient with configurable learning rate; stops early when gradient < 1e-6
        - Runs up to 200 iterations; the convergence curve and parameter trajectories are plotted in full
        """)

        oc1, oc2 = st.columns(2)
        lr_gd      = oc1.slider("Learning rate", 0.5, 5.0, 1.5, step=0.5,
                                 help="How aggressively the optimizer moves each step.")
        iterations = oc2.slider("Max iterations", 20, 200, 120, step=10,
                                 help="Maximum number of gradient steps.")

        run_opt = st.button("🚀 Run Gradient Descent (Full Trace)", type="primary",
                            use_container_width=True)

        if run_opt:
            base_vals_gd = {
                "budget_utilised_pct":  float(base.get("budget_utilised_pct",  70)),
                "land_acquisition_pct": float(base.get("land_acquisition_pct", 60)),
                "env_clearance_pct":    float(base.get("env_clearance_pct",    70)),
                "funding_released_pct": float(base.get("funding_released_pct", 75)),
                "timeline_buffer_mo":   float(base.get("timeline_buffer_mo",    4)),
                "contractor_rating":    float(base.get("contractor_rating",    3.5)),
                "num_past_delays":      float(base.get("num_past_delays",        1)),
                "elapsed_pct":          float(base.get("elapsed_pct",          50)),
                "planned_duration_mo":  float(base.get("planned_duration_mo",  36)),
                "revision_count":       float(base.get("revision_count",         1)),
            }
            with st.spinner("Running gradient descent…"):
                opt_params, loss_hist, param_hist = gradient_descent_optimize(
                    model, le, base_vals_gd, lr=lr_gd, iterations=iterations
                )
            st.session_state["gd_result"]  = (opt_params, loss_hist, param_hist, base_vals_gd)
            st.session_state["gd_project"] = sel

        if "gd_result" in st.session_state:
            opt_params, loss_hist, param_hist, base_vals_gd = st.session_state["gd_result"]

            orig_risk_lbl, orig_probs_gd = predict(model, le, base_vals_gd)
            opt_vals                      = {**base_vals_gd, **opt_params}
            opt_risk_lbl,  opt_probs_gd   = predict(model, le, opt_vals)
            orig_domain = domain_risk_score(base_vals_gd)
            opt_domain  = domain_risk_score(opt_vals)

            st.divider()
            st.subheader("Before vs after optimization")
            bc1, bc2, bc3 = st.columns(3)
            bc1.markdown(f"**Original risk**  \n{risk_badge(orig_risk_lbl)}", unsafe_allow_html=True)
            bc2.markdown(f"**Optimized risk**  \n{risk_badge(opt_risk_lbl)}", unsafe_allow_html=True)
            # Use smooth domain score reduction — classifier P(High) is always hard 0 or 1
            score_red = orig_domain - opt_domain
            bc3.metric("Risk score reduced by", f"{score_red:.1f} pts",
                       delta=f"-{score_red:.1f}", delta_color="inverse")

            st.divider()
            st.subheader("Gradient descent convergence")
            # loss_hist contains domain_risk_score values (0-100), not probabilities
            loss_df = pd.DataFrame({
                "Iteration":  list(range(len(loss_hist))),
                "Risk score": [round(v, 2) for v in loss_hist]
            })
            fig_loss = px.line(loss_df, x="Iteration", y="Risk score",
                               color_discrete_sequence=["#e74c3c"])
            fig_loss.add_hline(y=loss_hist[-1], line_dash="dash", line_color="#27ae60",
                               annotation_text=f"Converged at {loss_hist[-1]:.1f}")
            fig_loss.update_layout(height=280, margin=dict(t=10, b=10),
                                   yaxis_title="Domain risk score (0–100)")
            st.plotly_chart(fig_loss, use_container_width=True)

            st.divider()
            st.subheader("Optimal parameter values found")
            LABELS = {
                "land_acquisition_pct":  ("Land acquisition",  "%",   base_vals_gd["land_acquisition_pct"]),
                "env_clearance_pct":     ("Env. clearance",    "%",   base_vals_gd["env_clearance_pct"]),
                "funding_released_pct":  ("Funding released",  "%",   base_vals_gd["funding_released_pct"]),
                "timeline_buffer_mo":    ("Timeline buffer",   " mo", base_vals_gd["timeline_buffer_mo"]),
                "contractor_rating":     ("Contractor rating", "/5",  base_vals_gd["contractor_rating"]),
            }
            rows = []
            for key, (label, unit, original) in LABELS.items():
                opt_val = opt_params[key]
                change  = opt_val - original
                rows.append({
                    "Parameter": label,
                    "Current":   f"{original:.1f}{unit}",
                    "Optimal":   f"{opt_val:.1f}{unit}",
                    "Change":    f"{'+' if change >= 0 else ''}{change:.1f}{unit}",
                    "Direction": "↑ Increase" if change > 0.5 else ("↓ Decrease" if change < -0.5 else "→ No change"),
                })
            param_df = pd.DataFrame(rows)
            st.dataframe(param_df, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Parameter trajectories during optimization")
            fig_params = go.Figure()
            colors = ["#3498db","#2ecc71","#e67e22","#9b59b6","#e74c3c"]
            for i, (key, vals_hist) in enumerate(param_hist.items()):
                label = LABELS[key][0]
                fig_params.add_trace(go.Scatter(
                    x=list(range(len(vals_hist))),
                    y=[round(v, 2) for v in vals_hist],
                    name=label, mode="lines",
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            fig_params.update_layout(
                height=300, margin=dict(t=10, b=10),
                xaxis_title="Iteration", yaxis_title="Parameter value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_params, use_container_width=True)

            st.divider()
            st.subheader("Optimal intervention plan")
            for key, (label, unit, original) in LABELS.items():
                change = opt_params[key] - original
                if change > 0.5:
                    st.success(f"✅ **{label}**: Increase from {original:.1f}{unit} → {opt_params[key]:.1f}{unit}")
                elif change < -0.5:
                    st.warning(f"🟡 **{label}**: Reduce from {original:.1f}{unit} → {opt_params[key]:.1f}{unit}")
                else:
                    st.info(f"ℹ️ **{label}**: No significant change needed ({original:.1f}{unit})")
        else:
            st.info("Select a project above and click **Run Gradient Descent (Full Trace)** to visualise the full optimisation path.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE — CONTRACTOR PROFILES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👷 Contractor Profiles":
    st.title("👷 Contractor Performance Profiles")

    c1,c2,c3 = st.columns(3)
    c1.metric("Contractors tracked", len(CONTRACTORS_META))
    c2.metric("High-risk vendors",   len(CONTRACTORS_META[CONTRACTORS_META.rating < 3.0]))
    c3.metric("Avg. rating",         f"{CONTRACTORS_META['rating'].mean():.2f}/5.0")

    st.subheader("Performance radar comparison")
    fig = go.Figure()
    cats = ["On-time delivery","Budget adherence","Quality score"]
    for _, row in CONTRACTORS_META.iterrows():
        rc = "#27ae60" if row.rating >= 4.0 else ("#e67e22" if row.rating >= 3.0 else "#e74c3c")
        fig.add_trace(go.Scatterpolar(
            r=[row.ontime, row.budget_adh, row.quality, row.ontime],
            theta=cats+[cats[0]], fill="toself",
            name=row["name"], line_color=rc, opacity=0.65
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])), height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Individual profiles")
    for _, row in CONTRACTORS_META.iterrows():
        rl = "Low" if row.rating >= 4.0 else ("Medium" if row.rating >= 3.0 else "High")
        with st.expander(f"{row['name']}  ·  {row.rating}/5.0  [{rl} risk]"):
            e1,e2,e3,e4 = st.columns(4)
            e1.metric("On-time",  f"{row.ontime}%")
            e2.metric("Budget",   f"{row.budget_adh}%")
            e3.metric("Quality",  f"{row.quality}%")
            e4.metric("Projects", row.projects)
            bdf = pd.DataFrame({"Metric":["On-time","Budget","Quality"],
                                 "Score":[row.ontime, row.budget_adh, row.quality]})
            fig = px.bar(bdf, x="Metric", y="Score", color="Score",
                         color_continuous_scale=["#e74c3c","#f39c12","#2ecc71"],
                         range_y=[0,100], text="Score")
            fig.update_layout(height=200, margin=dict(t=10,b=10),
                               coloraxis_showscale=False, yaxis_title=None)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            if rl == "High":     st.error("⚠️ Do not assign to critical projects without enhanced monitoring.")
            elif rl == "Medium": st.warning("🟡 Use milestone-based payments and monthly reviews.")
            else:                st.success("✅ Reliable. Suitable for high-value assignments.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE — ADD NEW PROJECT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "➕ Add New Project":
    st.title("➕ Add New Project")
    st.caption("Pre-sanction AI screening — catch risk before public funds are committed.")

    fc1,fc2 = st.columns(2)
    name  = fc1.text_input("Project name", placeholder="e.g. NH-66 Coastal Highway")
    ptype = fc2.selectbox("Type", ["Highway","Metro Rail","Bridge","Urban Road","Flyover","Tunnel","Port","Airport"])

    fc3,fc4 = st.columns(2)
    state    = fc3.selectbox("State", ["Maharashtra","Karnataka","Delhi","Tamil Nadu","Uttar Pradesh",
                                        "Telangana","Gujarat","Rajasthan","West Bengal","Punjab","Kerala","Bihar"])
    ministry = fc4.selectbox("Ministry", ["MoRTH","CPWD","NHAI","State PWD","DMRC","AAI","MoPSW"])

    fc5,fc6 = st.columns(2)
    budget = fc5.number_input("Sanctioned budget (₹ Cr)", 100, 200000, 5000, step=100)
    dur    = fc6.number_input("Planned duration (months)", 6, 120, 36)

    rc1,rc2 = st.columns(2)
    land = rc1.slider("Land acquisition (%)",        0, 100, 50)
    env  = rc2.slider("Environmental clearance (%)", 0, 100, 60)

    rc3,rc4 = st.columns(2)
    cr  = rc3.slider("Contractor rating",          1.0, 5.0, 3.5, step=0.1)
    buf = rc4.slider("Timeline buffer (months)",   0, 24, 6)

    rc5,rc6 = st.columns(2)
    fund = rc5.slider("Funding released (%)",  0, 100, 70)
    rev  = rc6.slider("Expected revisions",    0,   5,  1)

    st.markdown("")
    predict_btn = st.button("🔮 Predict Risk", use_container_width=True, type="primary")

    if predict_btn:
        if not name.strip():
            st.error("Please enter a project name above before predicting.")
            st.session_state.pop("new_proj_result", None)
        else:
            input_vals = {
                "budget_utilised_pct":  20.0,
                "land_acquisition_pct": float(land),
                "contractor_rating":    float(cr),
                "num_past_delays":      0.0,
                "timeline_buffer_mo":   float(buf),
                "env_clearance_pct":    float(env),
                "revision_count":       float(rev),
                "funding_released_pct": float(fund),
                "elapsed_pct":          0.0,
                "planned_duration_mo":  float(dur),
            }
            rl, probs = predict(model, le, input_vals)
            st.session_state["new_proj_result"] = {
                "name": name, "rl": rl, "probs": probs,
                "land": land, "env": env, "cr": cr, "buf": buf, "fund": fund,
            }

    if "new_proj_result" in st.session_state:
        res   = st.session_state["new_proj_result"]
        rl    = res["rl"]
        probs = res["probs"]
        land  = res["land"]
        env   = res["env"]
        cr    = res["cr"]
        buf   = res["buf"]
        fund  = res["fund"]

        st.divider()
        st.subheader(f"Risk assessment: {res['name']}")

        r1, r2, r3 = st.columns(3)
        r1.markdown(f"**Predicted risk**  \n{risk_badge(rl)}", unsafe_allow_html=True)
        r2.metric("High probability",   f"{probs.get('High',   0):.1f}%")
        r3.metric("Medium probability", f"{probs.get('Medium', 0):.1f}%")

        st.divider()
        st.markdown("**Probability breakdown**")
        for lvl in ["High", "Medium", "Low"]:
            v = probs.get(lvl, 0)
            st.markdown(f"`{lvl:6s}` {v:.1f}%")
            st.progress(int(v))

        recos = []
        if land < 60:  recos.append("Ensure land acquisition ≥80% before commencement")
        if env  < 60:  recos.append("Obtain all clearances before tender award")
        if cr   < 3.0: recos.append("Select contractor with rating ≥3.5")
        if buf  < 4:   recos.append("Add minimum 6-month timeline buffer")
        if fund < 60:  recos.append("Secure full funding commitment before start")

        st.divider()
        if recos:
            reco_text = "\n".join(f"- {rec}" for rec in recos)
            st.warning(f"**Pre-sanction recommendations:**\n{reco_text}")
        else:
            st.success("✅ Parameters look healthy. Proceed with standard monitoring.")