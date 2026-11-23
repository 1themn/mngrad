from visualization.gradient_visualizer import (
    gradient_landscape_figure,
    get_parameter_labels
)
import os
import sys
import time
import tempfile
import subprocess
import webbrowser
import socket

import cloudpickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# ==========================================
# LAUNCH FROM YOUR NOTEBOOK
# ==========================================

def launch(model_factory, xs, ys):

    path = os.path.join(
        tempfile.mkdtemp(),
        "autograd_model.pkl"
    )

    with open(path, "wb") as f:
        cloudpickle.dump(
            (model_factory, xs, ys), f
        )

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    subprocess.Popen([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        os.path.abspath(__file__),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--",
        path
    ])

    time.sleep(2)
    webbrowser.open(f"http://localhost:{port}")


# ==========================================
# THEME
# ==========================================

BG = "#080c14"
PANEL = "#101722"
GRID = "#263244"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
CYAN = "#22d3ee"
PURPLE = "#a78bfa"
GREEN = "#34d399"
RED = "#fb7185"


CSS = """
<style>

.stApp {
    background: #080c14;
    color: #e2e8f0;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

[data-testid="stSidebar"] {
    background: #0d1420;
    border-right: 1px solid #243044;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg, #141e2d, #101722
    );
    border: 1px solid #263244;
    padding: 20px;
    border-radius: 16px;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
    font-size: 12px;
    letter-spacing: 1px;
}

[data-testid="stMetricValue"] {
    color: #f1f5f9;
    font-weight: 700;
}

.stButton > button {
    background: #162235;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 10px;
    min-height: 44px;
    width: 100%;
    font-weight: 600;
}

.stButton > button:hover {
    border-color: #22d3ee;
    color: #22d3ee;
}

.stButton > button[kind="primary"] {
    background: #0891b2;
    border: none;
    color: white;
}

.hero {
    margin-bottom: 26px;
}

.hero-kicker {
    color: #22d3ee;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -2px;
    margin-top: 6px;
    margin-bottom: 4px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 15px;
}

.section-title {
    font-size: 16px;
    font-weight: 650;
    margin: 14px 0 5px;
    color: #e2e8f0;
}

.section-subtitle {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 12px;
}

.status {
    display: inline-block;
    border: 1px solid #164e63;
    background: #083344;
    color: #67e8f9;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

</style>
"""


# ==========================================
# PLOTLY THEME
# ==========================================

def style_figure(fig, height=400):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(
            family="Inter, Arial, sans-serif",
            color=TEXT,
            size=12
        ),
        height=height,
        margin=dict(l=35, r=25, t=35, b=35),
        uirevision="keep-camera",
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_color=TEXT,
            bordercolor=GRID
        )
    )

    return fig


def section(title, subtitle):

    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# MODEL STATE
# ==========================================

def initialize(model_factory):

    st.session_state.model = model_factory()
    st.session_state.epoch = 0
    st.session_state.loss_history = []
    st.session_state.grad_history = []
    st.session_state.running = False
    st.session_state.surface = None
    st.session_state.initial_surface = None
    st.session_state.gradient_snapshot = None
    st.session_state.pop("initial_parameters", None)
    st.session_state.parameter_history = []


def predictions(model, xs):

    return np.array([
        model(x).data for x in xs
    ])


def calculate_loss(model, xs, ys):

    pred = predictions(model, xs)

    return float(
        np.sum((pred - np.array(ys)) ** 2)
    )


# ==========================================
# TRAINING CONTROLLER
# USES YOUR AUTOGRAD ENGINE
# ==========================================

def train_step(model, xs, ys, lr):

    ypred = [model(x) for x in xs]

    loss = sum(
        (yout - ygt) ** 2
        for ygt, yout in zip(ys, ypred)
    )

    for p in model.parameters():
        p.grad = 0.0

    # Your own backpropagation
    loss.backward()

    grad_norm = np.sqrt(
        sum(
            p.grad ** 2
            for p in model.parameters()
        )
    )

    # Capture the actual gradients before updating
    gradient_snapshot = [
        [
            [w.grad for w in neuron.w]
            for neuron in layer.neurons
        ]
        for layer in model.layers
    ]

    # Gradient descent
    for p in model.parameters():
        p.data += -lr * p.grad

    return (
        calculate_loss(model, xs, ys),
        float(grad_norm),
        gradient_snapshot
    )


# ==========================================
# 3D PREDICTION SURFACE
# ==========================================

def calculate_surface(model, x3, resolution=35):

    axis = np.linspace(-3.5, 3.5, resolution)

    X, Y = np.meshgrid(axis, axis)

    Z = np.empty_like(X)

    for i in range(resolution):
        for j in range(resolution):

            Z[i, j] = model([
                float(X[i, j]),
                float(Y[i, j]),
                float(x3)
            ]).data

    return X, Y, Z


def surface_figure(
    surface,
    xs,
    ys,
    x3,
    epoch,
    show_wireframe=True,
    initial_surface=None,
    show_initial=False
):

    X, Y, Z = surface

    fig = go.Figure()

    # Main learned surface
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=Z,
            colorscale=[
                [0.0, "#172554"],
                [0.2, "#4338ca"],
                [0.4, "#7c3aed"],
                [0.6, "#06b6d4"],
                [0.8, "#2dd4bf"],
                [1.0, "#fef08a"]
            ],
            cmin=-1,
            cmax=1,
            opacity=0.96,
            colorbar=dict(
                title="Prediction",
                thickness=12,
                len=0.65,
                tickfont=dict(color=MUTED)
            ),
            lighting=dict(
                ambient=0.65,
                diffuse=0.9,
                specular=0.5,
                roughness=0.35,
                fresnel=0.2
            ),
            lightposition=dict(
                x=100,
                y=200,
                z=300
            ),
            contours=dict(
                x=dict(
                    show=show_wireframe,
                    color="#334155",
                    width=1
                ),
                y=dict(
                    show=show_wireframe,
                    color="#334155",
                    width=1
                )
            ),
            hovertemplate=(
                "x₁: %{x:.2f}<br>"
                "x₂: %{y:.2f}<br>"
                "Prediction: %{z:.4f}"
                "<extra></extra>"
            ),
            name="Learned surface"
        )
    )

    # Optional ghost surface from epoch zero
    if show_initial and initial_surface is not None:

        X0, Y0, Z0 = initial_surface

        fig.add_trace(
            go.Surface(
                x=X0,
                y=Y0,
                z=Z0,
                colorscale=[
                    [0, "#64748b"],
                    [1, "#94a3b8"]
                ],
                opacity=0.18,
                showscale=False,
                hoverinfo="skip",
                name="Initial surface"
            )
        )

    # Training points projected onto the x1-x2 plane.
    # Their actual x3 values are retained in hover data.
    xs_array = np.array(xs)

    fig.add_trace(
        go.Scatter3d(
            x=xs_array[:, 0],
            y=xs_array[:, 1],
            z=ys,
            mode="markers",
            customdata=xs_array[:, 2],
            marker=dict(
                size=7,
                color=[
                    GREEN if y > 0 else RED
                    for y in ys
                ],
                line=dict(
                    color="white",
                    width=2
                ),
                symbol="diamond"
            ),
            name="Projected targets",
            hovertemplate=(
                "Training sample<br>"
                "x₁: %{x}<br>"
                "x₂: %{y}<br>"
                "Actual x₃: %{customdata}<br>"
                "Target: %{z}"
                "<extra></extra>"
            )
        )
    )

    axis_style = dict(
        backgroundcolor="#101722",
        gridcolor="#334155",
        zerolinecolor="#64748b",
        showbackground=True,
        showspikes=False
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Input x₁",
                range=[-3.5, 3.5],
                **axis_style
            ),
            yaxis=dict(
                title="Input x₂",
                range=[-3.5, 3.5],
                **axis_style
            ),
            zaxis=dict(
                title="Prediction",
                range=[-1.15, 1.15],
                **axis_style
            ),
            camera=dict(
                eye=dict(x=1.55, y=1.65, z=1.05)
            ),
            aspectratio=dict(
                x=1,
                y=1,
                z=0.65
            ),
            uirevision="surface-camera"
        ),
        showlegend=False,
        title=dict(
            text=f"Epoch {epoch} · x₃ = {x3:.2f}",
            font=dict(size=13, color=MUTED)
        )
    )

    style_figure(fig, 540)

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


# ==========================================
# NETWORK ARCHITECTURE
# ==========================================

def network_figure(
    model,
    sample,
    mode="Weights",
    gradient_snapshot=None
):

    fig = go.Figure()

    sizes = [
        len(model.layers[0].neurons[0].w)
    ] + [
        len(layer.neurons)
        for layer in model.layers
    ]

    activations = [
        list(sample)
    ]

    x = list(sample)

    for layer in model.layers:

        x = [
            neuron(x)
            for neuron in layer.neurons
        ]

        activations.append(
            [v.data for v in x]
        )

    positions = []

    for size in sizes:

        positions.append(
            np.linspace(
                -(size - 1) / 2,
                (size - 1) / 2,
                size
            )
        )

    # Connections
    for li, layer in enumerate(model.layers):

        for ni, neuron in enumerate(layer.neurons):

            for wi, weight in enumerate(neuron.w):

                if mode == "Gradients":

                    value = (
                        gradient_snapshot[li][ni][wi]
                        if gradient_snapshot is not None
                        else weight.grad
                    )

                    color = PURPLE if value >= 0 else RED

                elif mode == "Forward":

                    value = (
                        weight.data *
                        activations[li][wi]
                    )

                    color = GREEN if value >= 0 else RED

                else:

                    value = weight.data
                    color = CYAN if value >= 0 else RED

                fig.add_trace(
                    go.Scatter(
                        x=[li, li + 1],
                        y=[
                            positions[li][wi],
                            positions[li + 1][ni]
                        ],
                        mode="lines",
                        line=dict(
                            color=color,
                            width=0.6 + min(
                                abs(value) * 3, 5
                            )
                        ),
                        opacity=0.5,
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

    # Neurons
    for li, size in enumerate(sizes):

        labels = []
        hover = []

        for ni in range(size):

            activation = activations[li][ni]

            labels.append(
                f"{activation:.2f}"
            )

            if li == 0:

                hover.append(
                    f"Input {ni + 1}<br>"
                    f"Value: {activation:.5f}"
                )

            else:

                neuron = model.layers[
                    li - 1
                ].neurons[ni]

                info = (
                    f"Layer {li} · Neuron {ni + 1}<br>"
                    f"Activation: {activation:.5f}<br>"
                    f"Bias: {neuron.b.data:.5f}<br>"
                )

                for wi, w in enumerate(neuron.w):

                    info += (
                        f"w{wi + 1}: {w.data:.5f}<br>"
                        f"grad: {w.grad:.5f}<br>"
                    )

                hover.append(info)

        fig.add_trace(
            go.Scatter(
                x=[li] * size,
                y=positions[li],
                mode="markers+text",
                text=labels,
                textposition="middle center",
                textfont=dict(
                    color="white",
                    size=10
                ),
                hovertext=hover,
                hoverinfo="text",
                marker=dict(
                    size=40 if li != 0 else 34,
                    color=activations[li],
                    colorscale=[
                        [0, "#be123c"],
                        [0.5, "#1e293b"],
                        [1, "#0d9488"]
                    ],
                    cmin=-1,
                    cmax=1,
                    line=dict(
                        color="#94a3b8",
                        width=1.5
                    )
                ),
                showlegend=False
            )
        )

    names = (
        ["INPUT"] +
        [
            f"HIDDEN {i}"
            for i in range(1, len(sizes) - 1)
        ] +
        ["OUTPUT"]
    )

    fig.update_xaxes(
        tickvals=list(range(len(sizes))),
        ticktext=names,
        showgrid=False,
        zeroline=False,
        fixedrange=True
    )

    fig.update_yaxes(
        visible=False,
        range=[
            -max(sizes) / 2 - 0.5,
            max(sizes) / 2 + 0.5
        ]
    )

    style_figure(fig, 360)

    return fig


# ==========================================
# TRAINING CHARTS
# ==========================================

def history_figure(
    values,
    color,
    name,
    log_scale=False
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode="lines",
            name=name,
            line=dict(
                color=color,
                width=3,
                shape="linear"
            ),
            fill="tozeroy",
            fillcolor=(
                "rgba(34,211,238,0.10)"
                if color == CYAN
                else "rgba(167,139,250,0.10)"
            ),
            hovertemplate=(
                "Epoch %{x}<br>"
                + name +
                ": %{y:.6f}<extra></extra>"
            )
        )
    )

    fig.update_xaxes(
        title="Epoch",
        gridcolor=GRID
    )

    fig.update_yaxes(
        title=name,
        gridcolor=GRID,
        type="log" if log_scale else "linear"
    )

    style_figure(fig, 270)

    return fig


# ==========================================
# MAIN APPLICATION
# ==========================================

def main():

    st.set_page_config(
        page_title="MnGrad Lab",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(
        CSS,
        unsafe_allow_html=True
    )

    # Load your notebook model without changing it
    model_path = sys.argv[-1]

    with open(model_path, "rb") as f:
        model_factory, xs, ys = cloudpickle.load(f)

    if "model" not in st.session_state:
        initialize(model_factory)

    state = st.session_state

    # ======================================
    # SIDEBAR
    # ======================================

    with st.sidebar:

        st.markdown("## MNGRAD LAB")
        st.caption("NEURAL COMPUTATION STUDIO")

        st.divider()

        st.markdown("### Training")

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "▶ Train",
                type="primary"
            ):
                state.running = True

        with col2:

            if st.button("Ⅱ Pause"):
                state.running = False

        step = st.button("→ Single iteration")

        if st.button("↻ Reset model"):

            initialize(model_factory)
            st.rerun()

        lr = st.slider(
            "Learning rate",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f"
        )

        delay = st.slider(
            "Frame delay (ms)",
            50,
            1000,
            150,
            50
        )

        st.divider()

        st.markdown("### Surface controls")

        x3 = st.slider(
            "Third input — x₃",
            -3.0,
            3.0,
            0.0,
            0.1
        )

        resolution = st.select_slider(
            "Mesh resolution",
            options=[20, 30, 40, 50, 60],
            value=30
        )

        wireframe = st.toggle(
            "Surface grid",
            value=True
        )

        show_initial = st.toggle(
            "Compare initial surface",
            value=False
        )

        st.divider()

        st.markdown("### Network controls")

        mode = st.radio(
            "Connection visualization",
            ["Weights", "Forward", "Gradients"]
        )

        sample_index = st.selectbox(
            "Training sample",
            list(range(len(xs))),
            format_func=lambda i: (
                f"Sample {i + 1} · {xs[i]}"
            )
        )

        log_loss = st.toggle(
            "Logarithmic loss scale",
            value=False
        )

    

    
    # ======================================
    # LIVE TRAINING
    # ======================================

    model = state.model

    # Initialize optimization history
    if "parameter_history" not in state:
        state.parameter_history = []

    # Record the initial model at epoch 0
    if not state.parameter_history:

        state.parameter_history.append({
            "parameters": [
                p.data for p in model.parameters()
            ],
            "loss": calculate_loss(model, xs, ys)
        })

    # Perform training iteration
    if state.running or step:

        loss, grad_norm, snapshot = train_step(
            model,
            xs,
            ys,
            lr
        )

        state.epoch += 1

        state.loss_history.append(loss)
        state.grad_history.append(grad_norm)
        state.gradient_snapshot = snapshot

        state.parameter_history.append({
            "parameters": [
                p.data for p in model.parameters()
            ],
            "loss": loss
        })

    # Current predictions
    pred = predictions(model, xs)

    loss = float(
        np.sum(
            (pred - np.array(ys)) ** 2
        )
    )

    accuracy = 100 * np.mean(
        (pred >= 0) == (np.array(ys) >= 0)
    )

    grad_norm = (
        state.grad_history[-1]
        if state.grad_history
        else 0
    )


    # ======================================
    # HEADER
    # ======================================

    status = (
        "TRAINING"
        if state.running
        else "PAUSED"
    )

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">
                BUILT FROM FIRST PRINCIPLES
            </div>
            <div class="hero-title">
                MnGrad Playground
            </div>
            <div class="hero-subtitle">
                Explore neural computation,
                gradient descent and learned
                representations in real time.
            </div>
            <br>
            <span class="status">
                ● {status}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ======================================
    # METRICS
    # ======================================

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "EPOCH",
        state.epoch
    )

    m2.metric(
        "TOTAL LOSS",
        f"{loss:.5f}"
    )

    m3.metric(
        "ACCURACY",
        f"{accuracy:.0f}%"
    )

    m4.metric(
        "GRADIENT NORM",
        f"{grad_norm:.4f}"
    )

    m5.metric(
        "PARAMETERS",
        len(model.parameters())
    )

    st.write("")

    # ======================================
    # MAIN VISUALIZATIONS
    # ======================================

    left, right = st.columns(
        [1.65, 1]
    )

    
    with left:

        section(
            "Gradient Descent Laboratory",
            "Real-time optimization landscape · "
            "Powered by my autograd engine"
        )

        labels = get_parameter_labels(model)

        c1, c2, c3 = st.columns(3)

        with c1:

            param_x = st.selectbox(
                "X parameter",
                options=list(range(len(labels))),
                format_func=lambda i: labels[i],
                index=0
            )

        with c2:

            param_y = st.selectbox(
                "Y parameter",
                options=list(range(len(labels))),
                format_func=lambda i: labels[i],
                index=1
            )

        with c3:

            span = st.slider(
                "Landscape range",
                0.5,
                5.0,
                2.0,
                0.5
            )

        if param_x == param_y:

            st.warning(
                "Select two different parameters."
            )

        else:

            fig, gradient_info = gradient_landscape_figure(

                model=model,

                xs=xs,
                ys=ys,

                param_x=param_x,
                param_y=param_y,

                span=span,

                resolution=resolution,

                history=state.parameter_history,

                epoch=state.epoch,

                learning_rate=lr,

                wireframe=wireframe,

                show_contours=True

            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="gradient_landscape"
            )

            g1, g2, g3 = st.columns(3)

            g1.metric(
                "∂Loss / ∂Weight X",
                f"{gradient_info['gradient_x']:+.5f}"
            )

            g2.metric(
                "∂Loss / ∂Weight Y",
                f"{gradient_info['gradient_y']:+.5f}"
            )

            g3.metric(
                "Projected Gradient Norm",
                f"{gradient_info['gradient_norm']:.5f}"
            )

            st.caption(
                "Gold: current parameters · "
                "Green: descent direction · "
                "Cyan: next projected step · "
                "White: optimization history"
            )

            with st.expander("Gradient calculation details"):

                st.latex(
                    r"w_{new} = w - \eta "
                    r"\frac{\partial L}{\partial w}"
                )

                st.write(
                    f"Current weight X: "
                    f"{gradient_info['weight_x']:.6f}"
                )

                st.write(
                    f"Current weight Y: "
                    f"{gradient_info['weight_y']:.6f}"
                )

                st.write(
                    f"Next weight X: "
                    f"{gradient_info['next_x']:.6f}"
                )

                st.write(
                    f"Next weight Y: "
                    f"{gradient_info['next_y']:.6f}"
                )


    with right:

        section(
            "Neural Architecture",
            "3 → 4 → 4 → 1 · tanh activation"
        )

        st.plotly_chart(
            network_figure(
                model,
                xs[sample_index],
                mode,
                state.gradient_snapshot
            ),
            use_container_width=True,
            key="network"
        )

        section(
            "Sample Predictions",
            "Live output of your autograd model"
        )

        for i, (p, y) in enumerate(
            zip(pred, ys)
        ):

            a, b, c = st.columns(
                [1.3, 1, 1]
            )

            a.caption(f"Sample {i + 1}")

            b.markdown(
                f"**{p:+.4f}**"
            )

            c.caption(
                f"Target {y:+.1f}"
            )

    # ======================================
    # ANALYTICS
    # ======================================

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        section(
            "Optimization History",
            "Sum of squared errors"
        )

        st.plotly_chart(
            history_figure(
                state.loss_history,
                CYAN,
                "Loss",
                log_loss
            ),
            use_container_width=True,
            key="loss"
        )

    with c2:

        section(
            "Gradient Dynamics",
            "L2 norm of all parameter gradients"
        )

        st.plotly_chart(
            history_figure(
                state.grad_history,
                PURPLE,
                "Gradient norm"
            ),
            use_container_width=True,
            key="gradients"
        )

    # ======================================
    # AUTO REFRESH
    # ======================================

    if state.running:

        time.sleep(delay / 1000)
        st.rerun()


if __name__ == "__main__":
    main()
