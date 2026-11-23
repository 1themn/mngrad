
import numpy as np
import plotly.graph_objects as go


BG = "#080c14"
PANEL = "#101722"
CYAN = "#22d3ee"
PURPLE = "#a78bfa"
GREEN = "#4ade80"
RED = "#fb7185"
GOLD = "#facc15"


# ==========================================
# LOSS FUNCTION
# ==========================================

def compute_loss(model, xs, ys):

    predictions = [model(x) for x in xs]

    loss = sum(
        (pred - target) ** 2
        for pred, target in zip(predictions, ys)
    )

    return loss


# ==========================================
# CALCULATE THE 3D LOSS LANDSCAPE
# ==========================================

def calculate_landscape(
    model,
    xs,
    ys,
    param_x=0,
    param_y=1,
    span=2.0,
    resolution=45
):

    params = model.parameters()

    if param_x == param_y:
        raise ValueError("Choose two different parameters")

    p1 = params[param_x]
    p2 = params[param_y]

    original_x = p1.data
    original_y = p2.data

    x_values = np.linspace(
        original_x - span,
        original_x + span,
        resolution
    )

    y_values = np.linspace(
        original_y - span,
        original_y + span,
        resolution
    )

    X, Y = np.meshgrid(x_values, y_values)
    Z = np.zeros_like(X)

    try:

        for i in range(resolution):

            for j in range(resolution):

                # Temporarily change only two weights
                p1.data = float(X[i, j])
                p2.data = float(Y[i, j])

                # Actual loss from your MLP
                loss = compute_loss(model, xs, ys)

                Z[i, j] = loss.data

    finally:

        # Restore the original model weights
        p1.data = original_x
        p2.data = original_y

    return X, Y, Z


# ==========================================
# ACTUAL AUTOGRAD GRADIENTS
# ==========================================

def calculate_gradients(model, xs, ys):

    params = model.parameters()

    # Preserve existing gradients
    original_grads = [
        p.grad for p in params
    ]

    try:

        for p in params:
            p.grad = 0.0

        loss = compute_loss(model, xs, ys)

        # YOUR backward implementation
        loss.backward()

        gradients = np.array([
            p.grad for p in params
        ])

        return loss.data, gradients

    finally:

        for p, grad in zip(params, original_grads):
            p.grad = grad


# ==========================================
# PARAMETER LABELS
# ==========================================

def get_parameter_labels(model):

    labels = []

    for li, layer in enumerate(model.layers):

        for ni, neuron in enumerate(layer.neurons):

            for wi, weight in enumerate(neuron.w):

                labels.append(
                    f"L{li + 1} · N{ni + 1} · W{wi + 1}"
                )

            labels.append(
                f"L{li + 1} · N{ni + 1} · Bias"
            )

    return labels


# ==========================================
# MAIN 3D VISUALIZATION
# ==========================================

def gradient_landscape_figure(
    model,
    xs,
    ys,
    param_x=0,
    param_y=1,
    span=2.0,
    resolution=45,
    history=None,
    epoch=0,
    learning_rate=0.01,
    wireframe=True,
    show_contours=True
):

    params = model.parameters()

    labels = get_parameter_labels(model)

    current_x = params[param_x].data
    current_y = params[param_y].data

    # Calculate current loss surface
    X, Y, Z = calculate_landscape(
        model,
        xs,
        ys,
        param_x,
        param_y,
        span,
        resolution
    )

    # Calculate actual gradients
    current_loss, gradients = calculate_gradients(
        model,
        xs,
        ys
    )

    gx = gradients[param_x]
    gy = gradients[param_y]

    gradient_norm = np.sqrt(gx**2 + gy**2)

    fig = go.Figure()

    # ======================================
    # THE BLANKET
    # ======================================

    fig.add_trace(

        go.Surface(

            x=X,
            y=Y,
            z=Z,

            surfacecolor=Z,

            colorscale=[
                [0.00, "#0f766e"],
                [0.15, "#06b6d4"],
                [0.35, "#2563eb"],
                [0.55, "#7c3aed"],
                [0.75, "#c026d3"],
                [1.00, "#fb7185"]
            ],

            opacity=0.94,

            lighting=dict(
                ambient=0.55,
                diffuse=0.9,
                specular=0.65,
                roughness=0.3,
                fresnel=0.25
            ),

            lightposition=dict(
                x=100,
                y=200,
                z=300
            ),

            colorbar=dict(
                title="Loss",
                thickness=12,
                len=0.65,
                tickfont=dict(
                    color="#94a3b8"
                )
            ),

            contours=dict(

                x=dict(
                    show=wireframe,
                    color="#64748b",
                    width=1
                ),

                y=dict(
                    show=wireframe,
                    color="#64748b",
                    width=1
                ),

                z=dict(
                    show=show_contours,
                    usecolormap=True,
                    highlightcolor=CYAN,
                    project_z=True
                )

            ),

            hovertemplate=(
                labels[param_x] + ": %{x:.4f}<br>"
                + labels[param_y] + ": %{y:.4f}<br>"
                "Loss: %{z:.6f}"
                "<extra></extra>"
            ),

            name="Loss landscape"

        )

    )

    # ======================================
    # CURRENT POSITION
    # ======================================

    fig.add_trace(

        go.Scatter3d(

            x=[current_x],
            y=[current_y],
            z=[current_loss],

            mode="markers",

            marker=dict(
                size=10,
                color=GOLD,
                symbol="diamond",
                line=dict(
                    color="white",
                    width=3
                )
            ),

            name="Current parameters",

            hovertemplate=(
                f"Iteration: {epoch}<br>"
                f"Weight 1: {current_x:.5f}<br>"
                f"Weight 2: {current_y:.5f}<br>"
                f"Loss: {current_loss:.6f}"
                "<extra></extra>"
            )

        )

    )

    # ======================================
    # GRADIENT DESCENT DIRECTION
    # ======================================

    if gradient_norm > 1e-12:

        # Normalize for a visible arrow
        arrow_length = span * 0.35

        dx = -gx / gradient_norm * arrow_length
        dy = -gy / gradient_norm * arrow_length

        # Directional derivative along the arrow
        dz = gx * dx + gy * dy

        # Limit visual height of the arrow
        z_range = max(float(Z.max() - Z.min()), 0.1)

        dz = np.clip(
            dz,
            -z_range * 0.35,
            z_range * 0.35
        )

        # Arrow shaft
        fig.add_trace(

            go.Scatter3d(

                x=[
                    current_x,
                    current_x + dx
                ],

                y=[
                    current_y,
                    current_y + dy
                ],

                z=[
                    current_loss,
                    current_loss + dz
                ],

                mode="lines",

                line=dict(
                    color=GREEN,
                    width=9
                ),

                name="Negative gradient"

            )

        )

        # Arrow head
        fig.add_trace(

            go.Cone(

                x=[current_x + dx],
                y=[current_y + dy],
                z=[current_loss + dz],

                u=[dx],
                v=[dy],
                w=[dz],

                anchor="tip",
                sizemode="absolute",
                sizeref=span * 0.2,

                colorscale=[
                    [0, GREEN],
                    [1, GREEN]
                ],

                showscale=False,

                name="Descent direction",

                hovertemplate=(
                    f"∂L/∂w₁: {gx:.6f}<br>"
                    f"∂L/∂w₂: {gy:.6f}"
                    "<extra></extra>"
                )

            )

        )

    # ======================================
    # NEXT GRADIENT DESCENT STEP
    # ======================================

    next_x = current_x - learning_rate * gx
    next_y = current_y - learning_rate * gy

    p1 = params[param_x]
    p2 = params[param_y]

    try:

        p1.data = next_x
        p2.data = next_y

        next_loss = compute_loss(
            model, xs, ys
        ).data

    finally:

        p1.data = current_x
        p2.data = current_y

    fig.add_trace(

        go.Scatter3d(

            x=[current_x, next_x],
            y=[current_y, next_y],
            z=[current_loss, next_loss],

            mode="lines+markers",

            line=dict(
                color=CYAN,
                width=5,
                dash="dash"
            ),

            marker=dict(
                size=[0, 5],
                color=CYAN
            ),

            name="Next projected step"

        )

    )

    # ======================================
    # OPTIMIZATION TRAJECTORY
    # ======================================

    if history and len(history) > 1:

        path_x = [
            item["parameters"][param_x]
            for item in history
        ]

        path_y = [
            item["parameters"][param_y]
            for item in history
        ]

        path_z = [
            item["loss"]
            for item in history
        ]

        fig.add_trace(

            go.Scatter3d(

                x=path_x,
                y=path_y,
                z=path_z,

                mode="lines+markers",

                line=dict(
                    color="#f8fafc",
                    width=5
                ),

                marker=dict(
                    size=3,
                    color=list(range(len(path_x))),
                    colorscale="Viridis",
                    showscale=False
                ),

                name="Optimization history",

                hovertemplate=(
                    "Weight 1: %{x:.4f}<br>"
                    "Weight 2: %{y:.4f}<br>"
                    "Loss: %{z:.6f}"
                    "<extra></extra>"
                )

            )

        )

    # ======================================
    # PROFESSIONAL 3D STYLING
    # ======================================

    axis_style = dict(

        backgroundcolor=PANEL,
        gridcolor="#293548",
        zerolinecolor="#475569",

        showbackground=True,
        showspikes=False,

        tickfont=dict(
            color="#94a3b8",
            size=10
        )

    )

    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor=BG,
        plot_bgcolor=BG,

        height=680,

        margin=dict(
            l=0,
            r=0,
            t=60,
            b=0
        ),

        title=dict(
            text=(
                f"LOSS LANDSCAPE"
                f"  ·  ITERATION {epoch}"
                f"  ·  LOSS {current_loss:.5f}"
            ),
            font=dict(
                size=14,
                color="#cbd5e1"
            ),
            x=0.03
        ),

        scene=dict(

            xaxis=dict(
                title=labels[param_x],
                **axis_style
            ),

            yaxis=dict(
                title=labels[param_y],
                **axis_style
            ),

            zaxis=dict(
                title="Total Loss",
                **axis_style
            ),

            camera=dict(
                eye=dict(
                    x=1.6,
                    y=1.65,
                    z=1.1
                )
            ),

            aspectmode="manual",

            aspectratio=dict(
                x=1,
                y=1,
                z=0.75
            ),

            uirevision="gradient-camera"

        ),

        legend=dict(
            orientation="h",
            x=0.02,
            y=0.02,
            bgcolor="rgba(8,12,20,0.7)",
            font=dict(
                color="#cbd5e1",
                size=11
            )
        ),

        uirevision="gradient-landscape",

        font=dict(
            family="Inter, Arial, sans-serif",
            color="#e2e8f0"
        )

    )

    return fig, {
        "loss": current_loss,
        "weight_x": current_x,
        "weight_y": current_y,
        "gradient_x": gx,
        "gradient_y": gy,
        "gradient_norm": gradient_norm,
        "next_x": next_x,
        "next_y": next_y,
        "next_loss": next_loss
    }
