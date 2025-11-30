import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Logistic maps and chaos
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy import optimize
    return go, np, optimize, pl, px


@app.cell
def _(fig, np, pl, px):
    r_values = [1, 2, 3, 3.5]
    x = np.linspace(0, 1, 500)  # 500 points in [0, 1]

    # Generate data for the logistic curve
    data = []
    for r in r_values:
        y = r * x * (1 - x)
        data.extend(zip([str(r)] * len(x), x, y))

    df = pl.DataFrame(data, schema=["r", "x", "y"], orient="row")

    cobweb_cobweb_cobweb_cobweb_fig = px.line(df, x="x", y="y", color="r", title="Logistic Curve: y = r * x * (1 - x)",
                  labels={"x": "x", "y": "y", "r": "Growth Rate (r)"}, template="plotly_white")

    fig.show()
    return


@app.cell
def _(go, np):
    def calculate_logistic_map(r, num_points=100):
        """
        Calculates the points for the Logistic Map function for a given 'r'.
        y = r * x * (1 - x)

        Args:
            r (float): The growth rate parameter.
            num_points (int): The number of points to calculate between 0 and 1.

        Returns:
            tuple: (x_n, x_n_plus_1) coordinates.
        """
        # Create an array of x_n values ranging from 0 to 1
        x_n = np.linspace(0, 1, num_points)
        # Apply the Logistic Map equation
        x_n_plus_1 = r * x_n * (1 - x_n)
        return x_n, x_n_plus_1

    r_data = [
        {'r': 0.8, 'name': 'r = 0.8 (< 1)', 'color': 'rgb(148, 103, 189)'},      # Purple
        {'r': 1, 'name': 'r = 1', 'color': 'rgb(44, 160, 44)'},  # Green
        {'r': 2.5, 'name': 'r = 1.5 ', 'color': 'rgb(214, 39, 40)'}, # Red
        {'r': 3.5, 'name': 'r = 3.5 ', 'color': 'rgb(214, 39, 40)'}, # Red
        {'r': 3.9, 'name': 'r = 3.9 ', 'color': 'rgb(214, 39, 40)'}, # Red
    ]

    # Initialize the Plotly Figure
    stability_fig = go.Figure()

    # 1. Add the identity line: x_{n+1} = x_n
    stability_fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='x_{n+1} = x_n (Identity)',
        line=dict(color='black', dash='solid', width=1.5)
    ))

    for data_point in r_data:
        r_ = data_point['r']
        x_s, y_s = calculate_logistic_map(r_)
        stability_fig.add_trace(go.Scatter(
            x=x_s,
            y=y_s,
            mode='lines',
            name=data_point['name'],
            line=dict(color=data_point['color'], width=2)
        ))


    stability_fig.update_layout(
        title='Logistic Map Function Plot: $x_{n+1} = r x_n (1 - x_n)$',
        xaxis_title='x_n',
        yaxis_title='x_{n+1}',
        xaxis=dict(
            range=[0, 1.05], # Set axis limits slightly beyond [0, 1]
            zeroline=True,
            showgrid=True,
            gridcolor='#e5e7eb'
        ),
        yaxis=dict(
            range=[0, 1.05], # Set axis limits slightly beyond [0, 1]
            zeroline=True,
            showgrid=True,
            gridcolor='#e5e7eb',
            scaleanchor="x", # Crucial to enforce a square aspect ratio
            scaleratio=1      # Ensures the y-axis scale matches the x-axis scale
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="left",
            x=0.05
        ),
        font=dict(family='Inter, sans-serif', size=12, color="#333"),
        height=600,
        width=600,
        plot_bgcolor='#ffffff', # White plot background
        paper_bgcolor='#f0f4f8' # Light paper background
    )

    # Display the figure
    stability_fig.show()
    return


@app.cell
def _(go, np, optimize):

    r_qubic = 3.835
    resolution = 2000  # Resolution for plotting the curve

    # --- Functions ---
    def f(x, r):
        """The Logistic Map: f(x)"""
        return r * x * (1 - x)

    def f3(x, r):
        """The third iterate: f(f(f(x)))"""
        return f(f(f(x, r), r), r)

    def equation_to_solve(x, r):
        """Equation for fixed points of the 3rd iterate: f^3(x) - x = 0"""
        return f3(x, r) - x

    def df_qubic(x, r):
        """Derivative of f(x)"""
        return r * (1 - 2 * x)

    def df3(x, r):
        """
        Derivative of f^3(x) using the Chain Rule:
        (f(g(h(x))))' = f'(g(h(x))) * g'(h(x)) * h'(x)
        """
        val1 = x
        val2 = f(val1, r)
        val3 = f(val2, r)
    
        # Chain rule: f'(x_2) * f'(x_1) * f'(x_0)
        # Note: We use r_qubic here as per your snippet, though 'r' (the arg) would also work
        return df_qubic(val3, r_qubic) * df_qubic(val2, r_qubic) * df_qubic(val1, r_qubic)

    # --- Root Finding ---
    # We look for roots of f^3(x) - x = 0 to find intersection points.
    # We scan the domain [0, 1] to find brackets where the sign changes, then use brentq.

    x_scan = np.linspace(0, 1, 5000)
    y_scan = equation_to_solve(x_scan, r_qubic)

    roots = []
    # Identify intervals where the function crosses zero
    sign_changes = np.where(np.diff(np.sign(y_scan)))[0]

    for idx in sign_changes:
        # Bracket for the root
        a, b = x_scan[idx], x_scan[idx+1]
        try:
            # CORRECTION: changed 'r' to 'r_qubic' to match your variable definition
            root = optimize.brentq(equation_to_solve, a, b, args=(r_qubic,))
            roots.append(root)
        except ValueError:
            pass # No root found in this tiny interval (rare)

    roots = np.array(roots)

    # --- Classification (Stability) ---
    stable_points_x = []
    stable_points_y = []
    unstable_points_x = []
    unstable_points_y = []

    # Classify each fixed point based on the absolute value of the derivative (slope)
    for x_star in roots:
        slope = df3(x_star, r_qubic)
        y_star = f3(x_star, r_qubic) # Should be equal to x_star
    
        # If |slope| < 1, the fixed point is stable (attractor)
        if abs(slope) < 1:
            stable_points_x.append(x_star)
            stable_points_y.append(y_star)
        else:
            unstable_points_x.append(x_star)
            unstable_points_y.append(y_star)

    # --- Plotting ---

    # Generate curve data
    x_plot = np.linspace(0, 1, resolution)
    y_plot = f3(x_plot, r_qubic)

    fig_qubic = go.Figure()

    # 1. Identity Line (y = x)
    fig_qubic.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='y = x',
        line=dict(color='black', width=1)
    ))

    # 2. The f^3(x) Curve
    fig_qubic.add_trace(go.Scatter(
        x=x_plot, y=y_plot,
        mode='lines',
        name=f'f^3(x), r={r_qubic}',
        line=dict(color='black', width=1.5)
    ))

    # 3. Unstable Points (Open Circles)
    fig_qubic.add_trace(go.Scatter(
        x=unstable_points_x, y=unstable_points_y,
        mode='markers',
        name='Unstable Points (|slope| > 1)',
        marker=dict(
            size=10,
            color='white',
            line=dict(width=2, color='black')
        )
    ))

    # 4. Stable Points (Filled Dots)
    print("Stable points:", stable_points_x, stable_points_y)
    print("Unstable points:", unstable_points_x, unstable_points_y)

    fig_qubic.add_trace(go.Scatter(
        x=stable_points_x, y=stable_points_y,
        mode='markers',
        name='Stable Points (|slope| < 1)',
        marker=dict(
            size=10,
            color='black',
            line=dict(width=2, color='black')
        )
    ))

    # Layout settings to match the academic style of the image
    fig_qubic.update_layout(
        title=dict(text=f'Third Iterate $f^3(x)$ for r={r_qubic}', x=0.5),
        xaxis_title='x',
        yaxis_title='$f^3(x)$',
        xaxis=dict(range=[0, 1], showgrid=True, zeroline=False, mirror=True, ticks='outside', showline=True, linecolor='black'),
        yaxis=dict(range=[0, 1], showgrid=True, zeroline=False, mirror=True, ticks='outside', showline=True, linecolor='black', scaleanchor="x", scaleratio=1),
        plot_bgcolor='white',
        width=600,
        height=600,
        font=dict(family='Serif', size=14, color='black'), # Added color='black' globally
        legend=dict(
            x=0.02, 
            y=0.98, 
            bgcolor='rgba(255,255,255,0.9)', 
            bordercolor='black', 
            borderwidth=1,
            font=dict(color='black') # Explicitly set legend font color
        )
    )

    fig_qubic.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
