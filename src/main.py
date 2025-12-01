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
def _(pl, px):
    def run_logistic_map(r: float, x0: float, iterations: int) -> pl.DataFrame:
        if not 0 <= x0 <= 1:
            raise ValueError("Initial value x0 must be between 0 and 1.")

        iteration_list = list(range(iterations + 1))
        x_n_list = [x0]

        current_x = x0

        for _ in range(iterations):
            next_x = r * current_x * (1 - current_x)
            x_n_list.append(next_x)
            current_x = next_x

        df = pl.DataFrame({
            'Iteration': iteration_list,
            'X_n': x_n_list,
        })
        return df

    def plot_logistic_map(df: pl.DataFrame, r: float, x0: float):
        fig = px.line(
            df,
            x='Iteration',
            y='X_n',
            title=f"Logistic Map Iterations (r={r}, x₀={x0})",
            labels={'X_n': 'Population Ratio (X_n)', 'Iteration': 'Iteration (n)'}
        )

        fig.update_traces(
            mode='lines+markers',
            marker=dict(size=4),
            line=dict(width=1)
        )

        fig.update_layout(
            xaxis_title="Iteration (n)",
            yaxis_title="Xn",
            font=dict(family="Inter, sans-serif"),
            hovermode="x unified",
            template="plotly_white"
        )

        fig.update_yaxes(range=[0, 1])

        fig.show()


    R_PARAM = 3.8282
    X0_PARAM = 0.5
    NUM_ITERATIONS = 150


    results_df = run_logistic_map(
        r=R_PARAM,
        x0=X0_PARAM,
        iterations=NUM_ITERATIONS
    )


    plot_logistic_map(results_df, R_PARAM, X0_PARAM)

    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(np, pl, px):
    def calculate_lyapunov_exponent(r: float, x0: float = 0.5, n_iterations: int = 10000, n_transient: int = 100) -> float:
        """
        Calculate the Lyapunov exponent for the logistic map f(x) = r*x*(1-x).
    
        Args:
            r: The growth rate parameter
            x0: Initial condition (default 0.5)
            n_iterations: Number of iterations to sum over
            n_transient: Number of initial iterations to discard (let system settle)
    
        Returns:
            The Lyapunov exponent λ
        """
        x = x0
    
        # Transient iterations to let the system settle
        for _ in range(n_transient):
            x = r * x * (1 - x)
    
        # Calculate Lyapunov exponent
        lyapunov_sum = 0.0
        for _ in range(n_iterations):
            # Derivative f'(x) = r(1 - 2x)
            derivative = abs(r * (1 - 2 * x))
            if derivative > 0:
                lyapunov_sum += np.log(derivative)
            else:
                # If derivative is 0, we hit a critical point - return -inf
                return float('-inf')
            # Iterate the map
            x = r * x * (1 - x)
    
        return lyapunov_sum / n_iterations

    # Calculate Lyapunov exponent for r in range [3, 4] with step 0.01
    r_values_lyapunov = np.arange(3.0, 4.01, 0.01)
    lyapunov_values = [calculate_lyapunov_exponent(r) for r in r_values_lyapunov]

    # Create DataFrame with Polars
    lyapunov_df = pl.DataFrame({
        'r': r_values_lyapunov,
        'lyapunov_exponent': lyapunov_values
    })

    # Plot with Plotly
    lyapunov_fig = px.line(
        lyapunov_df,
        x='r',
        y='lyapunov_exponent',
        title='Lyapunov Exponent λ for Logistic Map f(x) = rx(1-x)',
        labels={'r': 'Growth Rate (r)', 'lyapunov_exponent': 'Lyapunov Exponent (λ)'}
    )

    # Add horizontal line at λ = 0 (boundary between order and chaos)
    lyapunov_fig.add_hline(y=0, line_dash="dash", line_color="red", 
                            annotation_text="λ = 0 (chaos threshold)")

    lyapunov_fig.update_layout(
        xaxis_title="Growth Rate (r)",
        yaxis_title="Lyapunov Exponent (λ)",
        font=dict(family="Inter, sans-serif"),
        template="plotly_white",
        height=500,
        width=800
    )

    lyapunov_fig.update_traces(line=dict(width=1))

    lyapunov_fig.show()

    # Display summary statistics
    print(f"Lyapunov Exponent Statistics:")
    print(f"Min λ: {min(lyapunov_values):.4f} at r = {r_values_lyapunov[np.argmin(lyapunov_values)]:.2f}")
    print(f"Max λ: {max(lyapunov_values):.4f} at r = {r_values_lyapunov[np.argmax(lyapunov_values)]:.2f}")
    print(f"\nChaotic regions (λ > 0): {sum(1 for l in lyapunov_values if l > 0)} out of {len(lyapunov_values)} r values")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
