import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import argparse

from mlp_core import NeuralNetwork

# Global variables for user interaction
action = None
plot_initialized = False


def on_key(event):
    global action

    print(f"Key pressed: {event.key}")

    # Exit / quit
    if event.key in ["escape", "q", "Q"]:
        action = 0  # Exit the viewer

    # Single step navigation
    elif event.key in ["left", "backspace"]:
        action = -1  # One step backward
    elif event.key in ["right", "space"]:
        action = +1  # One step forward

    # Page navigation (bigger jumps)
    elif event.key == "pagedown":
        action = +100  # Page forward
    elif event.key == "pageup":
        action = -100  # Page backward

    # Home / End for jumping to start/end
    elif event.key == "home":
        action = -10000  # Go to very beginning
    elif event.key == "end":
        action = +10000  # Go to very end
    else:
        # Any unrecognized key → default to single step forward
        action = 1


def on_click(event):
    global action
    print(f"Mouse button pressed: {event.button}")
    action = 1  # Step forward


def plot_neural_network_state(X, y, nn, step, sample_index=None):
    """
    Visualizes the state of the neural network at a given training step.

    Parameters:
    - X: Input features (numpy array of shape (n_samples, n_features))
    - y: True labels (numpy array)
    - nn: NeuralNetwork instance
    - step: Current training step
    - sample_index: Index of the current training sample (optional)
    """
    global plot_initialized

    # Compute activations at each layer
    activations_per_layer = nn.activations(X)

    # Extract activations for each layer
    input_space = X                                  # Original input space
    hidden_activations = activations_per_layer[1]  # Hidden layer activations
    output_activations = activations_per_layer[-1]        # Output layer activations

    # Predictions (thresholded at 0.5)
    predictions = (output_activations >= 0.5).astype(int).flatten()
    accuracy = np.mean(predictions == y) * 100  # in percent

    # Set up the plot
    if not plot_initialized:
        plt.ion()
        fig = plt.figure(figsize=(14, 7))

        # 2x2 grid for the four main panels
        gs_main = GridSpec(
            2, 2, figure=fig,
            left=0.05, right=0.93, bottom=0.13, top=0.93,
            wspace=0.3, hspace=0.3
        )

        axes = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                axes[i, j] = fig.add_subplot(gs_main[i, j])

        # Separate GridSpec just for the colorbar, on the right
        gs_cbar = GridSpec(
            2, 1, figure=fig,
            left=0.94, right=0.96,
            wspace=0.0,
            top=0.93, bottom=0.13
        )
        cax = fig.add_subplot(gs_cbar[:, 0])

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("button_press_event", on_click)

        # Accuracy banner
        accuracy_text = fig.text(
            0.5, 0.04, "", ha="center", va="bottom",
            fontsize=16, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray")
        )
        fig._accuracy_text = accuracy_text

        fig._axes = axes
        fig._cax = cax
        plot_initialized = True
    else:
        fig = plt.gcf()
        axes = fig._axes
        cax = fig._cax

    # Clear previous plots
    for ax_row in axes:
        for ax in ax_row:
            ax.cla()

    color_map = ListedColormap(["blue", "red"])

    # First Column - Hidden Layer Neurons (in input space)
    # Hidden Layer Neuron 1
    ax = axes[0, 0]
    plot_decision_boundary_for_single_neuron(
        input_space, y, nn.layers[0][0], ax, title="Hidden Layer Neuron 1"
    )
    if sample_index is not None:
        ax.scatter(
            input_space[sample_index, 0], input_space[sample_index, 1],
            facecolors="none", edgecolors="yellow", s=100, linewidths=2
        )

    # Hidden Layer Neuron 2
    ax = axes[1, 0]
    plot_decision_boundary_for_single_neuron(
        input_space, y, nn.layers[0][1], ax, title="Hidden Layer Neuron 2"
    )
    if sample_index is not None:
        ax.scatter(
            input_space[sample_index, 0], input_space[sample_index, 1],
            facecolors="none", edgecolors="yellow", s=100, linewidths=2
        )

    # Second Column - Output neuron (in hidden activation space) and predictions
    # Output neuron decision boundary in hidden-layer-1 activation space
    ax = axes[0, 1]
    plot_decision_boundary_for_output_neuron(
        hidden_activations, y, nn.output_neuron, ax
    )
    if sample_index is not None:
        ax.scatter(
            hidden_activations[sample_index, 0],
            hidden_activations[sample_index, 1],
            facecolors="none", edgecolors="yellow", s=100, linewidths=2
        )

    # Predictions in input space
    ax = axes[1, 1]
    scatter = plot_predictions_in_input_space(input_space, y, nn, ax, color_map)

    if sample_index is not None:
        ax.scatter(
            input_space[sample_index, 0], input_space[sample_index, 1],
            facecolors="none", edgecolors="yellow", s=100, linewidths=2
        )

    # Figure titles and labels
    fig.suptitle(
        f"Two-Layer Neural Network", fontsize=16
    )

    # Accuracy banner color by performance
    accuracy_str = f"Step: {step}  |  Accuracy: {accuracy:.2f}%  ({np.sum(predictions == y)}/{len(y)} correct)"
    fig._accuracy_text.set_text(accuracy_str)
    if accuracy >= 95.0:
        fig._accuracy_text.set_bbox(dict(facecolor="lightgreen", alpha=0.9))
    elif accuracy >= 75.0:
        fig._accuracy_text.set_bbox(dict(facecolor="lightyellow", alpha=0.9))
    else:
        fig._accuracy_text.set_bbox(dict(facecolor="lightcoral", alpha=0.9))

    if scatter is not None:
        cax.cla()
        cbar = plt.colorbar(scatter, cax=cax, orientation="vertical")
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["0", "0.5", "1"])
        cbar.ax.tick_params(labelsize=12)

    # Draw the updated plot
    plt.draw()
    plt.pause(0.01)


def plot_decision_boundary_for_single_neuron(
    X, y, neuron, ax, title="Neuron Decision Boundary"
):
    """
    Plots the decision boundary of a single neuron in the input space.

    Parameters:
    - X: Input features
    - y: True labels
    - neuron: Neuron instance
    - ax: Matplotlib axis to plot on
    - title: Title of the plot
    """
    color_map = ListedColormap(["blue", "red"])

    # Scatter plot of the data
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=color_map, edgecolor="k")

    # Create a grid to evaluate the neuron
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Evaluate the neuron on the grid
    Z = neuron.activations(grid)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary at activation 0.5
    ax.contourf(xx, yy, Z, levels=20, cmap="viridis", alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0.5], linestyles=["--"], colors=["k"])
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def plot_decision_boundary_for_output_neuron(activations, y, neuron, ax):
    """
    Plots the decision boundary of the output neuron in the activation space.

    Parameters:
    - activations: Activations from the previous layer (inputs to the output neuron)
    - y: True labels
    - neuron: Output neuron instance
    - ax: Matplotlib axis to plot on
    """
    color_map = ListedColormap(["blue", "red"])
    ax.scatter(activations[:, 0], activations[:, 1], c=y, cmap=color_map, edgecolor="k")
    a_min, a_max = activations.min(axis=0) - 0.1, activations.max(axis=0) + 0.1
    xx, yy = np.meshgrid(
        np.linspace(a_min[0], a_max[0], 200),
        np.linspace(a_min[1], a_max[1], 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = neuron.activations(grid)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=20, cmap="viridis", alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0.5], linestyles=["--"], colors=["k"])
    ax.set_title("Output Neuron in Hidden Layer Space")
    ax.set_xlabel("Activation of Hidden Neuron 1")
    ax.set_ylabel("Activation of Hidden Neuron 2")


def plot_predictions_in_input_space(X, y, nn, ax, color_map):
    """
    Plots the predictions of the neural network in the input space.

    Parameters:
    - X: Input features
    - y: True labels
    - nn: NeuralNetwork instance
    - ax: Matplotlib axis
    - color_map: Colormap for plotting
    """
    # Grid over input space
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Evaluate network on the grid
    outputs = []
    for point in grid:
        a = nn.forward(point)
        outputs.append(a)
    outputs = np.array(outputs).reshape(xx.shape)

    # Scatter plot of training data
    sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=color_map, edgecolor="k")

    # Contour plot for predictions
    contour = ax.contourf(xx, yy, outputs, levels=20, cmap="viridis", alpha=0.5)
    ax.contour(xx, yy, outputs, levels=[0.5], linestyles=["--"], colors=["k"])
    ax.set_title("Predictions in Input Space")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    return contour


def initialize_weights_for_xor(nn, nudge=0.0, init_scale=1.0, seed=None):
    """
    Initializes the neural network weights to approximate the XOR function with optional
    perturbation and scaling, for a single hidden layer with two neurons.

    Parameters:
    - nn: NeuralNetwork instance.
    - nudge: Float indicating the magnitude of perturbation to apply to each weight.
             A value of 0.0 means no perturbation.
    - init_scale: Float indicating the scaling factor for the initial weights.
                  A value <1.0 starts with smaller weights.
    - seed: Integer seed for the random number generator.
    """
    # Base weights that implement XOR very sharply for x in {0,1}^2, using the
    # network from lecture 21: two "and not" hidden neurons feeding an "or".
    # The thresholds sit at 0.5 to suit data clustered at 0 and 1 (the lecture's
    # 0.4/0.6 example uses 0.1).
    base_weights = {
        "hidden_neuron_1": np.array([-10.0, 20.0, -20.0]),  # x1 and not x2: x1 - x2 > 0.5
        "hidden_neuron_2": np.array([-10.0, -20.0, 20.0]),  # (not x1) and x2: x2 - x1 > 0.5
        "output_neuron":   np.array([-10.0, 20.0, 20.0]),   # h1 or h2: h1 + h2 > 0.5
    }

    # Apply nudge if specified
    if nudge != 0.0:
        rng = np.random.default_rng(seed)
        for key in base_weights:
            perturbation = rng.uniform(-nudge, nudge, size=base_weights[key].shape)
            base_weights[key] += perturbation

    # Apply scaling to initial weights
    if init_scale != 1.0:
        for key in base_weights:
            base_weights[key] *= init_scale

    # Set weights using the setter method for encapsulation
    nn.layers[0][0].set_weights(base_weights["hidden_neuron_1"])
    nn.layers[0][1].set_weights(base_weights["hidden_neuron_2"])
    nn.output_neuron.set_weights(base_weights["output_neuron"])


def main():
    global action

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Neural network with one hidden layer of two neurons for the XOR problem, with visualization."
    )
    parser.add_argument(
        "file", type=str,
        help="Data file (CSV format with features and labels)."
    )
    parser.add_argument(
        "--seed", type=int, default=83,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Learning rate"
    )
    parser.add_argument(
        "--answer", action="store_true",
        help="Initialize weights to (almost) solve XOR"
    )
    parser.add_argument(
        "--nudge", type=float, default=0.0,
        help="Magnitude of perturbation to apply to initial weights."
    )
    parser.add_argument(
        "--init-scale", type=float, default=1.0,
        help="Scaling factor for initial parameters. Values <1.0 start with smaller weights."
    )

    args = parser.parse_args()

    # Validate nudge factor
    if args.nudge < 0.0:
        parser.error("--nudge must be non-negative.")
    elif args.nudge > 3.0:
        print("WARNING: --nudge greater than 3.0 may be unstable.")

    # Set the seed
    if args.seed is not None:
        np.random.seed(args.seed)

    # Load data
    data = pd.read_csv(args.file)
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Labels

    # Initialize the neural network with a single hidden layer of two neurons
    learning_rate = args.alpha
    hidden_sizes = [2]
    nn = NeuralNetwork(
        input_size=X.shape[1],
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate
    )

    if args.answer:
        initialize_weights_for_xor(
            nn, nudge=args.nudge, init_scale=args.init_scale, seed=args.seed
        )
        print(f"Network weights initialized to solve XOR with nudge={args.nudge}.")

    step = 0
    i = 0

    while True:
        # Choose current sample
        x_sample = X[i]
        y_sample = y[i]

        # Visualize before each SGD step highlighting the point about to be used.
        plot_neural_network_state(X, y, nn, step=step, sample_index=i)

        # Wait for user interaction
        while not plt.waitforbuttonpress() and action is None:
            plt.pause(0.01)

        if action == 0:
            exit(0)

        # Handle user input for stepping forward or backward
        while action < 0:
            if step > 0:
                nn.update_backward()
                step -= 1
            action += 1
        else:
            while action > 0:
                # Step forward (train on one sample)
                nn.step_forward(x_sample, y_sample)
                step += 1
                action -= 1

                # Advance to next data point
                i = step % len(y)
                x_sample = X[i]
                y_sample = y[i]


if __name__ == "__main__":
    main()
    
