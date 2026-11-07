import argparse

import matplotlib.pyplot as plt
import numpy as np

# A surface with a single peak at the origin, surrounded by "swirling ripples":
# spiral ridges that climb gently toward the peak.
#
#   f(x) = H exp(-r^2 / (2 w^2))  -  c r  +  a D(r) cos(k r - m theta)
#          \___ central peak ___/   \gentle/  \_______ spiral ripples _______/
#                                    slope
#
# The spiral has m = 3 interleaved arms; going straight out, you cross an arm every
# 2 pi / k = 1 unit.  D(r) fades the ripples out near the
# peak so that the origin is the global maximum.
H, W = 0.9, 0.7          # height and width of the central peak
C = 0.069                # slope of the gentle cone toward the peak
A = 0.338                # amplitude of the ripples
K = 2 * np.pi            # ripples per unit radius (arm spacing = 1)
M = 3                    # number of spiral arms


def f(x, y):
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    D = 1 - np.exp(-(r / 1.2) ** 4)
    return H * np.exp(-r**2 / (2 * W**2)) - C * r + A * D * np.cos(K * r - M * theta)


def gradient(p, h=1e-6):
    """Central-difference gradient of f at the point p = (x, y)."""
    x, y = p
    return np.array([(f(x + h, y) - f(x - h, y)) / (2 * h),
                     (f(x, y + h) - f(x, y - h)) / (2 * h)])


def gradient_ascent(p0, alpha, steps, stop_radius=None):
    """Theta := Theta + alpha * grad f(Theta).  Returns the path as an array."""
    p = np.array(p0, dtype=float)
    path = [p.copy()]
    for _ in range(steps):
        p = p + alpha * gradient(p)
        path.append(p.copy())
        if stop_radius is not None and np.hypot(*p) < stop_radius:
            break
    return np.array(path)


def main():
    parser = argparse.ArgumentParser(
        description="Gradient ascent on swirling ripples: small steps climb, large steps spiral away.")
    parser.add_argument("--small-alpha", type=float, default=0.1, help="learning rate that climbs to the peak")
    parser.add_argument("--large-alpha", type=float, default=0.5, help="learning rate that walks away")
    parser.add_argument("--large-steps", type=int, default=7, help="number of large steps to draw")
    parser.add_argument("--out", type=str, default="spiral_gradient_ascent.png", help="output image")
    args = parser.parse_args()

    start = (2 * np.cos(1.5 * np.pi), 2 * np.sin(1.5 * np.pi))   # same starting point for both runs
    small = gradient_ascent(start, args.small_alpha, 20000, stop_radius=0.02)
    large = gradient_ascent(start, args.large_alpha, args.large_steps)

    print(f"small alpha={args.small_alpha}: {len(small) - 1} steps, "
          f"f goes from {f(*small[0]):.3f} to {f(*small[-1]):.3f} (the peak is f = {f(0, 0):.3f})")
    print(f"large alpha={args.large_alpha}: {len(large) - 1} steps, "
          f"distance from peak goes from {np.hypot(*large[0]):.2f} to {np.hypot(*large[-1]):.2f}, "
          f"f goes from {f(*large[0]):.3f} to {f(*large[-1]):.3f}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.35, 1]})

    # Contour plot of the surface with both paths
    lim = 9.5
    xs = np.linspace(-lim, lim, 900)
    X, Y = np.meshgrid(xs, xs)
    Z = f(X, Y)
    ax.contourf(X, Y, Z, levels=40, cmap="viridis")
    ax.contour(X, Y, Z, levels=40, colors="k", linewidths=0.3, alpha=0.4)

    ax.plot(small[:, 0], small[:, 1], color="white", lw=2,
            label=fr"small steps, $\alpha$ = {args.small_alpha}: follows the swirl up to the peak")
    ax.plot(large[:, 0], large[:, 1], "-o", color="red", lw=2, ms=6,
            label=fr"large steps, $\alpha$ = {args.large_alpha}: jumps from arm to arm, away from the peak")
    for t, (x, y) in enumerate(large[1:], start=1):
        ax.annotate(str(t), (x, y), textcoords="offset points", xytext=(9, -4),
                    color="red", fontsize=11, fontweight="bold")
    ax.plot(*start, "o", color="orange", ms=10, mec="k", zorder=5, label="start")
    ax.plot(0, 0, "*", color="gold", ms=20, mec="k", zorder=5, label="peak (global maximum)")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title("Gradient ascent on swirling ripples")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # f along the large-step path: every step is an ascent step, yet f ends up lower
    fl = f(large[:, 0], large[:, 1])
    ax2.plot(range(len(large)), fl, "-o", color="red", lw=2, ms=7, label=fr"large steps, $\alpha$ = {args.large_alpha}")
    ax2.axhline(f(0, 0), color="goldenrod", ls="--", label="value at the peak")
    ax2.set_xlabel("step $t$")
    ax2.set_ylabel(r"$f(\theta_t)$")
    ax2.set_title("Each step follows the gradient uphill,\nyet $f$ ends up far below where it started")
    ax2.legend(loc="center right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
