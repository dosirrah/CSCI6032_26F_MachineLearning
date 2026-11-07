"""Step through constant-learning-rate gradient descent on a 2-D surface.

    python3 gradient_descent_walk.py --surface swirl --alpha 0.5 --seed 7

Keys:
    right / space           one step forward
    left / backspace        one step backward
    page down / page up     100 steps forward / backward
    home                    back to the starting point
    q / escape              quit

The same --seed always gives the same starting point, so every experiment is
exactly repeatable.  Run with --list to see the available surfaces.

Run with -h for suggested experiments on each surface.
"""
import argparse

# Shown by -h.  Every value here was checked by running it.
EXPERIMENTS = """\
suggested experiments (all minimize f; same seed = same start = same path):

  paraboloid  f = x^2 + y^2; each step multiplies theta by (1 - 2 alpha)
    --alpha 0.1                      smooth, steady approach to the minimum
    --alpha 0.5                      lands exactly on the minimum in ONE step
    --alpha 0.9                      damped oscillation: flips sides, shrinking 20% per step
    --alpha 1.0                      oscillates between two points forever
    --alpha 1.05                     overshoots by more each step and diverges

  elongated   x shrinks by (1 - alpha), y by (1 - 20 alpha) each step
    --alpha 0.04 --seed 3            slides smoothly into the valley, then along it
    --alpha 0.09 --seed 3            damped zig-zag across the valley
    --alpha 0.1 --seed 3             bounces across the valley forever while x converges
    --alpha 0.105 --seed 3           bounces grow: diverges along the steep direction only

  swirl       swirling ripples around one minimum at (0, 0)
    --alpha 0.2 --seed 1             drops into an arm's trough, then crawls ~half a turn along
                                     it to the minimum: the trough is nearly flat, so this takes
                                     ~1000 steps (page down ~10 times); alpha 0.1 takes ~2100
    --alpha 0.35 --seed 1            hops inward across the arms: reaches the minimum in 12 steps
    --alpha 0.5 --seed 1             same start: locks onto one arm per step and walks away
    --alpha 0.45 --seed 1            walks out, then reverses and hops back in arm by arm
    --alpha 0.6 --seed 60            large out-and-back swings for ~650 steps, then caught
    --alpha 0.69 --seed 82           chaotic jitter that stays in a ring, never settling

  meander     river-like valley, wavelength 2, on a steady downhill slope
    --alpha 2.0                      step = one wavelength: path freezes on a straight line
    --alpha 2.04                     slow beat, period ~138 steps (page down to see it)
    --alpha 2.08   (or 1.92)         beat, period ~68 steps
    --alpha 2.16                     beat, period ~32 steps
    --alpha 2.24                     faster beat, period ~20 steps

  multimodal  five wells; well 0 at (-2, -1.5) is the global minimum
    --alpha 0.3 --seed 0             finds the global minimum (also seeds 1, 6, 7)
    --alpha 0.3 --seed 2             stuck in the shallow central well (also seed 9)
    --alpha 0.3 --seed 3             stuck in the well at (1.8, 1.6)
    --alpha 0.3 --seed 4             stuck in the well at (2.2, -2.0) (also seed 5)
    --alpha 0.3 --seed 8             stuck in the well at (-1.5, 2.2)
    --alpha 1.5 --seed 0             reaches the global well but cycles across it forever
    --alpha 3.0 --seed 0             chaotic: wanders between wells, never settles

  rosenbrock  banana-shaped valley, minimum at (1, 1)
    --alpha 0.001 --seed 2           crawls along the valley floor; page down many times
    --alpha 0.0005 --seed 2          half the rate: visibly slower crawl
    --alpha 0.0025 --seed 2          stalls in a two-point cycle across the valley, short of (1, 1)

  himmelblau  four minima of equal depth: (3, 2), (-2.81, 3.13), (-3.78, -3.28), (3.58, -1.85)
    --alpha 0.01 --seed 3            reaches (3, 2)
    --alpha 0.01 --seed 2            reaches (-2.81, 3.13)
    --alpha 0.01 --seed 0            reaches (-3.78, -3.28)
    --alpha 0.01 --seed 4            reaches (3.58, -1.85)
    --alpha 0.02 --seed 0            never settles: cycles through 6 points near a minimum
    --alpha 0.02 --seed 4            cycles between 2 points instead of converging
    --alpha 0.03 --seed 3            4-point cycle;  --alpha 0.03 --seed 0 diverges
"""

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Surfaces.  Each one is a cost function f(x, y) to be *minimized*, plus the
# region to draw, where random starting points come from, and a learning rate
# that shows the surface well.
# ---------------------------------------------------------------------------

def paraboloid(x, y):
    return x**2 + y**2


def elongated(x, y):
    # A paraboloid stretched along x: the curvature along y is 20 times larger,
    # so large steps zig-zag across the valley.
    return 0.5 * x**2 + 10 * y**2


def swirl(x, y):
    # The swirling ripples from spiral_gradient_ascent.py, turned upside down so
    # the peak becomes a minimum.
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    D = 1 - np.exp(-(r / 1.2) ** 4)
    return -(0.9 * np.exp(-r**2 / (2 * 0.7**2)) - 0.069 * r
             + 0.338 * D * np.cos(2 * np.pi * r - 3 * theta))


# (center x, center y, depth, width) of each well in the multimodal surface
WELLS = [(-2.0, -1.5, 1.2, 0.9),    # the global minimum
         (1.8, 1.6, 0.9, 0.8),
         (2.2, -2.0, 0.7, 0.6),
         (-1.5, 2.2, 0.6, 0.7),
         (0.3, 0.2, 0.45, 0.5)]


def multimodal(x, y):
    # Several Gaussian wells of different depths in a gently rising bowl.
    f = 0.03 * (x**2 + y**2)
    for cx, cy, depth, width in WELLS:
        f = f - depth * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width**2))
    return f


def rosenbrock(x, y):
    # A long, curved, nearly flat valley; the minimum is at (1, 1).
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def himmelblau(x, y):
    # Four minima of equal depth: which one you reach depends on the start.
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


# A valley that meanders like a river, y = MEANDER_A sin(2 pi x / MEANDER_WAVELENGTH),
# on a steady downhill slope in x.  With a step close to one wavelength, each step
# lands a little later (or earlier) in the meander's cycle, so the path traces a
# slow wave across the valley: a beat between the step length and the wavelength.
MEANDER_A = 0.6
MEANDER_WAVELENGTH = 2.0


def meander_center(x):
    return MEANDER_A * np.sin(2 * np.pi * x / MEANDER_WAVELENGTH)


def meander(x, y):
    return -x + 0.1 * (y - meander_center(x)) ** 2


SURFACES = {
    #  name          function     (xmin, xmax, ymin, ymax)  start radius range  alpha   contour spacing
    "paraboloid": dict(f=paraboloid, view=(-3, 3, -3, 3), start=(1.5, 2.8), alpha=0.1, levels="linear",
                       about="round bowl, one minimum at (0, 0); diverges for alpha > 1"),
    "elongated":  dict(f=elongated, view=(-4, 4, -2, 2), start=(1.5, 3.5), alpha=0.09, levels="linear",
                       about="stretched bowl; zig-zags as alpha approaches 0.1, diverges beyond it"),
    "swirl":      dict(f=swirl, view=(-9.5, 9.5, -9.5, 9.5), start=(1.5, 3.0), alpha=0.1, levels="linear",
                       about="swirling ripples; small alpha crawls slowly along an arm to the minimum; alpha 0.5 hops outward arm by arm (try seed 1)"),
    "multimodal": dict(f=multimodal, view=(-4, 4, -4, 4), start=(0.5, 3.5), alpha=0.3, levels="linear",
                       about="five wells; seed 0 finds the global minimum, seeds 2, 3, 4 and 8 get stuck in local ones"),
    "rosenbrock": dict(f=rosenbrock, view=(-2, 2, -1, 3), start=(0.8, 1.8), alpha=0.001, levels="log",
                       about="banana-shaped valley, minimum at (1, 1); needs a tiny alpha and many steps"),
    "himmelblau": dict(f=himmelblau, view=(-5, 5, -5, 5), start=(0.5, 4.5), alpha=0.01, levels="log",
                       about="four equally deep minima; seeds 0, 2, 3 and 4 each reach a different one"),
    "meander":    dict(f=meander, view=(-2, 160, -1.6, 1.6), start_box=(0, 1, -0.8, 0.8), alpha=2.16,
                       levels="linear", aspect="auto", centerline=meander_center,
                       about="meandering valley, wavelength 2: a beat whose period shrinks as alpha moves "
                             "away from 2 (alpha 2.08 or 1.92: ~68 steps, 2.16: ~32, 2.24: ~20; exactly 2 freezes)"),
}


def gradient(f, p, h=1e-6):
    """Central-difference gradient of f at the point p = (x, y)."""
    x, y = p
    return np.array([(f(x + h, y) - f(x - h, y)) / (2 * h),
                     (f(x, y + h) - f(x, y - h)) / (2 * h)])


def random_start(surface, seed):
    """A starting point determined entirely by the seed: a random direction, at a
    random distance within the surface's start range from the center of the view."""
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = surface["view"]
    if "start_box" in surface:                  # uniform in a box instead
        bx0, bx1, by0, by1 = surface["start_box"]
        return np.array([rng.uniform(bx0, bx1), rng.uniform(by0, by1)])
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
    rmin, rmax = surface["start"]
    angle = rng.uniform(0, 2 * np.pi)
    radius = rng.uniform(rmin, rmax)
    p = center + radius * np.array([np.cos(angle), np.sin(angle)])
    return np.clip(p, [xmin, ymin], [xmax, ymax])


class GradientDescentWalk:
    """Gradient descent with a constant learning rate, Theta := Theta - alpha grad f(Theta).
    Keeps every point visited, so stepping backward retraces the exact path."""

    def __init__(self, f, start, learning_rate):
        self._f = f
        self._learning_rate = learning_rate
        self._theta_history = [np.array(start, dtype=float)]

    @property
    def theta(self):
        return self._theta_history[-1]

    @property
    def step(self):
        return len(self._theta_history) - 1

    @property
    def path(self):
        return np.array(self._theta_history)

    def diverged(self):
        return not np.all(np.isfinite(self.theta)) or np.max(np.abs(self.theta)) > 1e6

    def update(self):
        if self.diverged():                     # nothing sensible left to compute
            return self.theta
        theta = self.theta - self._learning_rate * gradient(self._f, self.theta)
        self._theta_history.append(theta)
        return theta

    def update_backward(self):
        if len(self._theta_history) > 1:
            self._theta_history.pop()
        return self.theta


class Viewer:
    def __init__(self, name, surface, walk, alpha, seed):
        self.walk = walk
        self.f = surface["f"]
        xmin, xmax, ymin, ymax = surface["view"]

        # Free the keys this viewer uses from matplotlib's built-in shortcuts
        # (left/right/backspace = view history, home = reset view)
        for keymap in ("keymap.back", "keymap.forward", "keymap.home"):
            plt.rcParams[keymap] = []

        self.fig, self.ax = plt.subplots(figsize=(9, 8.5))
        xs = np.linspace(xmin, xmax, 500)
        ys = np.linspace(ymin, ymax, 500)
        X, Y = np.meshgrid(xs, ys)
        Z = self.f(X, Y)
        if surface["levels"] == "log":
            levels = np.geomspace(max(Z.min(), 1e-3), Z.max(), 40)
            self.ax.contourf(X, Y, Z, levels=levels, cmap="viridis_r", norm="log")
            self.ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
        else:
            self.ax.contourf(X, Y, Z, levels=40, cmap="viridis_r")
            self.ax.contour(X, Y, Z, levels=40, colors="k", linewidths=0.3, alpha=0.4)

        start = walk.theta
        self.ax.plot(*start, "o", color="orange", ms=10, mec="k", zorder=5, label="start")
        (self.path_line,) = self.ax.plot([], [], "-", color="red", lw=1.5, zorder=6, label="path")
        (self.path_dots,) = self.ax.plot([], [], ".", color="red", ms=5, zorder=6)
        (self.current,) = self.ax.plot([], [], "o", color="red", ms=10, mec="white", mew=1.5, zorder=7,
                                       label=r"current $\theta_t$")
        # Arrow from the current point to where the next step will land
        self.next_step = self.ax.annotate("", xy=start, xytext=start, zorder=8,
                                          arrowprops=dict(arrowstyle="->", color="white", lw=2))
        self.info = self.ax.text(0.02, 0.02, "", transform=self.ax.transAxes, fontsize=10,
                                 family="monospace", va="bottom",
                                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect(surface.get("aspect", "equal"))
        self.ax.set_xlabel(r"$\theta_1$")
        self.ax.set_ylabel(r"$\theta_2$")
        self.ax.set_title(f"Gradient descent on the {name} surface, "
                          fr"constant $\alpha$ = {alpha}, seed = {seed}")
        if "centerline" in surface:
            fine = np.linspace(xmin, xmax, 20000)
            self.ax.plot(fine, surface["centerline"](fine), "--", color="white", lw=1, alpha=0.8,
                         label="valley floor", zorder=4)
        self.ax.legend(loc="upper right", fontsize=9)

        self.fig.text(0.5, 0.01, "→ / space: step     ←: back     page down / page up: ±100 steps"
                                 "     home: restart     q: quit", ha="center", fontsize=9, color="gray")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.redraw()

    def move(self, n):
        for _ in range(abs(n)):
            if n > 0:
                self.walk.update()
            else:
                self.walk.update_backward()
        self.redraw()

    def on_key(self, event):
        print(f"Key pressed: {event.key}")
        if event.key in ("escape", "q"):
            plt.close(self.fig)
        elif event.key in ("left", "backspace"):
            self.move(-1)
        elif event.key in ("right", " "):
            self.move(+1)
        elif event.key == "pagedown":
            self.move(+100)
        elif event.key == "pageup":
            self.move(-100)
        elif event.key == "home":
            self.move(-self.walk.step)

    def on_click(self, _event):
        self.move(+1)

    def redraw(self):
        path = self.walk.path
        theta = self.walk.theta
        self.path_line.set_data(path[:, 0], path[:, 1])
        self.path_dots.set_data(path[:, 0], path[:, 1])
        self.current.set_data([theta[0]], [theta[1]])

        if self.walk.diverged():
            self.next_step.set_visible(False)
            status = "DIVERGED: alpha is too large for this surface"
        else:
            g = gradient(self.f, theta)
            nxt = theta - self.walk._learning_rate * g
            self.next_step.set_visible(True)
            self.next_step.xy = nxt
            self.next_step.set_position(theta)
            status = (f"f(theta)    = {self.f(*theta): .6g}\n"
                      f"|grad f|    = {np.linalg.norm(g):.4g}\n"
                      f"step length = {np.linalg.norm(nxt - theta):.4g}")
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            if not (xmin <= theta[0] <= xmax and ymin <= theta[1] <= ymax):
                status += "\n(current point is outside the view)"
        self.info.set_text(f"step {self.walk.step}\n"
                           f"theta = ({theta[0]: .4g}, {theta[1]: .4g})\n" + status)
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(
        description="Step through constant-learning-rate gradient descent on a 2-D surface.",
        epilog=EXPERIMENTS, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--surface", choices=SURFACES, default="paraboloid", help="surface to descend")
    parser.add_argument("--alpha", type=float, default=None,
                        help="constant learning rate (default: a value suited to the surface)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for the starting point; the same seed gives the same start")
    parser.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"), default=None,
                        help="explicit starting point (overrides --seed)")
    parser.add_argument("--list", action="store_true", help="list the surfaces and exit")
    args = parser.parse_args()

    if args.list:
        for name, s in SURFACES.items():
            print(f"{name:11s} default alpha {s['alpha']:<6g} {s['about']}")
        return

    surface = SURFACES[args.surface]
    alpha = surface["alpha"] if args.alpha is None else args.alpha
    start = np.array(args.start) if args.start is not None else random_start(surface, args.seed)
    print(f"surface={args.surface}  alpha={alpha}  seed={args.seed}  start=({start[0]:.4f}, {start[1]:.4f})")

    walk = GradientDescentWalk(surface["f"], start, alpha)
    # Keep a reference: matplotlib holds only weak references to the viewer's
    # key and mouse handlers, so an unreferenced Viewer would be garbage
    # collected and the keys would silently do nothing.
    viewer = Viewer(args.surface, surface, walk, alpha, args.seed)
    plt.show()
    return viewer


if __name__ == "__main__":
    main()
