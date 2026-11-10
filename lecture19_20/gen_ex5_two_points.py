"""Two points and a starting decision boundary that is aligned but offset.

Class 0 is at (1, 1) and class 1 at (2, 2).  Both lie on the line through the
origin along the unit normal n = (1, 1)/sqrt(2).  An SGD step changes theta by
a multiple of (1, x), so W only ever gains multiples of (1, 1): if W starts
along n it stays along n.  The boundary never rotates; only b and ||W|| change.

Placing the starting boundary: choose the unit normal n, the signed distance d
of the boundary from the origin along n, and the length s = ||W||.  Then

    W = s n,    b = -d s,

because the closest point q = d n satisfies W^T q + b = s d - d s = 0.
Here d = 0.5 (well short of the points, at distances 1.41 and 2.83) and
s = 1, rounded to the friendlier W = (0.7, 0.7), b = -0.5 (s = 0.99, d = 0.505).

The initial theta is written as the CSV's first line, "# theta: -0.5 0.7 0.7",
and logistic_regression_sgd.py and logistic_regression_sgd_sigmoid.py start
from it unless given --theta:

    python logistic_regression_sgd_sigmoid.py ex5_two_points.csv

Other programs reading the file need to skip the comment, as in
pd.read_csv(..., comment='#').

The updates alternate between the two points, so the boundary bobs back and
forth.  The class-0 point starts on the wrong side, and its update shrinks
||W|| and lowers b, pushing the boundary out toward and past (1, 1).  The
class-1 update grows ||W|| and raises b, pulling the boundary back toward the
origin.  The swings shrink as ||W|| grows (and as sigma(z) nears each label),
and on separable data ||W|| grows without bound, so the boundary slowly
settles into the gap between the points.
"""
import numpy as np
import pandas as pd

X = np.array([[1.0, 1.0],
              [2.0, 2.0]])
y = np.array([0.0, 1.0])

# Starting boundary: unit normal n, distance d along n, ||W|| = s (see above).
n = np.array([1.0, 1.0]) / np.sqrt(2)
d, s = 0.5, 1.0
W = np.round(s * n, 1)          # (0.7, 0.7)
b = round(-d * s, 1)            # -0.5
theta = [b, *W]
print(f"theta = {theta}: boundary at distance {-b / np.linalg.norm(W):.3f} along n")

data = pd.DataFrame(X, columns=['$x_1$', '$x_2$'])
data['target'] = y
with open("ex5_two_points.csv", "w") as f:
    f.write("# theta: " + " ".join(f"{t:g}" for t in theta) + "\n")
    data.to_csv(f, index=False)
print("Two-point dataset saved as 'ex5_two_points.csv'.")
