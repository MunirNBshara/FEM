import torch





'''
Symmetric quadrature lookup table. Given degree returns weights and points up to degree 6.

Triangle is isomorphic to ZZ3 and thus has 3 independent elements.
We call these elements orbits. In barycentric coordinates:
(n0) (1/3, 1/3, 1/3) -> 1 point
(n1) (a, a, 1-2a) -> 3 points
(n2) (a, b, 1 - a - b) -> 6 points
'''
def dunavant_rules(degree: int = 1):
    table = {
        1: [
            ("n0", 0.5),
        ],
        2: [
            ("n1", 1/6, 1/6),
        ],
        3: [
            ("n0", -9/32),
            ("n1", 1/5, 25/96),
        ],
        4: [
            ("n1", 0.445948490915965, 0.111690794839005),
            ("n1", 0.091576213509771, 0.054975871827661),
        ],
        5: [
            ("c", 9/80),
            ("n1", 0.470142064105115, 0.066197076394253),
            ("n1", 0.101286507323456, 0.062969590272414),
        ],
        6: [
            ("n1", 0.249286745170910, 0.058393137863189),
            ("n1", 0.063089014491502, 0.025422453185103),
            (
                "n2",
                0.053145049844816,
                0.310352451033785,
                0.041425537809187,
            ),
        ],
    }

    if degree not in table:
        raise ValueError("Available Dunavant degrees are 1 through 6.")
    barycentric_points = []
    quadrature_weights = []

    for entry in table[degree]:
        orbit_type = entry[0]
        if orbit_type == "n0":
            weight = entry[1]

            orbit = [(1/3, 1/3, 1/3)]

        elif orbit_type == "n1":
            a, weight = entry[1:]
            b = 1.0 - 2.0 * a

            orbit = [(a, a, b), (a, b, a), (b, a, a)]

        elif orbit_type == "n2":
            a, b, weight = entry[1:]
            c = 1.0 - a - b

            orbit = [
                (a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a),
            ]

        else:
            raise RuntimeError(f"Unknown orbit type: {orbit_type}")
        barycentric_points.extend(orbit)
        quadrature_weights.extend([weight] * len(orbit))

    barycentric_points = torch.tensor(
        barycentric_points,
    )

    quadrature_weights = torch.tensor(
        quadrature_weights,
    )
    return barycentric_points, quadrature_weights
'''

Investigated generation on the fly but seems challenging.
Come back to it later.



'''
'''
Minimum moment generation
'''
def minimal_moments(degree: int = 1):
    upper = degree // 3
    
    moments = 0
    for _ in range(upper+1):
        moments+=(degree - 3*upper) // 2 + 1

    return moments

'''
Triangle is isomorphic to ZZ3 and thus has 3 independent elements.
We call these elements orbits. In barycentric coordinates:
(n_0) (1/3, 1/3, 1/3) -> 1 point
(n_1) (a, a, 1-2a) -> 3 points
(n_2) (a, b, 1 - a - b) -> 6 points

min n_0 + 3*n_1 + 6*n_2
s.t.
n_0 + 2*n_1 + 3*n_2 >= moments
n_0 in {0,1}
n_1 >= 0
n_2 >= 0
if moments < 6 n_2==0
if moments >=6 n_2>=1
'''
def generate_orbit_choice(moments: int = 1):

    return moments



'''
Generate Hammer-Stroud quadrature points and weights
Hammer-Stroud are symmetric!
'''
def generate_hammer_stroud_quadrature_2d(degree: int = 1, minimal: bool = True):
    moments = degree*2
    if minimal:
        moments = minimal_moments(degree)
    generate_orbit_choice(moments)

    return 1
