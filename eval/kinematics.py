"""Approximate knee kinematics on the built parts: flexion on the femoral J-curve, with
the femoral rollback and axial rotation of a published in-vivo profile.

Private (patient-derived parts). A knee does not flex about a fixed hinge. Each femoral
condyle rolls and slides on the tibia, the lateral condyle rolling back more than the
medial (a medial pivot), so the femur also rotates externally on the tibia as it flexes.
The model, in the femoral component's CAD frame (+X anterior, +Y patient-left, +Z
proximal):

  1. the tibia turns by the flexion angle about the component's ML axis;
  2. it rises or drops along its own axis until it touches the lower of the two condyles
     (each condyle's sagittal profile through its centre, from the built component), so
     the joint stays in contact on the component's own J-curve, with no gap and no
     overlap; the other condyle's lift-off is reported;
  3. it slides antero-posteriorly and turns about its own axis so that each condyle's
     contact point sits where the profile puts it: the contact's AP position on the tibia
     at extension plus that condyle's AP translation at this angle.

With PUBLISHED the translations are the medial and lateral curves of the author's figure
(Flexion rotation and translation.png, Figs. 7-8: fluoroscopy of a TKA, means); the axial
rotation then follows from the two translations and this knee's condyle spacing, and is
reported against the figure's femur/tibia curve. With "fixed" both translations are zero:
the contacts stay put on the tibia and the femur only rolls on its J-curve.

The varus-valgus stress, drawer and distraction of the trial are applied on top, at rest.
"""

import numpy as np

# Figure 7-8 means, digitised at 10 degrees (mm, deg; AP anterior +, rotation external +).
PUBLISHED = {
    "flexion_deg": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    "medial_ap_mm": [-2.7, -4.2, -3.6, -3.1, -2.6, -1.2, -0.6, 0.8, 1.0, 0.6, 0.0, -1.3, -2.7],
    "lateral_ap_mm": [-9.0, -8.3, -8.5, -8.9, -9.6, -10.1, -10.4, -10.5, -11.4, -12.3, -13.3, -14.1, -14.7],
    "femur_tibia_rotation_deg": [7.6, 6.5, 7.2, 8.0, 9.4, 11.5, 13.5, 14.8, 16.0, 15.8, 17.0, 16.8, 16.8],
    "source": "author's figure (Figs. 7-8, in-vivo fluoroscopy of a TKA, means)",
}
MODELS = ("published", "fixed")
SLICE_MM = 2.5          # half-width of each condyle's slice of the component


def rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def translations(flexion_deg, model):
    """Each condyle's AP translation at this angle, relative to extension (mm)."""
    if model == "fixed":
        return 0.0, 0.0
    f = PUBLISHED["flexion_deg"]
    med = np.interp(flexion_deg, f, PUBLISHED["medial_ap_mm"]) - PUBLISHED["medial_ap_mm"][0]
    lat = np.interp(flexion_deg, f, PUBLISHED["lateral_ap_mm"]) - PUBLISHED["lateral_ap_mm"][0]
    return float(med), float(lat)


class Knee:
    """One knee's parts in the femoral CAD frame: the condyle profiles and the tibia's
    rest frame. `comp`: femoral component vertices (CAD, mm); `condyle_y`: {"medial": y,
    "lateral": y} (CAD, mm); F, T: femoral and tibial component poses (world, mm)."""

    def __init__(self, comp, condyle_y, F, T):
        self.F = np.asarray(F, float)
        self.T = np.asarray(T, float)
        K0 = np.linalg.inv(self.F) @ self.T                 # the tibial frame in CAD
        self.o0 = K0[:3, 3]
        u0 = K0[:3, 2] / np.linalg.norm(K0[:3, 2])          # tibial axis (tray normal)
        a0 = np.array([1.0, 0.0, 0.0]) - u0[0] * u0         # CAD anterior, on the tray
        a0 /= np.linalg.norm(a0)
        self.B0 = np.column_stack([a0, np.cross(u0, a0), u0])   # anterior, ML, proximal
        self.clouds = {}
        for name, y in condyle_y.items():
            sel = comp[np.abs(comp[:, 1] - y) < SLICE_MM]
            if len(sel) < 20:
                raise ValueError(f"No component surface at the {name} condyle (y {y:.1f})")
            self.clouds[name] = sel
        lat = 1.0 if condyle_y["lateral"] > condyle_y["medial"] else -1.0
        self.lateral = np.array([0.0, lat, 0.0])            # the femur's lateral direction
        self.rest = {n: self.support(n, u0) for n in self.clouds}
        d = {n: self.B0.T @ (p - self.o0) for n, p in self.rest.items()}  # (AP, ML, up)
        self.contact0 = d

    def support(self, name, u):
        """The condyle's lowest point along the tibial axis u: where it meets the tibia.
        At the lowest height, but placed at the mean of the surface points within
        0.3 mm of it, so the contact glides instead of hopping between mesh vertices."""
        c = self.clouds[name]
        h = c @ u
        low = h.min()
        mean = c[h < low + 0.3].mean(axis=0)
        return mean - (mean @ u - low) * u

    def frame(self, flexion_deg, model="published"):
        """The tibia's rigid motion in CAD (4x4) at this angle, and its readings."""
        R = rot_y(flexion_deg)
        B = R @ self.B0                                     # flexed axes: a, m, u
        a, m, u = B[:, 0], B[:, 1], B[:, 2]
        base = R @ self.o0
        dm, dl = translations(flexion_deg, model)
        p = {n: self.support(n, u) for n in self.clouds}
        c0 = self.contact0
        target = {"medial": c0["medial"][0] + dm, "lateral": c0["lateral"][0] + dl}
        # Contacts in the flexed, unslid tibial axes (AP, ML), relative to its origin.
        q = {n: np.array([(p[n] - base) @ a, (p[n] - base) @ m]) for n in p}
        v = q["lateral"] - q["medial"]
        L = np.linalg.norm(v)
        want_ap = target["lateral"] - target["medial"]
        psi = np.arctan2(v[1], v[0])
        # The contact-to-contact vector, turned by -phi, must have the wanted AP part; of
        # the two solutions take the one that keeps it pointing the same way in ML.
        ref = c0["lateral"][1] - c0["medial"][1]
        cands = [psi - s * np.arccos(np.clip(want_ap / L, -1, 1)) for s in (1, -1)]
        phi = min(cands, key=lambda f: abs(np.arctan2(np.sin(psi - f), np.cos(psi - f))
                                           - np.arctan2(ref, want_ap)))
        cph, sph = np.cos(phi), np.sin(phi)
        rot = lambda w: np.array([cph * w[0] - sph * w[1], sph * w[0] + cph * w[1]])
        mid_target = np.array([(target["medial"] + target["lateral"]) / 2,
                               (c0["medial"][1] + c0["lateral"][1]) / 2])
        t_am = (q["medial"] + q["lateral"]) / 2 - rot(mid_target)
        a2, m2 = cph * a + sph * m, -sph * a + cph * m      # the tibia turned about u
        # Down to the lower condyle: no gap, no overlap.
        gap = {n: (p[n] - base - t_am[0] * a - t_am[1] * m) @ u - c0[n][2] for n in p}
        t_u = min(gap.values())
        origin = base + t_am[0] * a + t_am[1] * m + t_u * u
        Bn = np.column_stack([a2, m2, u])
        M = np.eye(4)
        M[:3, :3] = Bn @ self.B0.T
        M[:3, 3] = origin - M[:3, :3] @ self.o0
        # Readings: the contacts on the tibia now, and the femur's rotation on it.
        loc = {n: Bn.T @ (p[n] - origin) for n in p}
        return M, {
            "flexion_deg": float(flexion_deg),
            "rotation_deg": self.external(Bn) - self.external(self.B0),
            "medial_ap_mm": float(loc["medial"][0] - c0["medial"][0]),
            "lateral_ap_mm": float(loc["lateral"][0] - c0["lateral"][0]),
            "lift_mm": {n: float(gap[n] - t_u) for n in gap},
            "contact_cad": {n: p[n].round(2).tolist() for n in p},
        }

    def external(self, B):
        """The femur's axial rotation on the tibia (deg, external +): how far the lateral
        end of its ML axis points back, seen down the tibial axis."""
        ap, ml = (B.T @ self.lateral)[:2]
        return float(np.degrees(np.arctan2(-ap, abs(ml))))

    def pose(self, flexion_deg, model="published"):
        """The tibial set's world pose (mm): P = F M F^-1."""
        M, info = self.frame(flexion_deg, model)
        return self.F @ M @ np.linalg.inv(self.F), info


