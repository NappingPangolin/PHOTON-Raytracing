"""
scene.py  -  branched pipeline, with front/back sided-module routing

Sided modules (BeamSplitter, DielectricMirror, Substrate, CompoundLens -
anything with ``SIDED = True``, see modules/base.py) expose two physically
distinct faces instead of one generic in/out pair:

    input_branches_front / input_branches_back
        Which incoming branch ids feed the module's front face / back face.
        A branch id may appear in at most one of the two lists (enforced in
        update_routing()) - it is either a front input or a back input, not
        both.

    output_branches = [front_out, back_out]
        front_out receives whatever exits the module's front face:
          - the reflected component of a front-input photon, and
          - the transmitted component of a back-input photon.
        back_out receives whatever exits the back face:
          - the transmitted component of a front-input photon, and
          - the reflected component of a back-input photon.

A photon fed into the back face is traced through a geometrically flipped
version of the module (last surface treated as first, stack order and/or
incidence medium reversed as appropriate for that module - see each
module's own process(photons, side=...) implementation) rather than through
the front-face geometry with the ray simply reversed.

Non-sided modules (flat mirrors, parabolic mirrors, sources, detectors,
free-propagate, gradient media, ...) are untouched: a single
``input_branches`` / ``output_branches`` pair, exactly as before.
"""
from __future__ import annotations
import json, uuid
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from optics_sim import Photon
from modules import MODULE_REGISTRY, create_module
from modules.base import OpticalModule


C_LIGHT = 299792458.0  # m/s


class BranchedPhoton:
    __slots__ = ("photon", "branch", "last_seg_id")
    def __init__(self, photon: Photon, branch: int = 0, last_seg_id: str | None = None):
        self.photon = photon
        self.branch = branch
        self.last_seg_id = last_seg_id


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):         return obj.tolist()
        if isinstance(obj, (np.integer,)):      return int(obj)
        if isinstance(obj, (np.floating,)):     return float(obj)
        if isinstance(obj, complex):            return {"real": obj.real, "imag": obj.imag}
        return super().default(obj)


def _elem_avg_n(mod) -> float:
    """Average refractive index across a multi-element stack (compound_lens /
    substrate). This is a simplification: real per-segment index would need
    to know which element a given intra-module hop belongs to, which isn't
    tracked explicitly. Good enough to get pulse travel *timing* roughly
    right (a few-% error at worst for typical multi-element glass stacks).
    Order-independent, so this is unaffected by front/back flipping."""
    n_elems = int(mod.params.get("n_elements", 1)) or 1
    vals = []
    for i in range(n_elems):
        key = f"n_{i}"
        if key in mod.params:
            try:
                vals.append(float(mod.params[key]))
            except (TypeError, ValueError):
                pass
    if not vals:
        return float(mod.params.get("n", 1.5))
    return sum(vals) / len(vals)


def _resolve_intra_n(mod, entry: dict | None, wps: list | None) -> float:
    """Refractive index to use for an intra-module (inside-glass / inside-box)
    hop. Prefers an exact effective index derived from the module's own
    reported optical path length (gradient_medium's RK4 integrator already
    computes this correctly); falls back to an element-average for lens /
    substrate stacks, and finally to a flat 'n' param."""
    if entry and isinstance(entry, dict) and "optical_path" in entry and wps and len(wps) >= 3:
        geom = 0.0
        for a, b in zip(wps[1:-1], wps[2:]):
            geom += float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))
        if geom > 1e-12:
            try:
                return max(1.0, float(entry["optical_path"]) / geom)
            except (TypeError, ValueError):
                pass
    if mod.MODULE_TYPE in ("compound_lens", "substrate"):
        return _elem_avg_n(mod)
    return float(mod.params.get("n", mod.params.get("n_0", 1.5)))


def _resolve_standard_n(mod, entry: dict | None, geom_len: float) -> float:
    """Refractive index for a plain (non-waypoint) src->out segment. Almost
    always vacuum/air (n=1), except for volumetric modules like
    plasma_gradient that bend a ray inside a single reported segment but
    still expose an accumulated optical path length we can use to recover
    an effective group index for correct pulse-timing."""
    if mod.MODULE_TYPE == "plasma_gradient" and entry and isinstance(entry, dict):
        opd = entry.get("optical_path")
        if opd is not None and geom_len > 1e-12:
            try:
                return max(1.0, float(opd) / geom_len)
            except (TypeError, ValueError):
                pass
    return 1.0


def _merge_side_infos(parts: list[tuple[str, dict]]) -> dict:
    """Combine the per-side info dicts returned by a sided module's two
    process() calls (front-input photons, back-input photons) into one dict
    for the step log / UI. List-valued diagnostic fields (reflections/rays)
    are concatenated (front entries first); count fields are summed; every
    other scalar field keeps whichever side set it first (both sides report
    the same static geometry for everything except EFL/BFL-style figures
    that are direction-dependent, in which case this simply shows whichever
    side ran first - a reasonable simplification when a module is driven
    from both faces in the same run)."""
    if not parts:
        return {}
    if len(parts) == 1:
        info = parts[0][1]
        return info if isinstance(info, dict) else {}
    merged: dict = {}
    for _side, info in parts:
        if not isinstance(info, dict):
            continue
        for k, v in info.items():
            if k in ("reflections", "rays") and isinstance(v, list):
                merged[k] = merged.get(k, []) + v
            elif k in ("n_in", "n_out", "n_blocked") and isinstance(v, (int, float)):
                merged[k] = merged.get(k, 0) + v
            elif k not in merged or merged[k] in (None, [], 0):
                merged[k] = v
    return merged


class Scene:
    def __init__(self):
        self.modules: list[OpticalModule] = []
        self.routing: dict[str, dict] = {}   # id -> routing entry, shape depends on SIDED

    @staticmethod
    def _is_sided(mod) -> bool:
        return bool(getattr(mod, "SIDED", False))

    @staticmethod
    def _input_branches(r: dict) -> list[int]:
        """Input branches for a *non-sided* module's routing entry."""
        return list(r.get("input_branches", [0]))

    @staticmethod
    def _all_input_branches(r: dict) -> list[int]:
        """Every branch feeding a module, sided or not - front+back union for
        sided modules, the plain list otherwise. Used wherever the caller
        only needs to know *which* branches are consumed, not through which
        face (e.g. the /api/scene/branches endpoint, pipeline-canvas grouping)."""
        if "input_branches_front" in r or "input_branches_back" in r:
            return list(r.get("input_branches_front", [])) + list(r.get("input_branches_back", []))
        return list(r.get("input_branches", [0]))

    # ── branch helpers ────────────────────────────────────────────────────────

    def _max_branch(self) -> int:
        if not self.routing:
            return 0
        return max(
            max(v.get("output_branches", [0]))
            for v in self.routing.values()
        )

    # ── module management ─────────────────────────────────────────────────────

    def add_module(self, module_type: str,
                   params: dict | None = None,
                   position: int | None = None,
                   input_branch: int = 0,
                   input_branches: list[int] | None = None,
                   input_branches_front: list[int] | None = None,
                   input_branches_back: list[int] | None = None,
                   output_branches: list[int] | None = None) -> str:
        """
        Add a module to the scene.

        For a *sided* module (BeamSplitter, DielectricMirror, Substrate,
        CompoundLens), pass ``input_branches_front`` / ``input_branches_back``
        directly, or - if neither is given - whatever ``input_branches`` /
        ``input_branch`` resolves to is treated as the FRONT input (the back
        face starts with no input, exactly reproducing the old single-sided
        default behaviour for newly-added modules). ``output_branches``
        becomes ``[front_out, back_out]``.

        For a plain (non-sided) module, ``input_branches`` /
        ``output_branches`` behave exactly as before.
        """
        mod = create_module(module_type, params)
        mod.id = str(uuid.uuid4())[:8]

        sided = self._is_sided(mod)

        # Resolve the branches this module will consume *before* deciding
        # where it goes in the pipeline, since auto-placement depends on it.
        if sided:
            if input_branches_front is None and input_branches_back is None:
                base = input_branches if input_branches is not None else [input_branch]
                input_branches_front = list(base)
                input_branches_back = []
            else:
                input_branches_front = list(input_branches_front or [])
                input_branches_back = list(input_branches_back or [])
                # Exclusivity: a branch can only be a front OR a back input.
                back_set = set(input_branches_back)
                input_branches_front = [b for b in input_branches_front if b not in back_set]
            consumed_branches = input_branches_front + input_branches_back
        else:
            if input_branches is None:
                input_branches = [input_branch]
            consumed_branches = list(input_branches)

        # Auto-placement: unless the caller pinned an explicit position
        # (e.g. the pipeline UI's drag-and-drop "ghost" repositioning),
        # insert the module right after the *current* last producer of any
        # branch it consumes, instead of always appending at the very end.
        # Appending unconditionally meant a module added later to branch B
        # would land *after* an existing module that already reads branch B,
        # so that existing module silently kept seeing the old (stale) last
        # output of branch B instead of picking up the new one - the module
        # order in self.modules, not just matching branch ids, is what
        # determines who feeds whom in run().
        if position is None:
            position = self._auto_position(consumed_branches)

        if position is None or position >= len(self.modules):
            self.modules.append(mod)
        else:
            self.modules.insert(position, mod)

        if sided:
            if output_branches is None:
                new_b = self._max_branch() + 1
                primary = consumed_branches or [input_branch]
                output_branches = [new_b, primary[0]]

            self.routing[mod.id] = {
                "input_branches_front": input_branches_front,
                "input_branches_back":  input_branches_back,
                "output_branches":      output_branches,
            }
        else:
            if output_branches is None:
                output_branches = [input_branches[0]]
            self.routing[mod.id] = {
                "input_branches":  input_branches,
                "output_branches": output_branches,
            }
        return mod.id

    def _auto_position(self, consumed_branches: list[int]) -> int | None:
        """Index at which to insert a newly-added module so that it lands
        immediately after the *current* last producer of any branch it
        consumes (and therefore before any already-existing module that
        also consumes that same branch). Returns None - meaning "append at
        the end", the previous unconditional behaviour - if no existing
        module produces any of the given branches (e.g. the very first
        module in the scene, or a brand-new branch id)."""
        if not consumed_branches:
            return None
        consumed = set(consumed_branches)
        last_idx = None
        for i, m in enumerate(self.modules):
            r = self.routing.get(m.id, {})
            obs = set(r.get("output_branches", []))
            if obs & consumed:
                last_idx = i
        return None if last_idx is None else last_idx + 1

    def remove_module(self, module_id: str) -> bool:
        for i, m in enumerate(self.modules):
            if m.id == module_id:
                self.modules.pop(i)
                self.routing.pop(module_id, None)
                return True
        return False

    def update_params(self, module_id: str, params: dict) -> bool:
        mod = self._get(module_id)
        if mod is None: return False
        mod.params.update(params)
        return True

    def update_routing(self, module_id: str,
                       input_branch: int | None = None,
                       input_branches: list[int] | None = None,
                       input_branches_front: list[int] | None = None,
                       input_branches_back: list[int] | None = None,
                       output_branches: list[int] | None = None) -> bool:
        """
        Update routing for a module.

        Sided modules use ``input_branches_front`` / ``input_branches_back``
        (a branch supplied to one is removed from the other, enforcing
        mutual exclusivity). Non-sided modules use ``input_branches``
        (``input_branch`` as a single-id convenience wrapper around it).
        """
        if module_id not in self.routing:
            return False
        mod = self._get(module_id)
        sided = self._is_sided(mod) if mod is not None else ("input_branches_front" in self.routing[module_id])
        r = self.routing[module_id]

        if sided:
            if input_branches_front is not None:
                r["input_branches_front"] = list(input_branches_front)
            if input_branches_back is not None:
                r["input_branches_back"] = list(input_branches_back)
            back_set = set(r.get("input_branches_back", []))
            r["input_branches_front"] = [b for b in r.get("input_branches_front", []) if b not in back_set]
        else:
            if input_branches is not None:
                r["input_branches"] = list(input_branches)
            elif input_branch is not None:
                r["input_branches"] = [input_branch]

        if output_branches is not None:
            r["output_branches"] = output_branches
        return True

    def reorder(self, ordered_ids: list[str]) -> bool:
        lookup = {m.id: m for m in self.modules}
        try:
            self.modules = [lookup[mid] for mid in ordered_ids]
            return True
        except KeyError:
            return False

    def _get(self, module_id: str) -> OpticalModule | None:
        for m in self.modules:
            if m.id == module_id: return m
        return None

    # ── execution ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        pool: list[BranchedPhoton] = []
        steps = []
        all_segments = []
        seg_end_time: dict[str, float] = {}   # seg_id -> absolute arrival time (s)

        def append_segment(*, frm, to, I, branch, module_id, photon,
                            n, parent_seg_id, intra_module=False) -> str:
            """Create one timed segment, wire it to its parent in the pulse
            DAG, and return its seg_id (to be used as the parent of whatever
            comes next for this photon)."""
            seg_id = uuid.uuid4().hex[:10]
            frm_a = np.asarray(frm, dtype=float)
            to_a  = np.asarray(to, dtype=float)
            length = float(np.linalg.norm(to_a - frm_a))
            n = max(float(n), 1.0)
            speed = C_LIGHT / n
            duration = (length / speed) if speed > 0 else 0.0
            t_start = 0.0 if parent_seg_id is None else seg_end_time.get(parent_seg_id, 0.0)
            t_end = t_start + duration
            seg = {
                "seg_id":         seg_id,
                "parent_seg_id":  parent_seg_id,
                "from":           frm_a.tolist(),
                "to":             to_a.tolist(),
                "I":              float(I),
                "branch":         branch,
                "module_id":      module_id,
                "photon_id":      photon.id,
                "n":              n,
                "wavelength_nm":  float(photon.wavelength) * 1e9,
                "E_re":           np.real(photon.E).tolist(),
                "E_im":           np.imag(photon.E).tolist(),
                "t_start":        t_start,
                "t_end":          t_end,
                "duration":       duration,
            }
            if intra_module:
                seg["intra_module"] = True
            all_segments.append(seg)
            seg_end_time[seg_id] = t_end
            return seg_id

        for mod in self.modules:
            sided = self._is_sided(mod)
            r = self.routing.get(
                mod.id,
                {"input_branches_front": [0], "input_branches_back": [], "output_branches": [0, 0]}
                if sided else
                {"input_branches": [0], "output_branches": [0]},
            )

            if sided:
                ibs_front = list(r.get("input_branches_front", []))
                ibs_back  = list(r.get("input_branches_back", []))
            else:
                ibs_front = self._input_branches(r)
                ibs_back  = []
            ibs_all = ibs_front + ibs_back
            obs = r.get("output_branches", [0])

            front_branch = obs[0] if len(obs) >= 1 else (ibs_all[0] if ibs_all else 0)
            back_branch  = obs[1] if len(obs) >= 2 else front_branch

            incoming_front = [bp for bp in pool if bp.branch in ibs_front]
            incoming_back  = [bp for bp in pool if bp.branch in ibs_back] if sided else []
            remaining      = [bp for bp in pool if bp.branch not in ibs_all]

            # Accumulators shared across this module's one or two side-calls
            out_bp:  list[BranchedPhoton] = []
            out_src: list[np.ndarray]     = []
            in_branches: list[int] = []
            waypoints_by_out: list[list | None] = []
            parent_seg_ids: list[str | None] = []
            entries_by_out: list[dict | None] = []
            in_E_by_out: list[np.ndarray | None] = []
            counts = {"n_in": 0, "n_out": 0}
            combined_info_parts: list[tuple[str, dict]] = []

            def process_group(side_incoming: list[BranchedPhoton], side: str):
                # Note: deliberately NOT skipping when side_incoming is empty -
                # source-type modules (e.g. PhotonSource) generate photons
                # from an empty input list, exactly like the pre-sided-routing
                # code always called mod.process(in_photons) unconditionally.
                side_photons  = [bp.photon for bp in side_incoming]
                side_src      = [np.real(ph.r).copy() for ph in side_photons]
                side_branch_of = {bp.photon.id: bp.branch for bp in side_incoming}
                counts["n_in"] += len(side_photons)

                if sided:
                    out_photons, info = mod.process(side_photons, side=side)
                else:
                    out_photons, info = mod.process(side_photons)
                counts["n_out"] += len(out_photons)
                combined_info_parts.append((side, info if isinstance(info, dict) else {}))

                # -- Robust input-output association (see original notes) --
                diag_log = None
                if isinstance(info, dict):
                    for key in ("reflections", "rays", "photon_status"):
                        cand = info.get(key)
                        if isinstance(cand, list) and len(cand) == len(side_photons):
                            diag_log = cand
                            break
                survivor_idx = (
                    [i for i, e in enumerate(diag_log) if not (isinstance(e, dict) and e.get("blocked"))]
                    if diag_log is not None else list(range(len(side_photons)))
                )

                same_branch = (front_branch if side == "front" else back_branch) if sided else front_branch
                opp_branch  = (back_branch if side == "front" else front_branch) if sided else front_branch

                if sided and mod.MODULE_TYPE in ("beamsplitter", "dielectric_mirror"):
                    # Interleaved pairs: out[2*i] = reflected (stays on the
                    # entry side), out[2*i+1] = transmitted (crosses to the
                    # opposite side) - see beam_splitter.py / dielectric_mirror.py.
                    for i, ph in enumerate(out_photons):
                        pair_idx = i // 2
                        is_reflected = (i % 2 == 0)
                        parent_idx = (survivor_idx[pair_idx] if pair_idx < len(survivor_idx)
                                      else (survivor_idx[-1] if survivor_idx else pair_idx))
                        parent_ph = side_photons[parent_idx] if parent_idx < len(side_photons) else None
                        branch = same_branch if is_reflected else opp_branch
                        out_bp.append(BranchedPhoton(ph, branch))
                        out_src.append(side_src[parent_idx] if parent_idx < len(side_src)
                                       else np.real(ph.r).copy())
                        parent_branch = (side_branch_of.get(parent_ph.id, side_incoming[0].branch)
                                          if parent_ph is not None else side_incoming[0].branch)
                        in_branches.append(parent_branch)
                        parent_seg_ids.append(side_incoming[parent_idx].last_seg_id
                                               if parent_idx < len(side_incoming) else None)
                        entries_by_out.append(None)
                        waypoints_by_out.append(None)
                        in_E_by_out.append(parent_ph.E.copy() if parent_ph is not None else ph.E.copy())

                elif sided and mod.MODULE_TYPE == "substrate":
                    # Substrate emits a *variable* number of beams per input
                    # photon, each already tagged "front" (exits the
                    # geometric front face) or "back" (exits the geometric
                    # back face) by Substrate._trace_photon - a label that is
                    # correct regardless of which face the photon entered
                    # through (Substrate.process() starts the trace at the
                    # correct surface for `side`), so front/back here map
                    # directly and unconditionally onto front_branch/back_branch.
                    rays_log = info.get("rays", []) if isinstance(info, dict) else []
                    out_idx = 0
                    for i, ph_in in enumerate(side_photons):
                        if i >= len(rays_log):
                            break
                        entry = rays_log[i]
                        if not isinstance(entry, dict) or entry.get("blocked"):
                            continue
                        parent_branch = side_branch_of.get(ph_in.id, side_incoming[0].branch)
                        src = side_src[i]
                        for group, branch in (
                            (entry.get("reflected", []), front_branch),
                            (entry.get("transmitted", []), back_branch),
                        ):
                            for beam_entry in group:
                                if out_idx >= len(out_photons):
                                    break
                                ph_out = out_photons[out_idx]
                                out_idx += 1
                                out_bp.append(BranchedPhoton(ph_out, branch))
                                out_src.append(src)
                                # The approach segment (source -> this hit) is
                                # still travelling on whatever branch the
                                # photon actually arrived on - NOT on
                                # front_branch/back_branch, which are this
                                # module's *output* branches. Using `branch`
                                # here made the incoming ray render in the
                                # module's back-output colour instead of the
                                # branch that produced it.
                                in_branches.append(parent_branch)
                                waypoints_by_out.append(beam_entry.get("waypoints"))
                                parent_seg_ids.append(side_incoming[i].last_seg_id
                                                       if i < len(side_incoming) else None)
                                entries_by_out.append(beam_entry)
                                in_E_by_out.append(ph_in.E.copy())

                else:
                    # Generic 1:1 (flat_mirror, parabolic_mirror, sources,
                    # gradient_medium - and SIDED CompoundLens's single
                    # pure-transmission output, which always exits the
                    # opposite face from where it entered).
                    rays_log = None
                    if isinstance(info, dict):
                        for key in ("rays", "reflections", "photon_status"):
                            cand = info.get(key)
                            if isinstance(cand, list):
                                rays_log = cand
                                break
                    survivor_entries = []
                    if rays_log is not None:
                        for entry in rays_log:
                            if isinstance(entry, dict) and entry.get("blocked"):
                                continue
                            survivor_entries.append(entry)

                    out_b = opp_branch if sided else front_branch

                    for i, ph in enumerate(out_photons):
                        parent_idx = survivor_idx[i] if i < len(survivor_idx) else i
                        out_bp.append(BranchedPhoton(ph, out_b))
                        out_src.append(side_src[parent_idx] if parent_idx < len(side_src)
                                       else np.real(ph.r).copy())
                        if parent_idx < len(side_photons):
                            input_branch = side_branch_of.get(side_photons[parent_idx].id,
                                                                side_incoming[0].branch)
                        else:
                            input_branch = side_incoming[0].branch if side_incoming else out_b
                        in_branches.append(input_branch)
                        parent_seg_ids.append(side_incoming[parent_idx].last_seg_id
                                               if parent_idx < len(side_incoming) else None)
                        entry = survivor_entries[i] if i < len(survivor_entries) else None
                        wps = entry.get("waypoints") if isinstance(entry, dict) else None
                        entries_by_out.append(entry)
                        waypoints_by_out.append(wps)

            process_group(incoming_front, "front")
            if sided:
                process_group(incoming_back, "back")

            info = _merge_side_infos(combined_info_parts)

            # Pad with None for any remaining output photons (shouldn't
            # happen, but keeps zip() safe)
            while len(waypoints_by_out) < len(out_bp): waypoints_by_out.append(None)
            while len(entries_by_out)   < len(out_bp): entries_by_out.append(None)
            while len(parent_seg_ids)   < len(out_bp): parent_seg_ids.append(None)
            while len(in_E_by_out)      < len(out_bp): in_E_by_out.append(None)

            # Now create segments with the input branch, wiring each one into
            # the pulse-travel-time DAG (seg_id / parent_seg_id / t_start / t_end)
            for bp_out, src, in_branch, wps, parent_seg_id, entry, in_E in zip(
                out_bp, out_src, in_branches, waypoints_by_out, parent_seg_ids, entries_by_out, in_E_by_out
            ):
                # The "approach" leg (previous element/source -> this
                # element's interaction point) is physically still carrying
                # the *incident* field - the module hasn't acted on the
                # photon yet at any point along that flight. Draw it with
                # in_E (falling back to the output field only if we
                # genuinely have no incident field on record) instead of
                # bp_out.photon's field, which is already post-interaction
                # (e.g. reflected/refracted).
                approach_photon = bp_out.photon
                if in_E is not None:
                    approach_photon = bp_out.photon._copy()
                    approach_photon.E = in_E

                if wps is not None and len(wps) >= 3:
                    # ── Waypoint path (lens modules: thick_lens, compound_lens) ──
                    # wps = [r_before, r_hit_0, r_hit_1, …, r_hit_N]
                    # Segment 0: pre-lens position -> first surface hit (air)
                    cur_parent = append_segment(
                        frm=wps[0], to=wps[1], I=bp_out.photon.I, branch=in_branch,
                        module_id=mod.id, photon=approach_photon,
                        n=1.0, parent_seg_id=parent_seg_id, intra_module=False,
                    )
                    # Segments 1 … N-1: surface hit -> next surface hit (inside glass)
                    n_intra = _resolve_intra_n(mod, entry, wps)
                    for k in range(1, len(wps) - 1):
                        cur_parent = append_segment(
                            frm=wps[k], to=wps[k + 1], I=bp_out.photon.I, branch=in_branch,
                            module_id=mod.id, photon=bp_out.photon,
                            n=n_intra, parent_seg_id=cur_parent, intra_module=True,
                        )
                    # The segment from the last hit to the next module is drawn
                    # by the next module's loop (src will be the photon's exit pos).
                    bp_out.last_seg_id = cur_parent
                else:
                    # ── Standard segment (no waypoints) ─────────────────────
                    to_pos = np.real(bp_out.photon.r).tolist()
                    geom_len = float(np.linalg.norm(np.asarray(to_pos) - np.asarray(src)))
                    n_std = _resolve_standard_n(mod, entry, geom_len)
                    bp_out.last_seg_id = append_segment(
                        frm=src.tolist(), to=to_pos, I=bp_out.photon.I, branch=in_branch,
                        module_id=mod.id, photon=approach_photon,
                        n=n_std, parent_seg_id=parent_seg_id, intra_module=False,
                    )

            steps.append({
                "module_id":       mod.id,
                "module_type":     mod.MODULE_TYPE,
                "module_label":    mod.LABEL,
                "sided":           sided,
                "input_branches":  ibs_all,
                "input_branches_front": ibs_front if sided else None,
                "input_branches_back":  ibs_back if sided else None,
                "output_branches": obs,
                "n_in":  counts["n_in"],
                "n_out": counts["n_out"],
                "info":  info,
                "photons_out": _serialise_photons(out_bp),
            })

            pool = remaining + out_bp

        max_t_end = max((seg["t_end"] for seg in all_segments), default=0.0)

        return {
            "steps":         steps,
            "segments":      all_segments,
            "photons_final": _serialise_photons(pool),
            "n_modules":     len(self.modules),
            "max_t_end":     max_t_end,   # seconds - total physical travel time, longest path
        }

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"modules": [m.to_dict() for m in self.modules], "routing": self.routing}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), cls=NumpyEncoder, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        s = cls()
        routing = d.get("routing", {})
        for md in d.get("modules", []):
            mtype = md.get("type")
            mcls  = MODULE_REGISTRY.get(mtype)
            if mcls is None: continue
            mod = mcls(md.get("params", {}))
            mod.id = md.get("id", str(uuid.uuid4())[:8])
            s.modules.append(mod)
            rv = routing.get(mod.id, {})
            if cls._is_sided(mod):
                s.routing[mod.id] = {
                    "input_branches_front": list(rv.get("input_branches_front", [0])),
                    "input_branches_back":  list(rv.get("input_branches_back", [])),
                    "output_branches":      rv.get("output_branches", [0, 0]),
                }
            else:
                s.routing[mod.id] = {
                    "input_branches":  list(rv.get("input_branches", [0])),
                    "output_branches": rv.get("output_branches", [0]),
                }
        return s

    @classmethod
    def from_json(cls, j: str) -> "Scene":
        return cls.from_dict(json.loads(j))

    def summary(self) -> list[dict]:
        out = []
        for m in self.modules:
            r = self.routing.get(m.id, {})
            sided = self._is_sided(m)
            if sided:
                ibf = list(r.get("input_branches_front", []))
                ibb = list(r.get("input_branches_back", []))
                union = ibf + ibb
                out.append({
                    "id": m.id, "type": m.MODULE_TYPE, "label": m.LABEL,
                    "color": m.COLOR, "icon": m.ICON, "params": m.params,
                    "sided": True,
                    "input_branches_front": ibf,
                    "input_branches_back":  ibb,
                    "input_branches": union,               # union, for pipeline-canvas grouping
                    "input_branch":   union[0] if union else 0,
                    "output_branches": r.get("output_branches", [0, 0]),
                })
            else:
                ibs = list(r.get("input_branches", [0]))
                out.append({
                    "id": m.id, "type": m.MODULE_TYPE, "label": m.LABEL,
                    "color": m.COLOR, "icon": m.ICON, "params": m.params,
                    "sided": False,
                    "input_branches": ibs,
                    "input_branch":   ibs[0],
                    "output_branches": r.get("output_branches", [0]),
                })
        return out


def _serialise_photons(photons) -> list[dict]:
    out = []
    for item in photons:
        if isinstance(item, BranchedPhoton):
            ph, branch = item.photon, item.branch
        else:
            ph, branch = item, 0
        out.append({
            "photon_id":     ph.id,
            "r":             np.real(ph.r).tolist(),
            "k":             np.real(ph.k).tolist(),
            "E_real":        np.real(ph.E).tolist(),
            "E_imag":        np.imag(ph.E).tolist(),
            "I":             float(ph.I),
            "wavelength_nm": float(ph.wavelength * 1e9),
            "branch":        branch,
        })
    return out