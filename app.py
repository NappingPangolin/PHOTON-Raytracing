"""
app.py - Flask backend (branched pipeline edition)
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from scene import Scene, NumpyEncoder
from modules import module_catalog

app = Flask(__name__, static_folder="static")
CORS(app)
app.json_encoder = NumpyEncoder

_scene = Scene()


def _json(obj):
    return app.response_class(json.dumps(obj, cls=NumpyEncoder), mimetype="application/json")


@app.route("/")
def raytracingindex():
    return send_from_directory("static", "raytracing.html")


@app.route("/api/catalog")
def catalog():
    return _json(module_catalog())


@app.route("/api/scene")
def get_scene():
    return _json(_scene.summary())


@app.route("/api/scene/add", methods=["POST"])
def add_module():
    data   = request.json or {}
    mtype  = data.get("type", "")
    params = data.get("params", None)
    pos    = data.get("position", None)
    # Accept both input_branches (list, new) and input_branch (scalar, legacy)
    ibs    = data.get("input_branches", None)
    ib     = int(data.get("input_branch", 0))
    if ibs is not None:
        ibs = [int(b) for b in ibs]
    # Sided modules (BeamSplitter/DielectricMirror/Substrate/CompoundLens)
    ibs_front = data.get("input_branches_front", None)
    ibs_back  = data.get("input_branches_back", None)
    if ibs_front is not None:
        ibs_front = [int(b) for b in ibs_front]
    if ibs_back is not None:
        ibs_back = [int(b) for b in ibs_back]
    obs    = data.get("output_branches", None)
    if obs is not None:
        obs = [int(b) for b in obs]
    try:
        mid = _scene.add_module(mtype, params, pos,
                                input_branch=ib,
                                input_branches=ibs,
                                input_branches_front=ibs_front,
                                input_branches_back=ibs_back,
                                output_branches=obs)
        return _json({"ok": True, "id": mid, "scene": _scene.summary()})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}), 400


@app.route("/api/scene/remove", methods=["POST"])
def remove_module():
    data = request.json or {}
    mid  = data.get("id", "")
    ok   = _scene.remove_module(mid)
    return _json({"ok": ok, "scene": _scene.summary()})


@app.route("/api/scene/update", methods=["POST"])
def update_module():
    data   = request.json or {}
    mid    = data.get("id", "")
    params = data.get("params", {})
    ok     = _scene.update_params(mid, params)
    return _json({"ok": ok, "scene": _scene.summary()})


@app.route("/api/scene/routing", methods=["POST"])
def update_routing():
    data = request.json or {}
    mid  = data.get("id", "")
    ib   = data.get("input_branch", None)
    ibs  = data.get("input_branches", None)
    ibf  = data.get("input_branches_front", None)
    ibb  = data.get("input_branches_back", None)
    obs  = data.get("output_branches", None)
    if ib  is not None: ib  = int(ib)
    if ibs is not None: ibs = [int(b) for b in ibs]
    if ibf is not None: ibf = [int(b) for b in ibf]
    if ibb is not None: ibb = [int(b) for b in ibb]
    if obs is not None: obs = [int(b) for b in obs]
    ok = _scene.update_routing(mid, ib, input_branches=ibs,
                               input_branches_front=ibf,
                               input_branches_back=ibb,
                               output_branches=obs)
    return _json({"ok": ok, "scene": _scene.summary()})


@app.route("/api/scene/branches")
def get_branches():
    """Return all branch ids currently produced in the scene."""
    branches = set()
    for v in _scene.routing.values():
        for b in v.get("output_branches", []):
            branches.add(b)
    # Also include all input branches (front+back for sided modules) so the
    # frontend knows what's available
    for v in _scene.routing.values():
        for b in _scene._all_input_branches(v):
            branches.add(b)
    return _json({"branches": sorted(branches)})


@app.route("/api/scene/reorder", methods=["POST"])
def reorder_modules():
    data = request.json or {}
    ids  = data.get("ids", [])
    ok   = _scene.reorder(ids)
    return _json({"ok": ok, "scene": _scene.summary()})


@app.route("/api/scene/reset", methods=["POST"])
def reset_scene():
    global _scene
    _scene = Scene()
    return _json({"ok": True})


@app.route("/api/scene/run", methods=["POST"])
def run_scene():
    try:
        result = _scene.run()
        return _json({"ok": True, "result": result})
    except Exception as e:
        import traceback
        return _json({"ok": False, "error": str(e),
                      "traceback": traceback.format_exc()}), 500


@app.route("/api/scene/save")
def save_scene():
    return app.response_class(
        _scene.to_json(), mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=scene.json"},
    )


@app.route("/api/scene/load", methods=["POST"])
def load_scene():
    global _scene
    data = request.json or {}
    try:
        _scene = Scene.from_dict(data)
        return _json({"ok": True, "scene": _scene.summary()})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    print("\n  ╔══════════════════════════════════════╗")
    print("  ║   Optical Ray Tracer  -  v3.3        ║")
    print("  ║   http://localhost:5050               ║")
    print("  ╚══════════════════════════════════════╝\n")
    app.run(host="0.0.0.0", port=5050, debug=False)