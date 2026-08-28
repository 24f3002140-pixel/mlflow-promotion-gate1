from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
import math
import re

app = FastAPI(title="MLflow Evidence Promotion Gate")

TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$"
)
SAFE_INT_MAX = 9007199254740991

CODES = {
    "INVALID_VERSION", "DUPLICATE_VERSION", "INVALID_POLICY",
    "MISSING_EVALUATION", "NON_FINITE", "METRIC_RANGE",
    "INVALID_TIMESTAMP", "FUTURE_EVALUATION", "STALE_EVALUATION",
    "ARTIFACT_MISMATCH", "DATASET_MISMATCH", "SCHEMA_MISMATCH",
    "ACCURACY_FLOOR", "LATENCY_LIMIT", "SIZE_LIMIT"
}

def parse_ts(value):
    if not isinstance(value, str) or not TS_RE.fullmatch(value):
        raise ValueError
    s = value[:-1] + "+00:00" if value.endswith("Z") else value
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc)

def finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def valid_safe_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= SAFE_INT_MAX

def valid_positive_version(v):
    if not isinstance(v, str) or not re.fullmatch(r"[1-9]\d*", v):
        return False
    try:
        n = int(v)
    except Exception:
        return False
    return 1 <= n <= SAFE_INT_MAX

def input_error():
    return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

@app.post("/promote")
async def promote(request: Request):
    try:
        body = await request.json()
    except Exception:
        return input_error()

    if not isinstance(body, dict):
        return input_error()

    # Top-level required fields and basic structural validation.
    if not isinstance(body.get("asOf"), str):
        return input_error()
    if not isinstance(body.get("championVersion"), str):
        return input_error()
    if not isinstance(body.get("policy"), dict):
        return input_error()
    if not isinstance(body.get("versions"), list):
        return input_error()

    try:
        as_of = parse_ts(body["asOf"])
    except Exception:
        return input_error()

    policy = body["policy"]

    # Required policy fields. Digests must be non-empty strings.
    digest_fields = ("datasetDigest", "schemaDigest")
    for k in digest_fields:
        if not isinstance(policy.get(k), str) or not policy[k]:
            return input_error()

    numeric_policy = (
        "maxAgeSeconds", "accuracyFloor", "maxLatencyMs",
        "maxSizeBytes", "minImprovement"
    )
    for k in numeric_policy:
        if k not in policy:
            return input_error()

    if not valid_safe_int(policy["maxAgeSeconds"]) or not valid_safe_int(policy["maxSizeBytes"]):
        return input_error()
    if policy["maxLatencyMs"] is None or not finite_number(policy["maxLatencyMs"]) or policy["maxLatencyMs"] < 0:
        return input_error()
    if not finite_number(policy["accuracyFloor"]) or not 0 <= policy["accuracyFloor"] <= 1:
        return input_error()
    if not finite_number(policy["minImprovement"]) or not 0 <= policy["minImprovement"] <= 1:
        return input_error()

    rs = policy.get("requiredSlices")
    if not isinstance(rs, dict):
        return input_error()
    for name, floor in rs.items():
        if not isinstance(name, str):
            return input_error()
        if not finite_number(floor) or not 0 <= floor <= 1:
            return input_error()

    champion = body["championVersion"]
    versions = body["versions"]

    # Reject every duplicate/noncanonical version before lookup maps are constructed.
    seen = set()
    structural_failed = {}
    for item in versions:
        if not isinstance(item, dict) or not valid_positive_version(item.get("version")):
            v = item.get("version") if isinstance(item, dict) and isinstance(item.get("version"), str) else "<invalid>"
            structural_failed.setdefault(v, set()).add("INVALID_VERSION")
            continue
        v = item["version"]
        if v in seen:
            structural_failed.setdefault(v, set()).add("DUPLICATE_VERSION")
        else:
            seen.add(v)

    if not valid_positive_version(champion):
        return input_error()

    # Any duplicate/invalid occurrence means the request cannot safely proceed.
    # The task specifies failedGates for versions; malformed version input is represented there.
    # Champion must be listed exactly once and valid.
    if champion not in seen or champion in structural_failed:
        # If malformed/duplicate versioning prevents identifying a unique champion, block.
        # We can still return a deterministic result for well-formed version objects.
        pass

    failed = {k: set(v) for k, v in structural_failed.items()}
    by_version = {}
    for item in versions:
        if isinstance(item, dict) and valid_positive_version(item.get("version")):
            v = item["version"]
            if v not in failed:
                by_version[v] = item

    # If champion is not uniquely represented, block.
    if champion not in by_version:
        eligible = []
        return JSONResponse(content={
            "action": "block",
            "championVersion": champion,
            "selectedVersion": None,
            "eligibleVersions": eligible,
            "failedGates": {k: sorted(v) for k, v in sorted(failed.items())},
            "aliasMutation": None,
            "evidence": None
        })

    now = as_of
    cutoff = now - timedelta(seconds=policy["maxAgeSeconds"])

    def evaluate(item):
        v = item["version"]
        gates = set()
        ev = item.get("evaluation")
        if not isinstance(ev, dict):
            gates.add("MISSING_EVALUATION")
            return gates, None

        # Timestamp validation is separate from future/stale classification.
        try:
            created = parse_ts(ev.get("createdAt"))
        except Exception:
            gates.add("INVALID_TIMESTAMP")
            created = None

        for k in ("accuracy", "latencyMs", "sizeBytes"):
            if k not in ev or not finite_number(ev[k]):
                gates.add("NON_FINITE")

        if "accuracy" in ev and finite_number(ev["accuracy"]) and not 0 <= ev["accuracy"] <= 1:
            gates.add("METRIC_RANGE")
        if "latencyMs" in ev and finite_number(ev["latencyMs"]) and ev["latencyMs"] < 0:
            gates.add("METRIC_RANGE")
        if "sizeBytes" in ev and (
            not isinstance(ev["sizeBytes"], int) or isinstance(ev["sizeBytes"], bool)
            or ev["sizeBytes"] < 0 or ev["sizeBytes"] > SAFE_INT_MAX
        ):
            gates.add("METRIC_RANGE")

        if created is not None:
            if created > now:
                gates.add("FUTURE_EVALUATION")
            if created < cutoff:
                gates.add("STALE_EVALUATION")

        # Lineage/evidence binding.
        if ev.get("artifactDigest") != item.get("artifactDigest"):
            gates.add("ARTIFACT_MISMATCH")
        if ev.get("datasetDigest") != policy["datasetDigest"]:
            gates.add("DATASET_MISMATCH")
        if ev.get("schemaDigest") != policy["schemaDigest"]:
            gates.add("SCHEMA_MISMATCH")

        if finite_number(ev.get("accuracy")) and 0 <= ev["accuracy"] <= 1:
            if ev["accuracy"] < policy["accuracyFloor"]:
                gates.add("ACCURACY_FLOOR")

        if finite_number(ev.get("latencyMs")) and ev["latencyMs"] >= 0:
            if ev["latencyMs"] > policy["maxLatencyMs"]:
                gates.add("LATENCY_LIMIT")

        if isinstance(ev.get("sizeBytes"), int) and not isinstance(ev.get("sizeBytes"), bool) and 0 <= ev["sizeBytes"] <= SAFE_INT_MAX:
            if ev["sizeBytes"] > policy["maxSizeBytes"]:
                gates.add("SIZE_LIMIT")

        slices = ev.get("slices")
        if not isinstance(slices, dict):
            slices = {}
        for name, floor in rs.items():
            if name not in slices:
                gates.add(f"MISSING_SLICE:{name}")
            else:
                val = slices[name]
                if not finite_number(val) or not 0 <= val <= 1:
                    gates.add(f"SLICE_RANGE:{name}")
                elif val < floor:
                    gates.add(f"SLICE_FLOOR:{name}")

        return gates, ev

    eligible_items = []
    for item in versions:
        if not isinstance(item, dict) or not valid_positive_version(item.get("version")):
            continue
        v = item["version"]
        if v in failed:
            continue
        gates, ev = evaluate(item)
        if gates:
            failed.setdefault(v, set()).update(gates)
        else:
            eligible_items.append(item)

    # Champion evidence must be valid. A champion with any failed gate blocks promotion.
    champ_item = by_version[champion]
    champ_gates, champ_ev = evaluate(champ_item)
    if champ_gates:
        failed.setdefault(champion, set()).update(champ_gates)
        return JSONResponse(content={
            "action": "block",
            "championVersion": champion,
            "selectedVersion": None,
            "eligibleVersions": sorted([x["version"] for x in eligible_items], key=int),
            "failedGates": {k: sorted(v) for k, v in sorted(failed.items())},
            "aliasMutation": None,
            "evidence": None
        })

    # Reconstruct eligible list after champion validation.
    eligible_versions = sorted([x["version"] for x in eligible_items], key=int)

    # Rank eligible versions: accuracy desc, latency asc, size asc, numeric version asc.
    ranked = sorted(
        eligible_items,
        key=lambda x: (
            -x["evaluation"]["accuracy"],
            x["evaluation"]["latencyMs"],
            x["evaluation"]["sizeBytes"],
            int(x["version"])
        )
    )

    # The best eligible candidate is selected. If champion is already the winner,
    # retain. Otherwise compare the winner against champion evidence.
    best = ranked[0] if ranked else None
    selected = champion
    action = "retain"
    evidence = champ_ev

    if best is not None:
        if best["version"] != champion:
            improvement = round(best["evaluation"]["accuracy"] - champ_ev["accuracy"], 12)
            if improvement >= policy["minImprovement"]:
                action = "promote"
                selected = best["version"]
                evidence = best["evaluation"]
            else:
                action = "retain"
                selected = champion
                evidence = champ_ev
        else:
            action = "retain"
            selected = champion
            evidence = champ_ev

    return JSONResponse(content={
        "action": action,
        "championVersion": champion,
        "selectedVersion": selected,
        "eligibleVersions": eligible_versions,
        "failedGates": {k: sorted(v) for k, v in sorted(failed.items())},
        "aliasMutation": (
            {"alias": "champion", "version": selected} if action == "promote" else None
        ),
        "evidence": evidence
    })

@app.get("/")
def root():
    return {"service": "MLflow Evidence Promotion Gate", "endpoint": "POST /promote"}
