from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
import math
import re

app = FastAPI()

SAFE_INT_MAX = 9007199254740991

TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$"
)


def bad():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


def parse_ts(v):
    if not isinstance(v, str) or not TS_RE.fullmatch(v):
        return None
    try:
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            return None
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def finite(v):
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
    )


def safe_int(v):
    return (
        isinstance(v, int)
        and not isinstance(v, bool)
        and 0 <= v <= SAFE_INT_MAX
    )


def valid_version(v):
    if not isinstance(v, str):
        return False
    if not re.fullmatch(r"[1-9][0-9]*", v):
        return False
    try:
        return 1 <= int(v) <= SAFE_INT_MAX
    except Exception:
        return False


def add(failed, version, code):
    key = str(version)
    failed.setdefault(key, set()).add(code)


def finalize_failed(failed):
    result = {}

    for version in sorted(
        failed.keys(),
        key=lambda x: x.encode("utf-8")
    ):
        result[version] = sorted(
            failed[version],
            key=lambda x: x.encode("utf-8")
        )

    return result


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/promote")
async def promote(request: Request):

    try:
        data = await request.json()
    except Exception:
        return bad()

    if not isinstance(data, dict):
        return bad()

    # Explicit INVALID_INPUT cases
    if "policy" not in data:
        return bad()

    if "versions" not in data or not isinstance(data["versions"], list):
        return bad()

    if "championVersion" not in data:
        return bad()

    if not isinstance(data["championVersion"], str):
        return bad()

    if "asOf" not in data or not isinstance(data["asOf"], str):
        return bad()

    as_of = parse_ts(data["asOf"])
    if as_of is None:
        return bad()

    policy = data["policy"]
    versions = data["versions"]
    champion = data["championVersion"]

    # ---------------------------------------------------------
    # FAILED GATES MAP
    # Every supplied version gets an entry, including [].
    # ---------------------------------------------------------

    failed = {}

    for item in versions:
        if isinstance(item, dict):
            v = item.get("version")
            if isinstance(v, str):
                failed.setdefault(v, set())
            else:
                failed.setdefault("<invalid>", set())
        else:
            failed.setdefault("<invalid>", set())

    # ---------------------------------------------------------
    # CANONICAL VERSION VALIDATION + DUPLICATES
    # ---------------------------------------------------------

    counts = {}
    canonical_items = []

    for item in versions:

        if not isinstance(item, dict):
            add(failed, "<invalid>", "INVALID_VERSION")
            continue

        v = item.get("version")

        if not valid_version(v):
            add(
                failed,
                v if isinstance(v, str) else "<invalid>",
                "INVALID_VERSION"
            )
            continue

        counts[v] = counts.get(v, 0) + 1
        canonical_items.append(item)

    duplicates = {
        v for v, count in counts.items()
        if count > 1
    }

    for v in duplicates:
        add(failed, v, "DUPLICATE_VERSION")

    # ---------------------------------------------------------
    # POLICY VALIDATION
    # ---------------------------------------------------------

    policy_ok = isinstance(policy, dict)

    required_policy = [
        "datasetDigest",
        "schemaDigest",
        "maxAgeSeconds",
        "accuracyFloor",
        "requiredSlices",
        "maxLatencyMs",
        "maxSizeBytes",
        "minImprovement"
    ]

    if policy_ok:
        for key in required_policy:
            if key not in policy:
                policy_ok = False
                break

    if policy_ok:

        if (
            not isinstance(policy["datasetDigest"], str)
            or not policy["datasetDigest"]
        ):
            policy_ok = False

        if (
            not isinstance(policy["schemaDigest"], str)
            or not policy["schemaDigest"]
        ):
            policy_ok = False

        if not safe_int(policy["maxAgeSeconds"]):
            policy_ok = False

        if (
            not finite(policy["accuracyFloor"])
            or not 0 <= policy["accuracyFloor"] <= 1
        ):
            policy_ok = False

        if not isinstance(policy["requiredSlices"], dict):
            policy_ok = False

        if (
            not finite(policy["maxLatencyMs"])
            or policy["maxLatencyMs"] < 0
        ):
            policy_ok = False

        if not safe_int(policy["maxSizeBytes"]):
            policy_ok = False

        if (
            not finite(policy["minImprovement"])
            or not 0 <= policy["minImprovement"] <= 1
        ):
            policy_ok = False

    if policy_ok:
        for name, floor in policy["requiredSlices"].items():
            if not isinstance(name, str):
                policy_ok = False
                break

            if (
                not finite(floor)
                or not 0 <= floor <= 1
            ):
                policy_ok = False
                break

    if not policy_ok:

        # INVALID_POLICY applies to every valid supplied version.
        for v in counts:
            add(failed, v, "INVALID_POLICY")

        return {
            "action": "block",
            "championVersion": champion,
            "selectedVersion": None,
            "eligibleVersions": [],
            "failedGates": finalize_failed(failed),
            "aliasMutation": None,
            "evidence": None
        }

    # ---------------------------------------------------------
    # CHAMPION MUST BE CANONICAL AND UNIQUE
    # ---------------------------------------------------------

    if not valid_version(champion):
        add(failed, champion, "INVALID_VERSION")

        return {
            "action": "block",
            "championVersion": champion,
            "selectedVersion": None,
            "eligibleVersions": [],
            "failedGates": finalize_failed(failed),
            "aliasMutation": None,
            "evidence": None
        }

    if champion not in counts:
        return {
            "action": "block",
            "championVersion": champion,
            "selectedVersion": None,
            "eligibleVersions": [],
            "failedGates": finalize_failed(failed),
            "aliasMutation": None,
            "evidence": None
        }

    if champion in duplicates:
        return {
            "action": "block",
            "championVersion": champion,
            "selectedVersion": None,
            "eligibleVersions": [],
            "failedGates": finalize_failed(failed),
            "aliasMutation": None,
            "evidence": None
        }

    # ---------------------------------------------------------
    # ONLY NOW BUILD LOOKUP MAP
    # ---------------------------------------------------------

    lookup = {}

    for item in canonical_items:
        v = item["version"]

        if v not in duplicates:
            lookup[v] = item

    cutoff = as_of - timedelta(
        seconds=policy["maxAgeSeconds"]
    )

    # ---------------------------------------------------------
    # EVIDENCE VALIDATOR
    # ---------------------------------------------------------

    def validate(item):

        gates = set()

        if "evaluation" not in item:
            return {"MISSING_EVALUATION"}

        ev = item["evaluation"]

        if not isinstance(ev, dict):
            return {"MISSING_EVALUATION"}

        # Timestamp
        created = parse_ts(ev.get("createdAt"))

        if created is None:
            gates.add("INVALID_TIMESTAMP")
        else:
            if created > as_of:
                gates.add("FUTURE_EVALUATION")
            elif created < cutoff:
                gates.add("STALE_EVALUATION")

        # Metrics
        accuracy = ev.get("accuracy")
        latency = ev.get("latencyMs")
        size = ev.get("sizeBytes")

        accuracy_finite = finite(accuracy)
        latency_finite = finite(latency)
        size_finite = finite(size)

        if not accuracy_finite:
            gates.add("NON_FINITE")

        if not latency_finite:
            gates.add("NON_FINITE")

        if not size_finite:
            gates.add("NON_FINITE")

        # Accuracy range
        if accuracy_finite:
            if accuracy < 0 or accuracy > 1:
                gates.add("METRIC_RANGE")

        # Latency range
        if latency_finite:
            if latency < 0:
                gates.add("METRIC_RANGE")

        # Size must be non-negative safe integer
        if size_finite:
            if not safe_int(size):
                gates.add("METRIC_RANGE")

        # Artifact binding
        if (
            "artifactDigest" not in item
            or not isinstance(item.get("artifactDigest"), str)
            or not item.get("artifactDigest")
            or ev.get("artifactDigest") != item.get("artifactDigest")
        ):
            gates.add("ARTIFACT_MISMATCH")

        # Dataset binding
        if ev.get("datasetDigest") != policy["datasetDigest"]:
            gates.add("DATASET_MISMATCH")

        # Schema binding
        if ev.get("schemaDigest") != policy["schemaDigest"]:
            gates.add("SCHEMA_MISMATCH")

        # Aggregate gates
        if (
            accuracy_finite
            and 0 <= accuracy <= 1
            and accuracy < policy["accuracyFloor"]
        ):
            gates.add("ACCURACY_FLOOR")

        if (
            latency_finite
            and latency >= 0
            and latency > policy["maxLatencyMs"]
        ):
            gates.add("LATENCY_LIMIT")

        if (
            safe_int(size)
            and size > policy["maxSizeBytes"]
        ):
            gates.add("SIZE_LIMIT")

        # Slices
        slices = ev.get("slices")

        if not isinstance(slices, dict):
            slices = {}

        for name, floor in policy["requiredSlices"].items():

            if name not in slices:
                gates.add(f"MISSING_SLICE:{name}")
                continue

            value = slices[name]

            if (
                not finite(value)
                or value < 0
                or value > 1
            ):
                gates.add(f"SLICE_RANGE:{name}")
                continue

            if value < floor:
                gates.add(f"SLICE_FLOOR:{name}")

        return gates

    # ---------------------------------------------------------
    # VALIDATE ALL UNIQUE CANONICAL VERSIONS
    # ---------------------------------------------------------

    eligible = []

    for v, item in lookup.items():

        gates = validate(item)

        if gates:
            failed[v].update(gates)
        else:
            eligible.append(item)

    # ---------------------------------------------------------
    # CHAMPION EVIDENCE
    # ---------------------------------------------------------

    champion_item = lookup.get(champion)

    if champion_item is None:
        return {
            "action": "block",
            "championVersion": champion,
            "selectedVersion": None,
            "eligibleVersions": [],
            "failedGates": finalize_failed(failed),
            "aliasMutation": None,
            "evidence": None
        }

    champion_gates = validate(champion_item)

    if champion_gates:

        failed[champion].update(champion_gates)

        ranked = sorted(
            eligible,
            key=lambda x: (
                -x["evaluation"]["accuracy"],
                x["evaluation"]["latencyMs"],
                x["evaluation"]["sizeBytes"],
                int(x["version"])
            )
        )

        return {
            "action": "block",
            "championVersion": champion,
            "selectedVersion": None,
            "eligibleVersions": [
                x["version"] for x in ranked
            ],
            "failedGates": finalize_failed(failed),
            "aliasMutation": None,
            "evidence": None
        }

    champion_evidence = champion_item["evaluation"]

    # ---------------------------------------------------------
    # REQUIRED RANKING
    #
    # accuracy DESC
    # latency ASC
    # size ASC
    # numeric version ASC
    # ---------------------------------------------------------

    ranked = sorted(
        eligible,
        key=lambda x: (
            -x["evaluation"]["accuracy"],
            x["evaluation"]["latencyMs"],
            x["evaluation"]["sizeBytes"],
            int(x["version"])
        )
    )

    eligible_versions = [
        x["version"] for x in ranked
    ]

    # Champion is valid, therefore it must be in eligible.
    # But keep this defensive check.
    if not ranked:
        return {
            "action": "retain",
            "championVersion": champion,
            "selectedVersion": champion,
            "eligibleVersions": [],
            "failedGates": finalize_failed(failed),
            "aliasMutation": None,
            "evidence": champion_evidence
        }

    winner = ranked[0]

    # ---------------------------------------------------------
    # CHAMPION IS ALREADY WINNER
    # ---------------------------------------------------------

    if winner["version"] == champion:

        return {
            "action": "retain",
            "championVersion": champion,
            "selectedVersion": champion,
            "eligibleVersions": eligible_versions,
            "failedGates": finalize_failed(failed),
            "aliasMutation": None,
            "evidence": champion_evidence
        }

    # ---------------------------------------------------------
    # IMPROVEMENT
    # ---------------------------------------------------------

    improvement = round(
        winner["evaluation"]["accuracy"]
        - champion_evidence["accuracy"],
        12
    )

    # ---------------------------------------------------------
    # PROMOTION
    # ---------------------------------------------------------

    if improvement >= policy["minImprovement"]:

        return {
            "action": "promote",
            "championVersion": champion,
            "selectedVersion": winner["version"],
            "eligibleVersions": eligible_versions,
            "failedGates": finalize_failed(failed),
            "aliasMutation": {
                "alias": "champion",
                "version": winner["version"]
            },
            "evidence": winner["evaluation"]
        }

    # ---------------------------------------------------------
    # RETAIN
    # ---------------------------------------------------------

    return {
        "action": "retain",
        "championVersion": champion,
        "selectedVersion": champion,
        "eligibleVersions": eligible_versions,
        "failedGates": finalize_failed(failed),
        "aliasMutation": None,
        "evidence": champion_evidence
    }
