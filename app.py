from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
import math
import re

app = FastAPI()

SAFE_INT_MAX = 9007199254740991

TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$"
)


def invalid_input():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


def parse_timestamp(value):
    if not isinstance(value, str) or not TS_RE.fullmatch(value):
        return None

    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        dt = datetime.fromisoformat(value)

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def is_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


def is_positive_version(value):
    if not isinstance(value, str):
        return False

    if not re.fullmatch(r"[1-9][0-9]*", value):
        return False

    try:
        number = int(value)
        return 1 <= number <= SAFE_INT_MAX
    except Exception:
        return False


def gate_sort(codes):
    return sorted(set(codes), key=lambda x: x.encode("utf-8"))


@app.post("/promote")
async def promote(request: Request):

    # ------------------------------------------------------------
    # Parse JSON
    # ------------------------------------------------------------
    try:
        body = await request.json()
    except Exception:
        return invalid_input()

    if not isinstance(body, dict):
        return invalid_input()

    # The explicitly-invalid top-level inputs
    if "asOf" not in body:
        return invalid_input()

    if "championVersion" not in body:
        return invalid_input()

    if "policy" not in body:
        return invalid_input()

    if "versions" not in body:
        return invalid_input()

    if not isinstance(body["asOf"], str):
        return invalid_input()

    if not isinstance(body["championVersion"], str):
        return invalid_input()

    if not isinstance(body["policy"], dict):
        return invalid_input()

    if not isinstance(body["versions"], list):
        return invalid_input()

    as_of = parse_timestamp(body["asOf"])

    if as_of is None:
        return invalid_input()

    champion_version = body["championVersion"]
    policy = body["policy"]
    versions = body["versions"]

    # ------------------------------------------------------------
    # Policy validation
    # ------------------------------------------------------------
    required_policy = [
        "datasetDigest",
        "schemaDigest",
        "maxAgeSeconds",
        "accuracyFloor",
        "requiredSlices",
        "maxLatencyMs",
        "maxSizeBytes",
        "minImprovement",
    ]

    for key in required_policy:
        if key not in policy:
            return invalid_input()

    if (
        not isinstance(policy["datasetDigest"], str)
        or policy["datasetDigest"] == ""
    ):
        return invalid_input()

    if (
        not isinstance(policy["schemaDigest"], str)
        or policy["schemaDigest"] == ""
    ):
        return invalid_input()

    if not is_safe_integer(policy["maxAgeSeconds"]):
        return invalid_input()

    if not is_finite_number(policy["accuracyFloor"]):
        return invalid_input()

    if not 0 <= policy["accuracyFloor"] <= 1:
        return invalid_input()

    if not is_finite_number(policy["maxLatencyMs"]):
        return invalid_input()

    if policy["maxLatencyMs"] < 0:
        return invalid_input()

    if not is_safe_integer(policy["maxSizeBytes"]):
        return invalid_input()

    if not is_finite_number(policy["minImprovement"]):
        return invalid_input()

    if not 0 <= policy["minImprovement"] <= 1:
        return invalid_input()

    required_slices = policy["requiredSlices"]

    if not isinstance(required_slices, dict):
        return invalid_input()

    for name, floor in required_slices.items():

        if not isinstance(name, str):
            return invalid_input()

        if not is_finite_number(floor):
            return invalid_input()

        if not 0 <= floor <= 1:
            return invalid_input()

    # ------------------------------------------------------------
    # Version validation
    #
    # IMPORTANT:
    # We scan ALL occurrences before constructing the lookup map.
    # ------------------------------------------------------------
    failed_gates = {}
    seen_versions = set()

    valid_version_items = []

    for item in versions:

        if not isinstance(item, dict):
            failed_gates.setdefault("<invalid>", set()).add(
                "INVALID_VERSION"
            )
            continue

        version = item.get("version")

        if not is_positive_version(version):
            failed_gates.setdefault(
                version if isinstance(version, str) else "<invalid>",
                set()
            ).add("INVALID_VERSION")
            continue

        if version in seen_versions:
            failed_gates.setdefault(version, set()).add(
                "DUPLICATE_VERSION"
            )
            continue

        seen_versions.add(version)
        valid_version_items.append(item)

    # Champion must identify a listed canonical version.
    if (
        not is_positive_version(champion_version)
        or champion_version not in seen_versions
        or champion_version in failed_gates
    ):
        return JSONResponse(
            content={
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": [],
                "failedGates": {
                    k: gate_sort(v)
                    for k, v in sorted(failed_gates.items())
                },
                "aliasMutation": None,
                "evidence": None,
            }
        )

    # ------------------------------------------------------------
    # Build lookup map ONLY after duplicate/noncanonical scan.
    # ------------------------------------------------------------
    version_map = {
        item["version"]: item
        for item in valid_version_items
        if item["version"] not in failed_gates
    }

    cutoff = as_of - timedelta(
        seconds=policy["maxAgeSeconds"]
    )

    # ------------------------------------------------------------
    # Evaluate evidence
    # ------------------------------------------------------------
    def evaluate(item):

        gates = set()

        if "evaluation" not in item:
            gates.add("MISSING_EVALUATION")
            return gates, None

        evaluation = item["evaluation"]

        if not isinstance(evaluation, dict):
            gates.add("MISSING_EVALUATION")
            return gates, None

        # Timestamp
        created_at = evaluation.get("createdAt")
        created = parse_timestamp(created_at)

        if created is None:
            gates.add("INVALID_TIMESTAMP")
        else:
            if created > as_of:
                gates.add("FUTURE_EVALUATION")

            if created < cutoff:
                gates.add("STALE_EVALUATION")

        # Required numeric metrics
        for key in ["accuracy", "latencyMs", "sizeBytes"]:

            if key not in evaluation:
                gates.add("NON_FINITE")
                continue

            value = evaluation[key]

            if not is_finite_number(value):
                gates.add("NON_FINITE")

        accuracy = evaluation.get("accuracy")
        latency = evaluation.get("latencyMs")
        size = evaluation.get("sizeBytes")

        # Metric ranges
        if is_finite_number(accuracy):
            if not 0 <= accuracy <= 1:
                gates.add("METRIC_RANGE")

        if is_finite_number(latency):
            if latency < 0:
                gates.add("METRIC_RANGE")

        if is_finite_number(size):
            if (
                not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or size > SAFE_INT_MAX
            ):
                gates.add("METRIC_RANGE")

        # --------------------------------------------------------
        # Evidence / lineage
        # --------------------------------------------------------
        if evaluation.get("artifactDigest") != item.get("artifactDigest"):
            gates.add("ARTIFACT_MISMATCH")

        if evaluation.get("datasetDigest") != policy["datasetDigest"]:
            gates.add("DATASET_MISMATCH")

        if evaluation.get("schemaDigest") != policy["schemaDigest"]:
            gates.add("SCHEMA_MISMATCH")

        # --------------------------------------------------------
        # Aggregate gates
        # --------------------------------------------------------
        if (
            is_finite_number(accuracy)
            and 0 <= accuracy <= 1
            and accuracy < policy["accuracyFloor"]
        ):
            gates.add("ACCURACY_FLOOR")

        if (
            is_finite_number(latency)
            and latency >= 0
            and latency > policy["maxLatencyMs"]
        ):
            gates.add("LATENCY_LIMIT")

        if (
            isinstance(size, int)
            and not isinstance(size, bool)
            and 0 <= size <= SAFE_INT_MAX
            and size > policy["maxSizeBytes"]
        ):
            gates.add("SIZE_LIMIT")

        # --------------------------------------------------------
        # Required slices
        # --------------------------------------------------------
        slices = evaluation.get("slices")

        if not isinstance(slices, dict):
            slices = {}

        for name, floor in required_slices.items():

            if name not in slices:
                gates.add(f"MISSING_SLICE:{name}")
                continue

            value = slices[name]

            if not is_finite_number(value) or not 0 <= value <= 1:
                gates.add(f"SLICE_RANGE:{name}")
                continue

            if value < floor:
                gates.add(f"SLICE_FLOOR:{name}")

        return gates, evaluation

    # ------------------------------------------------------------
    # Evaluate all valid versions
    # ------------------------------------------------------------
    eligible_items = []

    for item in valid_version_items:

        version = item["version"]

        if version in failed_gates:
            continue

        gates, evaluation = evaluate(item)

        if gates:
            failed_gates.setdefault(version, set()).update(gates)
        else:
            eligible_items.append(item)

    # ------------------------------------------------------------
    # Champion evidence must be valid
    # ------------------------------------------------------------
    champion_item = version_map.get(champion_version)

    if champion_item is None:
        return JSONResponse(
            content={
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": sorted(
                    [x["version"] for x in eligible_items],
                    key=int
                ),
                "failedGates": {
                    k: gate_sort(v)
                    for k, v in sorted(failed_gates.items())
                },
                "aliasMutation": None,
                "evidence": None,
            }
        )

    champion_gates, champion_evaluation = evaluate(
        champion_item
    )

    if champion_gates:
        failed_gates.setdefault(
            champion_version,
            set()
        ).update(champion_gates)

        return JSONResponse(
            content={
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": sorted(
                    [x["version"] for x in eligible_items],
                    key=int
                ),
                "failedGates": {
                    k: gate_sort(v)
                    for k, v in sorted(failed_gates.items())
                },
                "aliasMutation": None,
                "evidence": None,
            }
        )

    # ------------------------------------------------------------
    # Rank eligible versions
    # ------------------------------------------------------------
    ranked = sorted(
        eligible_items,
        key=lambda item: (
            -item["evaluation"]["accuracy"],
            item["evaluation"]["latencyMs"],
            item["evaluation"]["sizeBytes"],
            int(item["version"]),
        ),
    )

    # No eligible challenger.
    if not ranked:
        return JSONResponse(
            content={
                "action": "retain",
                "championVersion": champion_version,
                "selectedVersion": champion_version,
                "eligibleVersions": [],
                "failedGates": {
                    k: gate_sort(v)
                    for k, v in sorted(failed_gates.items())
                },
                "aliasMutation": None,
                "evidence": champion_evaluation,
            }
        )

    winner = ranked[0]

    # ------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------
    if winner["version"] == champion_version:

        return JSONResponse(
            content={
                "action": "retain",
                "championVersion": champion_version,
                "selectedVersion": champion_version,
                "eligibleVersions": sorted(
                    [x["version"] for x in eligible_items],
                    key=int
                ),
                "failedGates": {
                    k: gate_sort(v)
                    for k, v in sorted(failed_gates.items())
                },
                "aliasMutation": None,
                "evidence": champion_evaluation,
            }
        )

    challenger_accuracy = winner["evaluation"]["accuracy"]
    champion_accuracy = champion_evaluation["accuracy"]

    improvement = round(
        challenger_accuracy - champion_accuracy,
        12
    )

    # ------------------------------------------------------------
    # Promote
    # ------------------------------------------------------------
    if improvement >= policy["minImprovement"]:

        return JSONResponse(
            content={
                "action": "promote",
                "championVersion": champion_version,
                "selectedVersion": winner["version"],
                "eligibleVersions": sorted(
                    [x["version"] for x in eligible_items],
                    key=int
                ),
                "failedGates": {
                    k: gate_sort(v)
                    for k, v in sorted(failed_gates.items())
                },
                "aliasMutation": {
                    "alias": "champion",
                    "version": winner["version"],
                },
                "evidence": winner["evaluation"],
            }
        )

    # ------------------------------------------------------------
    # Retain champion
    # ------------------------------------------------------------
    return JSONResponse(
        content={
            "action": "retain",
            "championVersion": champion_version,
            "selectedVersion": champion_version,
            "eligibleVersions": sorted(
                [x["version"] for x in eligible_items],
                key=int
            ),
            "failedGates": {
                k: gate_sort(v)
                for k, v in sorted(failed_gates.items())
            },
            "aliasMutation": None,
            "evidence": champion_evaluation,
        }
    )


@app.get("/")
def root():
    return {
        "service": "MLflow Evidence Promotion Gate",
        "endpoint": "POST /promote",
    }
