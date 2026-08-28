from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
import math
import re

app = FastAPI(
    title="MLflow Model Promotion Gate",
    version="1.0.0"
)

SAFE_INT_MAX = 9007199254740991

TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)


def invalid_input():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


def parse_timestamp(value):
    if not isinstance(value, str):
        return None

    if not TIMESTAMP_PATTERN.fullmatch(value):
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


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


def canonical_version(value):
    if not isinstance(value, str):
        return False

    if not re.fullmatch(r"[1-9][0-9]*", value):
        return False

    try:
        n = int(value)
        return 1 <= n <= SAFE_INT_MAX
    except Exception:
        return False


def sorted_codes(codes):
    return sorted(set(codes), key=lambda x: x.encode("utf-8"))


def failed_output(failed):
    return {
        version: sorted_codes(codes)
        for version, codes in sorted(
            failed.items(),
            key=lambda x: str(x[0]).encode("utf-8")
        )
    }


@app.get("/")
async def home():
    return {
        "service": "MLflow Model Promotion Gate",
        "status": "running",
        "endpoint": "POST /promote"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/promote")
async def promote(request: Request):

    # ============================================================
    # JSON INPUT
    # ============================================================

    try:
        body = await request.json()
    except Exception:
        return invalid_input()

    if not isinstance(body, dict):
        return invalid_input()

    # Explicitly required input fields
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

    # ============================================================
    # POLICY
    # ============================================================

    required_policy_fields = [
        "datasetDigest",
        "schemaDigest",
        "maxAgeSeconds",
        "accuracyFloor",
        "requiredSlices",
        "maxLatencyMs",
        "maxSizeBytes",
        "minImprovement"
    ]

    for field in required_policy_fields:
        if field not in policy:
            return invalid_input()

    if (
        not isinstance(policy["datasetDigest"], str)
        or len(policy["datasetDigest"]) == 0
    ):
        return invalid_input()

    if (
        not isinstance(policy["schemaDigest"], str)
        or len(policy["schemaDigest"]) == 0
    ):
        return invalid_input()

    if not safe_integer(policy["maxAgeSeconds"]):
        return invalid_input()

    if not finite_number(policy["accuracyFloor"]):
        return invalid_input()

    if not 0 <= policy["accuracyFloor"] <= 1:
        return invalid_input()

    if not finite_number(policy["maxLatencyMs"]):
        return invalid_input()

    if policy["maxLatencyMs"] < 0:
        return invalid_input()

    if not safe_integer(policy["maxSizeBytes"]):
        return invalid_input()

    if not finite_number(policy["minImprovement"]):
        return invalid_input()

    if not 0 <= policy["minImprovement"] <= 1:
        return invalid_input()

    required_slices = policy["requiredSlices"]

    if not isinstance(required_slices, dict):
        return invalid_input()

    for slice_name, floor in required_slices.items():

        if not isinstance(slice_name, str):
            return invalid_input()

        if not finite_number(floor):
            return invalid_input()

        if not 0 <= floor <= 1:
            return invalid_input()

    # ============================================================
    # VERSION STRUCTURE
    #
    # IMPORTANT:
    # Check duplicates BEFORE constructing lookup maps.
    # ============================================================

    failed = {}
    occurrences = {}
    valid_items = []

    for item in versions:

        if not isinstance(item, dict):
            failed.setdefault("<invalid>", set()).add(
                "INVALID_VERSION"
            )
            continue

        version = item.get("version")

        if not canonical_version(version):
            key = version if isinstance(version, str) else "<invalid>"

            failed.setdefault(key, set()).add(
                "INVALID_VERSION"
            )

            continue

        occurrences.setdefault(version, 0)
        occurrences[version] += 1
        valid_items.append(item)

    # Every duplicate occurrence gets DUPLICATE_VERSION.
    duplicate_versions = {
        version
        for version, count in occurrences.items()
        if count > 1
    }

    for version in duplicate_versions:
        failed.setdefault(version, set()).add(
            "DUPLICATE_VERSION"
        )

    # ============================================================
    # CHAMPION
    # ============================================================

    if not canonical_version(champion_version):
        return JSONResponse(
            content={
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": [],
                "failedGates": failed_output(failed),
                "aliasMutation": None,
                "evidence": None
            }
        )

    if champion_version not in occurrences:
        return JSONResponse(
            content={
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": [],
                "failedGates": failed_output(failed),
                "aliasMutation": None,
                "evidence": None
            }
        )

    if champion_version in duplicate_versions:
        return JSONResponse(
            content={
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": [],
                "failedGates": failed_output(failed),
                "aliasMutation": None,
                "evidence": None
            }
        )

    # ============================================================
    # LOOKUP MAP
    #
    # Construct only after duplicate/noncanonical validation.
    # ============================================================

    version_map = {}

    for item in valid_items:

        version = item["version"]

        if version in duplicate_versions:
            continue

        if version in version_map:
            continue

        version_map[version] = item

    cutoff = as_of - timedelta(
        seconds=policy["maxAgeSeconds"]
    )

    # ============================================================
    # EVIDENCE EVALUATION
    # ============================================================

    def check_version(item):

        gates = set()

        if "evaluation" not in item:
            gates.add("MISSING_EVALUATION")
            return gates, None

        evaluation = item["evaluation"]

        if not isinstance(evaluation, dict):
            gates.add("MISSING_EVALUATION")
            return gates, None

        # --------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------

        created_at = evaluation.get("createdAt")
        created = parse_timestamp(created_at)

        if created is None:
            gates.add("INVALID_TIMESTAMP")
        else:

            if created > as_of:
                gates.add("FUTURE_EVALUATION")

            if created < cutoff:
                gates.add("STALE_EVALUATION")

        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

        accuracy = evaluation.get("accuracy")
        latency = evaluation.get("latencyMs")
        size = evaluation.get("sizeBytes")

        if not finite_number(accuracy):
            gates.add("NON_FINITE")

        if not finite_number(latency):
            gates.add("NON_FINITE")

        if not finite_number(size):
            gates.add("NON_FINITE")

        # --------------------------------------------------------
        # Metric ranges
        # --------------------------------------------------------

        if finite_number(accuracy):
            if accuracy < 0 or accuracy > 1:
                gates.add("METRIC_RANGE")

        if finite_number(latency):
            if latency < 0:
                gates.add("METRIC_RANGE")

        if finite_number(size):

            if (
                not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or size > SAFE_INT_MAX
            ):
                gates.add("METRIC_RANGE")

        # --------------------------------------------------------
        # Artifact lineage
        # --------------------------------------------------------

        if evaluation.get("artifactDigest") != item.get(
            "artifactDigest"
        ):
            gates.add("ARTIFACT_MISMATCH")

        if evaluation.get("datasetDigest") != policy[
            "datasetDigest"
        ]:
            gates.add("DATASET_MISMATCH")

        if evaluation.get("schemaDigest") != policy[
            "schemaDigest"
        ]:
            gates.add("SCHEMA_MISMATCH")

        # --------------------------------------------------------
        # Aggregate gates
        # --------------------------------------------------------

        if (
            finite_number(accuracy)
            and 0 <= accuracy <= 1
            and accuracy < policy["accuracyFloor"]
        ):
            gates.add("ACCURACY_FLOOR")

        if (
            finite_number(latency)
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

        for slice_name, floor in required_slices.items():

            if slice_name not in slices:
                gates.add(
                    f"MISSING_SLICE:{slice_name}"
                )
                continue

            value = slices[slice_name]

            if (
                not finite_number(value)
                or value < 0
                or value > 1
            ):
                gates.add(
                    f"SLICE_RANGE:{slice_name}"
                )
                continue

            if value < floor:
                gates.add(
                    f"SLICE_FLOOR:{slice_name}"
                )

        return gates, evaluation

    # ============================================================
    # EVALUATE VERSIONS
    # ============================================================

    eligible_items = []

    for item in valid_items:

        version = item["version"]

        if version in duplicate_versions:
            continue

        gates, evaluation = check_version(item)

        if gates:

            failed.setdefault(version, set()).update(
                gates
            )

        else:

            eligible_items.append(item)

    # ============================================================
    # CHAMPION EVIDENCE
    # ============================================================

    champion_item = version_map.get(champion_version)

    if champion_item is None:

        return JSONResponse(
            content={
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": [],
                "failedGates": failed_output(failed),
                "aliasMutation": None,
                "evidence": None
            }
        )

    champion_gates, champion_evidence = check_version(
        champion_item
    )

    if champion_gates:

        failed.setdefault(
            champion_version,
            set()
        ).update(champion_gates)

        return JSONResponse(
            content={
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": sorted(
                    [
                        item["version"]
                        for item in eligible_items
                    ],
                    key=int
                ),
                "failedGates": failed_output(failed),
                "aliasMutation": None,
                "evidence": None
            }
        )

    # ============================================================
    # RANKING
    #
    # accuracy DESC
    # latency ASC
    # size ASC
    # version numeric ASC
    # ============================================================

    ranked = sorted(
        eligible_items,
        key=lambda item: (
            -item["evaluation"]["accuracy"],
            item["evaluation"]["latencyMs"],
            item["evaluation"]["sizeBytes"],
            int(item["version"])
        )
    )

    eligible_versions = sorted(
        [
            item["version"]
            for item in eligible_items
        ],
        key=int
    )

    # ============================================================
    # NO ELIGIBLE MODELS
    # ============================================================

    if not ranked:

        return JSONResponse(
            content={
                "action": "retain",
                "championVersion": champion_version,
                "selectedVersion": champion_version,
                "eligibleVersions": [],
                "failedGates": failed_output(failed),
                "aliasMutation": None,
                "evidence": champion_evidence
            }
        )

    winner = ranked[0]

    # ============================================================
    # CHAMPION ALREADY WINS
    # ============================================================

    if winner["version"] == champion_version:

        return JSONResponse(
            content={
                "action": "retain",
                "championVersion": champion_version,
                "selectedVersion": champion_version,
                "eligibleVersions": eligible_versions,
                "failedGates": failed_output(failed),
                "aliasMutation": None,
                "evidence": champion_evidence
            }
        )

    # ============================================================
    # IMPROVEMENT
    # ============================================================

    challenger_accuracy = winner[
        "evaluation"
    ]["accuracy"]

    champion_accuracy = champion_evidence[
        "accuracy"
    ]

    improvement = round(
        challenger_accuracy - champion_accuracy,
        12
    )

    # ============================================================
    # PROMOTE
    # ============================================================

    if improvement >= policy["minImprovement"]:

        return JSONResponse(
            content={
                "action": "promote",
                "championVersion": champion_version,
                "selectedVersion": winner["version"],
                "eligibleVersions": eligible_versions,
                "failedGates": failed_output(failed),
                "aliasMutation": {
                    "alias": "champion",
                    "version": winner["version"]
                },
                "evidence": winner["evaluation"]
            }
        )

    # ============================================================
    # RETAIN
    # ============================================================

    return JSONResponse(
        content={
            "action": "retain",
            "championVersion": champion_version,
            "selectedVersion": champion_version,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_output(failed),
            "aliasMutation": None,
            "evidence": champion_evidence
        }
    )
