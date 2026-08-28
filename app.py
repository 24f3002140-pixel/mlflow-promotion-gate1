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

TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)


def bad_input():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


def parse_time(value):
    if not isinstance(value, str):
        return None

    if not TIMESTAMP_RE.fullmatch(value):
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


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def safe_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


def valid_version(value):
    if not isinstance(value, str):
        return False

    if not re.fullmatch(r"[1-9][0-9]*", value):
        return False

    try:
        number = int(value)
        return 1 <= number <= SAFE_INT_MAX
    except Exception:
        return False


def sorted_codes(codes):
    return sorted(
        set(codes),
        key=lambda x: x.encode("utf-8")
    )


def output_failed(failed):
    return {
        str(version): sorted_codes(codes)
        for version, codes in sorted(
            failed.items(),
            key=lambda pair: str(pair[0]).encode("utf-8")
        )
    }


def empty_response(champion, failed=None):
    return {
        "action": "block",
        "championVersion": champion,
        "selectedVersion": None,
        "eligibleVersions": [],
        "failedGates": output_failed(failed or {}),
        "aliasMutation": None,
        "evidence": None
    }


@app.get("/")
async def root():
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
    # PARSE JSON
    # ============================================================

    try:
        body = await request.json()
    except Exception:
        return bad_input()

    if not isinstance(body, dict):
        return bad_input()

    # These are explicitly required to return HTTP 400.
    if "policy" not in body:
        return bad_input()

    if "versions" not in body:
        return bad_input()

    if not isinstance(body["versions"], list):
        return bad_input()

    if "championVersion" not in body:
        return bad_input()

    if not isinstance(body["championVersion"], str):
        return bad_input()

    if "asOf" not in body:
        return bad_input()

    if not isinstance(body["asOf"], str):
        return bad_input()

    champion_version = body["championVersion"]
    versions = body["versions"]
    policy = body["policy"]

    as_of = parse_time(body["asOf"])

    if as_of is None:
        return bad_input()

    # ============================================================
    # FIRST PASS:
    # Validate versions and detect duplicates BEFORE maps.
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

        if not valid_version(version):
            key = version if isinstance(version, str) else "<invalid>"

            failed.setdefault(key, set()).add(
                "INVALID_VERSION"
            )

            continue

        occurrences[version] = occurrences.get(version, 0) + 1
        valid_items.append(item)

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
    # POLICY VALIDATION
    #
    # INVALID_POLICY is a gate, not a server error.
    # Only a completely missing policy is HTTP 400.
    # ============================================================

    policy_valid = True

    if not isinstance(policy, dict):
        policy_valid = False

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

    if policy_valid:
        for field in required_policy:
            if field not in policy:
                policy_valid = False
                break

    if policy_valid:

        if (
            not isinstance(policy["datasetDigest"], str)
            or policy["datasetDigest"] == ""
        ):
            policy_valid = False

        if (
            not isinstance(policy["schemaDigest"], str)
            or policy["schemaDigest"] == ""
        ):
            policy_valid = False

        if not safe_int(policy["maxAgeSeconds"]):
            policy_valid = False

        if (
            not finite(policy["accuracyFloor"])
            or not 0 <= policy["accuracyFloor"] <= 1
        ):
            policy_valid = False

        if not isinstance(policy["requiredSlices"], dict):
            policy_valid = False

        if (
            not finite(policy["maxLatencyMs"])
            or policy["maxLatencyMs"] < 0
        ):
            policy_valid = False

        if not safe_int(policy["maxSizeBytes"]):
            policy_valid = False

        if (
            not finite(policy["minImprovement"])
            or not 0 <= policy["minImprovement"] <= 1
        ):
            policy_valid = False

    if policy_valid:

        for name, floor in policy["requiredSlices"].items():

            if not isinstance(name, str):
                policy_valid = False
                break

            if (
                not finite(floor)
                or floor < 0
                or floor > 1
            ):
                policy_valid = False
                break

    if not policy_valid:

        for item in versions:

            if isinstance(item, dict) and isinstance(
                item.get("version"), str
            ):
                version = item["version"]

                if valid_version(version):
                    failed.setdefault(
                        version,
                        set()
                    ).add("INVALID_POLICY")

        return JSONResponse(
            content=empty_response(
                champion_version,
                failed
            )
        )

    required_slices = policy["requiredSlices"]

    # ============================================================
    # CHAMPION STRUCTURE
    # ============================================================

    if not valid_version(champion_version):

        failed.setdefault(
            champion_version,
            set()
        ).add("INVALID_VERSION")

        return JSONResponse(
            content=empty_response(
                champion_version,
                failed
            )
        )

    if champion_version not in occurrences:

        return JSONResponse(
            content=empty_response(
                champion_version,
                failed
            )
        )

    if champion_version in duplicate_versions:

        return JSONResponse(
            content=empty_response(
                champion_version,
                failed
            )
        )

    # ============================================================
    # LOOKUP MAP
    # Only after duplicate/noncanonical validation.
    # ============================================================

    version_map = {}

    for item in valid_items:

        version = item["version"]

        if version in duplicate_versions:
            continue

        if version not in version_map:
            version_map[version] = item

    cutoff = as_of - timedelta(
        seconds=policy["maxAgeSeconds"]
    )

    # ============================================================
    # EVALUATION GATES
    # ============================================================

    def evaluate(item):

        gates = set()

        # --------------------------------------------------------
        # Evaluation
        # --------------------------------------------------------

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
        created = parse_time(created_at)

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

        if not finite(accuracy):
            gates.add("NON_FINITE")

        if not finite(latency):
            gates.add("NON_FINITE")

        if not finite(size):
            gates.add("NON_FINITE")

        # --------------------------------------------------------
        # Metric ranges
        # --------------------------------------------------------

        if finite(accuracy):
            if accuracy < 0 or accuracy > 1:
                gates.add("METRIC_RANGE")

        if finite(latency):
            if latency < 0:
                gates.add("METRIC_RANGE")

        if finite(size):

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
            finite(accuracy)
            and 0 <= accuracy <= 1
            and accuracy < policy["accuracyFloor"]
        ):
            gates.add("ACCURACY_FLOOR")

        if (
            finite(latency)
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

                gates.add(
                    f"MISSING_SLICE:{name}"
                )

                continue

            value = slices[name]

            if (
                not finite(value)
                or value < 0
                or value > 1
            ):

                gates.add(
                    f"SLICE_RANGE:{name}"
                )

                continue

            if value < floor:

                gates.add(
                    f"SLICE_FLOOR:{name}"
                )

        return gates, evaluation

    # ============================================================
    # EVALUATE ALL NON-DUPLICATE VERSIONS
    # ============================================================

    eligible = []

    for item in valid_items:

        version = item["version"]

        if version in duplicate_versions:
            continue

        gates, evaluation = evaluate(item)

        if gates:

            failed.setdefault(
                version,
                set()
            ).update(gates)

        else:

            eligible.append(item)

    # ============================================================
    # CHAMPION EVIDENCE
    # ============================================================

    champion_item = version_map.get(
        champion_version
    )

    if champion_item is None:

        return JSONResponse(
            content=empty_response(
                champion_version,
                failed
            )
        )

    champion_gates, champion_evidence = evaluate(
        champion_item
    )

    if champion_gates:

        failed.setdefault(
            champion_version,
            set()
        ).update(champion_gates)

        eligible_versions = sorted(
            [
                item["version"]
                for item in eligible
            ],
            key=int
        )

        return JSONResponse(
            content={
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": eligible_versions,
                "failedGates": output_failed(failed),
                "aliasMutation": None,
                "evidence": None
            }
        )

    # ============================================================
    # RANK ELIGIBLE MODELS
    # ============================================================

    ranked = sorted(
        eligible,
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
            for item in eligible
        ],
        key=int
    )

    # ============================================================
    # CHAMPION IS THE ONLY ELIGIBLE MODEL
    # ============================================================

    if not ranked:

        return JSONResponse(
            content={
                "action": "retain",
                "championVersion": champion_version,
                "selectedVersion": champion_version,
                "eligibleVersions": [],
                "failedGates": output_failed(failed),
                "aliasMutation": None,
                "evidence": champion_evidence
            }
        )

    # ============================================================
    # BEST ELIGIBLE VERSION
    # ============================================================

    winner = ranked[0]

    # Champion already ranks first.
    if winner["version"] == champion_version:

        return JSONResponse(
            content={
                "action": "retain",
                "championVersion": champion_version,
                "selectedVersion": champion_version,
                "eligibleVersions": eligible_versions,
                "failedGates": output_failed(failed),
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
    # PROMOTION
    # ============================================================

    if improvement >= policy["minImprovement"]:

        return JSONResponse(
            content={
                "action": "promote",
                "championVersion": champion_version,
                "selectedVersion": winner["version"],
                "eligibleVersions": eligible_versions,
                "failedGates": output_failed(failed),
                "aliasMutation": {
                    "alias": "champion",
                    "version": winner["version"]
                },
                "evidence": winner["evaluation"]
            }
        )

    # ============================================================
    # RETAIN CHAMPION
    # ============================================================

    return JSONResponse(
        content={
            "action": "retain",
            "championVersion": champion_version,
            "selectedVersion": champion_version,
            "eligibleVersions": eligible_versions,
            "failedGates": output_failed(failed),
            "aliasMutation": None,
            "evidence": champion_evidence
        }
    )
