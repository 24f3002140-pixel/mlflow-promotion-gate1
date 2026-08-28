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


# ============================================================
# BASIC HELPERS
# ============================================================

def invalid_input():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


def parse_timestamp(value):
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


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def safe_nonnegative_integer(value):
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
        number = int(value)
        return 1 <= number <= SAFE_INT_MAX
    except Exception:
        return False


def sorted_unique_codes(codes):
    return sorted(
        set(codes),
        key=lambda code: code.encode("utf-8")
    )


def failed_gates_json(failed):
    result = {}

    for version in sorted(
        failed.keys(),
        key=lambda x: str(x).encode("utf-8")
    ):
        result[str(version)] = sorted_unique_codes(
            failed[version]
        )

    return result


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "MLflow Model Promotion Gate",
        "status": "running",
        "endpoint": "POST /promote"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok"
    }


# ============================================================
# PROMOTION ENDPOINT
# ============================================================

@app.post("/promote")
async def promote(request: Request):

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:
        body = await request.json()
    except Exception:
        return invalid_input()

    if not isinstance(body, dict):
        return invalid_input()

    # Explicit HTTP 400 requirements.
    if "policy" not in body:
        return invalid_input()

    if "versions" not in body:
        return invalid_input()

    if not isinstance(body["versions"], list):
        return invalid_input()

    if "championVersion" not in body:
        return invalid_input()

    if not isinstance(body["championVersion"], str):
        return invalid_input()

    if "asOf" not in body:
        return invalid_input()

    if not isinstance(body["asOf"], str):
        return invalid_input()

    as_of = parse_timestamp(body["asOf"])

    if as_of is None:
        return invalid_input()

    policy = body["policy"]
    versions = body["versions"]
    champion_version = body["championVersion"]

    # ========================================================
    # FIRST PASS
    #
    # Validate canonical versions and detect ALL duplicates
    # before creating the lookup map.
    # ========================================================

    failed = {}

    occurrences = {}

    valid_version_items = []

    for item in versions:

        if not isinstance(item, dict):

            failed.setdefault(
                "<invalid>",
                set()
            ).add("INVALID_VERSION")

            continue

        version = item.get("version")

        if not canonical_version(version):

            key = (
                version
                if isinstance(version, str)
                else "<invalid>"
            )

            failed.setdefault(
                key,
                set()
            ).add("INVALID_VERSION")

            continue

        occurrences[version] = (
            occurrences.get(version, 0) + 1
        )

        valid_version_items.append(item)

    # Every occurrence of a duplicate receives
    # DUPLICATE_VERSION.
    duplicate_versions = {
        version
        for version, count in occurrences.items()
        if count > 1
    }

    for version in duplicate_versions:

        failed.setdefault(
            version,
            set()
        ).add("DUPLICATE_VERSION")

    # ========================================================
    # POLICY VALIDATION
    # ========================================================

    policy_valid = True

    if not isinstance(policy, dict):
        policy_valid = False

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

    if policy_valid:

        for field in required_policy_fields:

            if field not in policy:
                policy_valid = False
                break

    if policy_valid:

        # Digests must be non-empty strings.
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

        # Age is a non-negative safe integer.
        if not safe_nonnegative_integer(
            policy["maxAgeSeconds"]
        ):
            policy_valid = False

        # Accuracy floor.
        if (
            not finite_number(policy["accuracyFloor"])
            or not 0 <= policy["accuracyFloor"] <= 1
        ):
            policy_valid = False

        # Required slices.
        if not isinstance(
            policy["requiredSlices"],
            dict
        ):
            policy_valid = False

        # Latency.
        if (
            not finite_number(policy["maxLatencyMs"])
            or policy["maxLatencyMs"] < 0
        ):
            policy_valid = False

        # Size.
        if not safe_nonnegative_integer(
            policy["maxSizeBytes"]
        ):
            policy_valid = False

        # Improvement.
        if (
            not finite_number(policy["minImprovement"])
            or not 0 <= policy["minImprovement"] <= 1
        ):
            policy_valid = False

    if policy_valid:

        for slice_name, floor in policy[
            "requiredSlices"
        ].items():

            if not isinstance(slice_name, str):
                policy_valid = False
                break

            if (
                not finite_number(floor)
                or not 0 <= floor <= 1
            ):
                policy_valid = False
                break

    # --------------------------------------------------------
    # Invalid policy is represented as a gate.
    # --------------------------------------------------------

    if not policy_valid:

        for item in versions:

            if not isinstance(item, dict):
                continue

            version = item.get("version")

            if canonical_version(version):

                failed.setdefault(
                    version,
                    set()
                ).add("INVALID_POLICY")

        return JSONResponse(
            content={
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": [],
                "failedGates": failed_gates_json(failed),
                "aliasMutation": None,
                "evidence": None
            }
        )

    required_slices = policy["requiredSlices"]

    # ========================================================
    # CHAMPION VERSION VALIDATION
    # ========================================================

    if not canonical_version(champion_version):

        failed.setdefault(
            champion_version,
            set()
        ).add("INVALID_VERSION")

        return JSONResponse(
            content={
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": [],
                "failedGates": failed_gates_json(failed),
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
                "failedGates": failed_gates_json(failed),
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
                "failedGates": failed_gates_json(failed),
                "aliasMutation": None,
                "evidence": None
            }
        )

    # ========================================================
    # LOOKUP MAP
    #
    # Created ONLY after duplicate/noncanonical validation.
    # ========================================================

    version_map = {}

    for item in valid_version_items:

        version = item["version"]

        if version in duplicate_versions:
            continue

        version_map[version] = item

    # ========================================================
    # AGE CUTOFF
    # ========================================================

    cutoff = (
        as_of
        - timedelta(
            seconds=policy["maxAgeSeconds"]
        )
    )

    # ========================================================
    # EVIDENCE VALIDATION
    # ========================================================

    def validate_evidence(item):

        gates = set()

        # ----------------------------------------------------
        # Evaluation must exist and be an object.
        # ----------------------------------------------------

        if "evaluation" not in item:

            gates.add("MISSING_EVALUATION")

            return gates, None

        evaluation = item["evaluation"]

        if not isinstance(evaluation, dict):

            gates.add("MISSING_EVALUATION")

            return gates, None

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        created_at = evaluation.get("createdAt")

        created = parse_timestamp(
            created_at
        )

        if created is None:

            gates.add("INVALID_TIMESTAMP")

        else:

            # Future check is independent of accuracy/tags/etc.
            if created > as_of:
                gates.add("FUTURE_EVALUATION")

            if created < cutoff:
                gates.add("STALE_EVALUATION")

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = evaluation.get(
            "accuracy"
        )

        latency = evaluation.get(
            "latencyMs"
        )

        size = evaluation.get(
            "sizeBytes"
        )

        # Accuracy and latency must be finite.
        if not finite_number(accuracy):
            gates.add("NON_FINITE")

        if not finite_number(latency):
            gates.add("NON_FINITE")

        # Size must be finite too.
        if not finite_number(size):
            gates.add("NON_FINITE")

        # ----------------------------------------------------
        # Accuracy range
        # ----------------------------------------------------

        if finite_number(accuracy):

            if accuracy < 0 or accuracy > 1:
                gates.add("METRIC_RANGE")

        # ----------------------------------------------------
        # Latency range
        # ----------------------------------------------------

        if finite_number(latency):

            if latency < 0:
                gates.add("METRIC_RANGE")

        # ----------------------------------------------------
        # Size range/type
        #
        # Size is required to be a non-negative safe integer.
        # ----------------------------------------------------

        if finite_number(size):

            if not safe_nonnegative_integer(size):
                gates.add("METRIC_RANGE")

        # ----------------------------------------------------
        # Immutable artifact lineage
        #
        # Registered artifactDigest must exactly match evidence.
        # ----------------------------------------------------

        if (
            "artifactDigest" not in item
            or evaluation.get("artifactDigest")
            != item.get("artifactDigest")
        ):

            gates.add("ARTIFACT_MISMATCH")

        # ----------------------------------------------------
        # Dataset lineage
        # ----------------------------------------------------

        if (
            evaluation.get("datasetDigest")
            != policy["datasetDigest"]
        ):

            gates.add("DATASET_MISMATCH")

        # ----------------------------------------------------
        # Schema lineage
        # ----------------------------------------------------

        if (
            evaluation.get("schemaDigest")
            != policy["schemaDigest"]
        ):

            gates.add("SCHEMA_MISMATCH")

        # ----------------------------------------------------
        # Accuracy floor
        # ----------------------------------------------------

        if (
            finite_number(accuracy)
            and 0 <= accuracy <= 1
            and accuracy < policy["accuracyFloor"]
        ):

            gates.add("ACCURACY_FLOOR")

        # ----------------------------------------------------
        # Latency limit
        # ----------------------------------------------------

        if (
            finite_number(latency)
            and latency >= 0
            and latency > policy["maxLatencyMs"]
        ):

            gates.add("LATENCY_LIMIT")

        # ----------------------------------------------------
        # Size limit
        # ----------------------------------------------------

        if (
            safe_nonnegative_integer(size)
            and size > policy["maxSizeBytes"]
        ):

            gates.add("SIZE_LIMIT")

        # ----------------------------------------------------
        # Required slices
        # ----------------------------------------------------

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
                not finite_number(value)
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

    # ========================================================
    # EVALUATE ALL UNIQUE CANONICAL VERSIONS
    # ========================================================

    eligible_items = []

    for item in valid_version_items:

        version = item["version"]

        # Duplicate versions are never eligible.
        if version in duplicate_versions:
            continue

        gates, evidence = validate_evidence(
            item
        )

        if gates:

            failed.setdefault(
                version,
                set()
            ).update(gates)

        else:

            eligible_items.append(item)

    # ========================================================
    # CHAMPION EVIDENCE
    # ========================================================

    champion_item = version_map.get(
        champion_version
    )

    if champion_item is None:

        return JSONResponse(
            content={
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": [],
                "failedGates": failed_gates_json(failed),
                "aliasMutation": None,
                "evidence": None
            }
        )

    champion_gates, champion_evidence = (
        validate_evidence(champion_item)
    )

    # --------------------------------------------------------
    # Invalid champion evidence ALWAYS blocks.
    # --------------------------------------------------------

    if champion_gates:

        failed.setdefault(
            champion_version,
            set()
        ).update(champion_gates)

        # IMPORTANT:
        # eligibleVersions remains in RANKED order.
        ranked_eligible = sorted(
            eligible_items,
            key=lambda item: (
                -item["evaluation"]["accuracy"],
                item["evaluation"]["latencyMs"],
                item["evaluation"]["sizeBytes"],
                int(item["version"])
            )
        )

        return JSONResponse(
            content={
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": [
                    item["version"]
                    for item in ranked_eligible
                ],
                "failedGates": failed_gates_json(failed),
                "aliasMutation": None,
                "evidence": None
            }
        )

    # ========================================================
    # RANK ELIGIBLE VERSIONS
    #
    # 1. accuracy DESC
    # 2. latency ASC
    # 3. size ASC
    # 4. numeric version ASC
    # ========================================================

    ranked = sorted(
        eligible_items,
        key=lambda item: (
            -item["evaluation"]["accuracy"],
            item["evaluation"]["latencyMs"],
            item["evaluation"]["sizeBytes"],
            int(item["version"])
        )
    )

    # IMPORTANT:
    # Response must use ranking order.
    eligible_versions = [
        item["version"]
        for item in ranked
    ]

    # ========================================================
    # NO ELIGIBLE VERSIONS
    # ========================================================

    if not ranked:

        return JSONResponse(
            content={
                "action": "retain",
                "championVersion": champion_version,
                "selectedVersion": champion_version,
                "eligibleVersions": [],
                "failedGates": failed_gates_json(failed),
                "aliasMutation": None,
                "evidence": champion_evidence
            }
        )

    # ========================================================
    # BEST VERSION
    # ========================================================

    winner = ranked[0]

    # ========================================================
    # CHAMPION ALREADY BEST
    # ========================================================

    if winner["version"] == champion_version:

        return JSONResponse(
            content={
                "action": "retain",
                "championVersion": champion_version,
                "selectedVersion": champion_version,
                "eligibleVersions": eligible_versions,
                "failedGates": failed_gates_json(failed),
                "aliasMutation": None,
                "evidence": champion_evidence
            }
        )

    # ========================================================
    # IMPROVEMENT
    #
    # Round challenger accuracy - champion accuracy
    # to exactly 12 decimal places.
    # ========================================================

    challenger_accuracy = winner[
        "evaluation"
    ]["accuracy"]

    champion_accuracy = champion_evidence[
        "accuracy"
    ]

    improvement = round(
        challenger_accuracy
        - champion_accuracy,
        12
    )

    # ========================================================
    # PROMOTE
    # ========================================================

    if improvement >= policy["minImprovement"]:

        return JSONResponse(
            content={
                "action": "promote",
                "championVersion": champion_version,
                "selectedVersion": winner["version"],
                "eligibleVersions": eligible_versions,
                "failedGates": failed_gates_json(failed),
                "aliasMutation": {
                    "alias": "champion",
                    "version": winner["version"]
                },
                "evidence": winner["evaluation"]
            }
        )

    # ========================================================
    # RETAIN
    # ========================================================

    return JSONResponse(
        content={
            "action": "retain",
            "championVersion": champion_version,
            "selectedVersion": champion_version,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates_json(failed),
            "aliasMutation": None,
            "evidence": champion_evidence
        }
    )
