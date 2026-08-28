from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
import math
import re

app = FastAPI()

SAFE_INT_MAX = 9007199254740991

TIMESTAMP_RE = re.compile(
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

    if not TIMESTAMP_RE.fullmatch(value):
        return None

    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        result = datetime.fromisoformat(value)

        if result.tzinfo is None:
            return None

        return result.astimezone(timezone.utc)

    except Exception:
        return None


def is_finite(value):
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


def is_canonical_version(value):
    if not isinstance(value, str):
        return False

    if not re.fullmatch(r"[1-9][0-9]*", value):
        return False

    try:
        number = int(value)
        return 1 <= number <= SAFE_INT_MAX
    except Exception:
        return False


def add_failure(failed, version, code):
    failed.setdefault(version, set()).add(code)


def format_failed(failed):
    result = {}

    for version, codes in failed.items():
        if not codes:
            continue

        result[version] = sorted(
            set(codes),
            key=lambda x: x.encode("utf-8")
        )

    return result


def result(
    action,
    champion,
    selected,
    eligible,
    failed,
    mutation,
    evidence
):
    return {
        "action": action,
        "championVersion": champion,
        "selectedVersion": selected,
        "eligibleVersions": eligible,
        "failedGates": format_failed(failed),
        "aliasMutation": mutation,
        "evidence": evidence
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

    # =========================================================
    # INPUT
    # =========================================================

    try:
        body = await request.json()
    except Exception:
        return invalid_input()

    if not isinstance(body, dict):
        return invalid_input()

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
    champion = body["championVersion"]

    failed = {}

    # =========================================================
    # VERSION VALIDATION
    # =========================================================

    counts = {}
    canonical_items = []

    for item in versions:

        if not isinstance(item, dict):
            add_failure(
                failed,
                "<invalid>",
                "INVALID_VERSION"
            )
            continue

        version = item.get("version")

        if not is_canonical_version(version):

            key = (
                version
                if isinstance(version, str)
                else "<invalid>"
            )

            add_failure(
                failed,
                key,
                "INVALID_VERSION"
            )

            continue

        counts[version] = counts.get(version, 0) + 1
        canonical_items.append(item)

    # Every occurrence of an exact duplicate is rejected.
    duplicates = {
        version
        for version, count in counts.items()
        if count > 1
    }

    for version in duplicates:
        add_failure(
            failed,
            version,
            "DUPLICATE_VERSION"
        )

    # =========================================================
    # POLICY
    # =========================================================

    policy_valid = isinstance(policy, dict)

    fields = (
        "datasetDigest",
        "schemaDigest",
        "maxAgeSeconds",
        "accuracyFloor",
        "requiredSlices",
        "maxLatencyMs",
        "maxSizeBytes",
        "minImprovement"
    )

    if policy_valid:

        for field in fields:
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

        if not is_safe_integer(
            policy["maxAgeSeconds"]
        ):
            policy_valid = False

        if (
            not is_finite(policy["accuracyFloor"])
            or not 0 <= policy["accuracyFloor"] <= 1
        ):
            policy_valid = False

        if not isinstance(
            policy["requiredSlices"],
            dict
        ):
            policy_valid = False

        if (
            not is_finite(policy["maxLatencyMs"])
            or policy["maxLatencyMs"] < 0
        ):
            policy_valid = False

        if not is_safe_integer(
            policy["maxSizeBytes"]
        ):
            policy_valid = False

        if (
            not is_finite(policy["minImprovement"])
            or not 0 <= policy["minImprovement"] <= 1
        ):
            policy_valid = False

    if policy_valid:

        for name, floor in policy[
            "requiredSlices"
        ].items():

            if not isinstance(name, str):
                policy_valid = False
                break

            if (
                not is_finite(floor)
                or floor < 0
                or floor > 1
            ):
                policy_valid = False
                break

    if not policy_valid:

        for version in counts:
            add_failure(
                failed,
                version,
                "INVALID_POLICY"
            )

        return result(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    # =========================================================
    # CHAMPION VALIDATION
    # =========================================================

    if not is_canonical_version(champion):

        add_failure(
            failed,
            champion,
            "INVALID_VERSION"
        )

        return result(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    if champion not in counts:
        return result(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    if champion in duplicates:
        return result(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    # =========================================================
    # LOOKUP ONLY AFTER DUPLICATE VALIDATION
    # =========================================================

    lookup = {}

    for item in canonical_items:

        version = item["version"]

        if version in duplicates:
            continue

        lookup[version] = item

    cutoff = (
        as_of
        - timedelta(
            seconds=policy["maxAgeSeconds"]
        )
    )

    # =========================================================
    # EVIDENCE VALIDATION
    # =========================================================

    def validate(item):

        gates = set()

        # -----------------------------------------------------
        # Evaluation
        # -----------------------------------------------------

        if "evaluation" not in item:
            gates.add("MISSING_EVALUATION")
            return gates

        evaluation = item["evaluation"]

        if not isinstance(evaluation, dict):
            gates.add("MISSING_EVALUATION")
            return gates

        # -----------------------------------------------------
        # Timestamp
        # -----------------------------------------------------

        created_at = evaluation.get(
            "createdAt"
        )

        created = parse_timestamp(
            created_at
        )

        if created is None:

            gates.add("INVALID_TIMESTAMP")

        else:

            if created > as_of:
                gates.add("FUTURE_EVALUATION")

            if created < cutoff:
                gates.add("STALE_EVALUATION")

        # -----------------------------------------------------
        # Numeric metrics
        # -----------------------------------------------------

        accuracy = evaluation.get(
            "accuracy"
        )

        latency = evaluation.get(
            "latencyMs"
        )

        size = evaluation.get(
            "sizeBytes"
        )

        accuracy_ok = is_finite(accuracy)
        latency_ok = is_finite(latency)
        size_ok = is_finite(size)

        if not accuracy_ok:
            gates.add("NON_FINITE")

        if not latency_ok:
            gates.add("NON_FINITE")

        if not size_ok:
            gates.add("NON_FINITE")

        # -----------------------------------------------------
        # Metric ranges
        # -----------------------------------------------------

        if accuracy_ok:
            if accuracy < 0 or accuracy > 1:
                gates.add("METRIC_RANGE")

        if latency_ok:
            if latency < 0:
                gates.add("METRIC_RANGE")

        if size_ok:
            if not is_safe_integer(size):
                gates.add("METRIC_RANGE")

        # -----------------------------------------------------
        # Immutable artifact binding
        # -----------------------------------------------------

        registered_artifact = item.get(
            "artifactDigest"
        )

        evidence_artifact = evaluation.get(
            "artifactDigest"
        )

        if evidence_artifact != registered_artifact:
            gates.add("ARTIFACT_MISMATCH")

        # -----------------------------------------------------
        # Dataset binding
        # -----------------------------------------------------

        if evaluation.get(
            "datasetDigest"
        ) != policy["datasetDigest"]:

            gates.add("DATASET_MISMATCH")

        # -----------------------------------------------------
        # Schema binding
        # -----------------------------------------------------

        if evaluation.get(
            "schemaDigest"
        ) != policy["schemaDigest"]:

            gates.add("SCHEMA_MISMATCH")

        # -----------------------------------------------------
        # Aggregate accuracy
        # -----------------------------------------------------

        if (
            accuracy_ok
            and 0 <= accuracy <= 1
            and accuracy < policy["accuracyFloor"]
        ):
            gates.add("ACCURACY_FLOOR")

        # -----------------------------------------------------
        # Aggregate latency
        # -----------------------------------------------------

        if (
            latency_ok
            and latency >= 0
            and latency > policy["maxLatencyMs"]
        ):
            gates.add("LATENCY_LIMIT")

        # -----------------------------------------------------
        # Aggregate size
        # -----------------------------------------------------

        if (
            size_ok
            and is_safe_integer(size)
            and size > policy["maxSizeBytes"]
        ):
            gates.add("SIZE_LIMIT")

        # -----------------------------------------------------
        # Slices
        # -----------------------------------------------------

        slices = evaluation.get("slices")

        if not isinstance(slices, dict):
            slices = {}

        for name, floor in policy[
            "requiredSlices"
        ].items():

            if name not in slices:

                gates.add(
                    "MISSING_SLICE:" + name
                )

                continue

            value = slices[name]

            if (
                not is_finite(value)
                or value < 0
                or value > 1
            ):

                gates.add(
                    "SLICE_RANGE:" + name
                )

                continue

            if value < floor:

                gates.add(
                    "SLICE_FLOOR:" + name
                )

        return gates

    # =========================================================
    # VALIDATE ALL UNIQUE CANONICAL VERSIONS
    # =========================================================

    eligible_items = []

    for version, item in lookup.items():

        gates = validate(item)

        if gates:

            failed.setdefault(
                version,
                set()
            ).update(gates)

        else:

            eligible_items.append(item)

    # =========================================================
    # CHAMPION EVIDENCE
    # =========================================================

    champion_item = lookup.get(
        champion
    )

    if champion_item is None:

        return result(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    champion_gates = validate(
        champion_item
    )

    if champion_gates:

        failed.setdefault(
            champion,
            set()
        ).update(
            champion_gates
        )

        # Even in a blocked response, eligible versions
        # must be ranked deterministically.
        ranked = sorted(
            eligible_items,
            key=lambda item: (
                -item["evaluation"]["accuracy"],
                item["evaluation"]["latencyMs"],
                item["evaluation"]["sizeBytes"],
                int(item["version"])
            )
        )

        return result(
            "block",
            champion,
            None,
            [
                item["version"]
                for item in ranked
            ],
            failed,
            None,
            None
        )

    champion_evidence = (
        champion_item["evaluation"]
    )

    # =========================================================
    # FINAL ELIGIBLE RANKING
    # =========================================================

    ranked = sorted(
        eligible_items,
        key=lambda item: (
            -item["evaluation"]["accuracy"],
            item["evaluation"]["latencyMs"],
            item["evaluation"]["sizeBytes"],
            int(item["version"])
        )
    )

    eligible_versions = [
        item["version"]
        for item in ranked
    ]

    # =========================================================
    # DEFENSIVE RETAIN
    # =========================================================

    if not ranked:

        return result(
            "retain",
            champion,
            champion,
            [],
            failed,
            None,
            champion_evidence
        )

    winner = ranked[0]

    # =========================================================
    # CHAMPION IS BEST
    # =========================================================

    if winner["version"] == champion:

        return result(
            "retain",
            champion,
            champion,
            eligible_versions,
            failed,
            None,
            champion_evidence
        )

    # =========================================================
    # IMPROVEMENT
    # =========================================================

    improvement = round(
        winner["evaluation"]["accuracy"]
        - champion_evidence["accuracy"],
        12
    )

    # =========================================================
    # PROMOTE
    # =========================================================

    if improvement >= policy["minImprovement"]:

        return result(
            "promote",
            champion,
            winner["version"],
            eligible_versions,
            failed,
            {
                "alias": "champion",
                "version": winner["version"]
            },
            winner["evaluation"]
        )

    # =========================================================
    # RETAIN
    # =========================================================

    return result(
        "retain",
        champion,
        champion,
        eligible_versions,
        failed,
        None,
        champion_evidence
    )
