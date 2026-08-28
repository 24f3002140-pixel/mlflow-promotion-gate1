from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
import math
import re

app = FastAPI()

MAX_SAFE_INTEGER = 9007199254740991

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
        text = value
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        dt = datetime.fromisoformat(text)

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
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def is_canonical_version(value):
    if not isinstance(value, str):
        return False

    if not re.fullmatch(r"[1-9][0-9]*", value):
        return False

    try:
        number = int(value)
    except Exception:
        return False

    return 1 <= number <= MAX_SAFE_INTEGER


def add_failure(failed, version, code):
    failed.setdefault(version, set()).add(code)


def sort_codes(codes):
    return sorted(
        set(codes),
        key=lambda value: value.encode("utf-8")
    )


def build_failed_gates(failed):
    return {
        version: sort_codes(codes)
        for version, codes in failed.items()
        if codes
    }


def response(
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
        "failedGates": build_failed_gates(failed),
        "aliasMutation": mutation,
        "evidence": evidence
    }


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/promote")
async def promote(request: Request):

    # =========================================================
    # TOP LEVEL INPUT
    # =========================================================

    try:
        payload = await request.json()
    except Exception:
        return invalid_input()

    if not isinstance(payload, dict):
        return invalid_input()

    # These are explicitly HTTP 400 cases.
    if "policy" not in payload:
        return invalid_input()

    if "versions" not in payload:
        return invalid_input()

    if not isinstance(payload["versions"], list):
        return invalid_input()

    if "championVersion" not in payload:
        return invalid_input()

    if not isinstance(payload["championVersion"], str):
        return invalid_input()

    if "asOf" not in payload:
        return invalid_input()

    if not isinstance(payload["asOf"], str):
        return invalid_input()

    as_of = parse_timestamp(payload["asOf"])

    if as_of is None:
        return invalid_input()

    policy = payload["policy"]
    versions = payload["versions"]
    champion = payload["championVersion"]

    failed = {}

    # =========================================================
    # CANONICAL / UNIQUE VERSION VALIDATION
    # =========================================================

    occurrences = {}
    valid_entries = []

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

        occurrences[version] = (
            occurrences.get(version, 0) + 1
        )

        valid_entries.append(item)

    duplicate_versions = {
        version
        for version, count in occurrences.items()
        if count > 1
    }

    for version in duplicate_versions:
        add_failure(
            failed,
            version,
            "DUPLICATE_VERSION"
        )

    # =========================================================
    # POLICY VALIDATION
    # =========================================================

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
            not is_finite_number(
                policy["accuracyFloor"]
            )
            or not 0 <= policy["accuracyFloor"] <= 1
        ):
            policy_valid = False

        if not isinstance(
            policy["requiredSlices"],
            dict
        ):
            policy_valid = False

        if (
            not is_finite_number(
                policy["maxLatencyMs"]
            )
            or policy["maxLatencyMs"] < 0
        ):
            policy_valid = False

        if not is_safe_integer(
            policy["maxSizeBytes"]
        ):
            policy_valid = False

        if (
            not is_finite_number(
                policy["minImprovement"]
            )
            or not 0 <= policy["minImprovement"] <= 1
        ):
            policy_valid = False

    if policy_valid:

        for name, floor in policy[
            "requiredSlices"
        ].items():

            if (
                not isinstance(name, str)
                or not is_finite_number(floor)
                or floor < 0
                or floor > 1
            ):
                policy_valid = False
                break

    # Invalid policy is a gate, not HTTP 400.
    if not policy_valid:

        for version in occurrences:
            add_failure(
                failed,
                version,
                "INVALID_POLICY"
            )

        return response(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    # =========================================================
    # CHAMPION VERSION VALIDATION
    # =========================================================

    if not is_canonical_version(champion):

        add_failure(
            failed,
            champion,
            "INVALID_VERSION"
        )

        return response(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    if champion not in occurrences:

        return response(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    if champion in duplicate_versions:

        return response(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    # =========================================================
    # BUILD LOOKUP ONLY AFTER INVALID/DUPLICATE REJECTION
    # =========================================================

    lookup = {}

    for item in valid_entries:

        version = item["version"]

        if version in duplicate_versions:
            continue

        lookup[version] = item

    # =========================================================
    # AGE WINDOW
    # =========================================================

    minimum_created_at = (
        as_of
        - timedelta(
            seconds=policy["maxAgeSeconds"]
        )
    )

    # =========================================================
    # EVALUATE EVIDENCE
    # =========================================================

    def evaluate(item):

        gates = set()

        # -----------------------------------------------------
        # EVALUATION OBJECT
        # -----------------------------------------------------

        if "evaluation" not in item:
            gates.add("MISSING_EVALUATION")
            return gates

        evaluation = item["evaluation"]

        if not isinstance(evaluation, dict):
            gates.add("MISSING_EVALUATION")
            return gates

        # -----------------------------------------------------
        # CREATED AT
        # -----------------------------------------------------

        created_at = parse_timestamp(
            evaluation.get("createdAt")
        )

        if created_at is None:

            gates.add("INVALID_TIMESTAMP")

        else:

            if created_at > as_of:
                gates.add("FUTURE_EVALUATION")

            if created_at < minimum_created_at:
                gates.add("STALE_EVALUATION")

        # -----------------------------------------------------
        # ACCURACY
        # -----------------------------------------------------

        accuracy = evaluation.get("accuracy")

        if not is_finite_number(accuracy):

            gates.add("NON_FINITE")

        elif accuracy < 0 or accuracy > 1:

            gates.add("METRIC_RANGE")

        # -----------------------------------------------------
        # LATENCY
        # -----------------------------------------------------

        latency = evaluation.get("latencyMs")

        if not is_finite_number(latency):

            gates.add("NON_FINITE")

        elif latency < 0:

            gates.add("METRIC_RANGE")

        # -----------------------------------------------------
        # SIZE
        # -----------------------------------------------------

        size = evaluation.get("sizeBytes")

        if not is_finite_number(size):

            gates.add("NON_FINITE")

        elif not is_safe_integer(size):

            gates.add("METRIC_RANGE")

        # -----------------------------------------------------
        # IMMUTABLE ARTIFACT LINEAGE
        # -----------------------------------------------------

        if (
            not isinstance(
                item.get("artifactDigest"),
                str
            )
            or not isinstance(
                evaluation.get("artifactDigest"),
                str
            )
            or evaluation["artifactDigest"]
            != item["artifactDigest"]
        ):

            gates.add("ARTIFACT_MISMATCH")

        # -----------------------------------------------------
        # DATASET LINEAGE
        # -----------------------------------------------------

        if (
            not isinstance(
                evaluation.get("datasetDigest"),
                str
            )
            or evaluation["datasetDigest"]
            != policy["datasetDigest"]
        ):

            gates.add("DATASET_MISMATCH")

        # -----------------------------------------------------
        # SCHEMA LINEAGE
        # -----------------------------------------------------

        if (
            not isinstance(
                evaluation.get("schemaDigest"),
                str
            )
            or evaluation["schemaDigest"]
            != policy["schemaDigest"]
        ):

            gates.add("SCHEMA_MISMATCH")

        # -----------------------------------------------------
        # ACCURACY FLOOR
        # -----------------------------------------------------

        if (
            is_finite_number(accuracy)
            and 0 <= accuracy <= 1
            and accuracy < policy["accuracyFloor"]
        ):

            gates.add("ACCURACY_FLOOR")

        # -----------------------------------------------------
        # LATENCY LIMIT
        # -----------------------------------------------------

        if (
            is_finite_number(latency)
            and latency >= 0
            and latency > policy["maxLatencyMs"]
        ):

            gates.add("LATENCY_LIMIT")

        # -----------------------------------------------------
        # SIZE LIMIT
        # -----------------------------------------------------

        if (
            is_safe_integer(size)
            and size > policy["maxSizeBytes"]
        ):

            gates.add("SIZE_LIMIT")

        # -----------------------------------------------------
        # REQUIRED SLICES
        # -----------------------------------------------------

        slices = evaluation.get("slices")

        if not isinstance(slices, dict):
            slices = {}

        for name, floor in policy[
            "requiredSlices"
        ].items():

            if name not in slices:

                gates.add(
                    f"MISSING_SLICE:{name}"
                )

                continue

            value = slices[name]

            if not is_finite_number(value):

                gates.add(
                    f"SLICE_RANGE:{name}"
                )

                continue

            if value < 0 or value > 1:

                gates.add(
                    f"SLICE_RANGE:{name}"
                )

                continue

            if value < floor:

                gates.add(
                    f"SLICE_FLOOR:{name}"
                )

        return gates

    # =========================================================
    # EVALUATE EVERY UNIQUE VALID VERSION
    # =========================================================

    eligible_items = []

    for version, item in lookup.items():

        gates = evaluate(item)

        if gates:

            failed.setdefault(
                version,
                set()
            ).update(gates)

        else:

            eligible_items.append(item)

    # =========================================================
    # CHAMPION EVIDENCE MUST BE VALID
    # =========================================================

    champion_item = lookup.get(champion)

    if champion_item is None:

        return response(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    champion_gates = evaluate(
        champion_item
    )

    if champion_gates:

        failed.setdefault(
            champion,
            set()
        ).update(champion_gates)

        # Still rank the versions that actually passed.
        ranked = sorted(
            eligible_items,
            key=lambda item: (
                -item["evaluation"]["accuracy"],
                item["evaluation"]["latencyMs"],
                item["evaluation"]["sizeBytes"],
                int(item["version"])
            )
        )

        return response(
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
    # DETERMINISTIC ELIGIBLE RANKING
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
    # CHAMPION ALREADY BEST
    # =========================================================

    if not ranked:

        return response(
            "retain",
            champion,
            champion,
            [],
            failed,
            None,
            champion_evidence
        )

    winner = ranked[0]

    if winner["version"] == champion:

        return response(
            "retain",
            champion,
            champion,
            eligible_versions,
            failed,
            None,
            champion_evidence
        )

    # =========================================================
    # IMPROVEMENT — EXACTLY 12 DECIMAL PLACES
    # =========================================================

    improvement = round(
        winner["evaluation"]["accuracy"]
        - champion_evidence["accuracy"],
        12
    )

    # =========================================================
    # PROMOTION
    # =========================================================

    if improvement >= policy["minImprovement"]:

        return response(
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

    return response(
        "retain",
        champion,
        champion,
        eligible_versions,
        failed,
        None,
        champion_evidence
    )
