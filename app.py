from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
import math
import re

app = FastAPI()

MAX_SAFE_INT = 9007199254740991

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


def parse_instant(value):
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


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def nonnegative_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE_INT
    )


def canonical_version(value):
    if not isinstance(value, str):
        return False

    if not re.fullmatch(r"[1-9][0-9]*", value):
        return False

    try:
        number = int(value)
        return 1 <= number <= MAX_SAFE_INT
    except Exception:
        return False


def add_gate(failed, version, gate):
    failed.setdefault(version, set()).add(gate)


def utf8_sorted(values):
    return sorted(
        set(values),
        key=lambda x: x.encode("utf-8")
    )


def format_failed_gates(failed):
    output = {}

    for version, gates in failed.items():

        if not gates:
            continue

        output[version] = utf8_sorted(gates)

    return output


def make_response(
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
        "failedGates": format_failed_gates(failed),
        "aliasMutation": mutation,
        "evidence": evidence
    }


@app.get("/")
async def home():
    return {
        "status": "ok",
        "service": "MLflow Promotion Gate"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/promote")
async def promote(request: Request):

    # =========================================================
    # REQUEST VALIDATION
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

    as_of = parse_instant(body["asOf"])

    if as_of is None:
        return invalid_input()

    policy = body["policy"]
    versions = body["versions"]
    champion = body["championVersion"]

    # =========================================================
    # POLICY VALIDATION
    # =========================================================

    if not isinstance(policy, dict):
        return invalid_input()

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

    for field in required_policy:
        if field not in policy:
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

    if not nonnegative_safe_integer(
        policy["maxAgeSeconds"]
    ):
        return invalid_input()

    if (
        not finite(policy["accuracyFloor"])
        or policy["accuracyFloor"] < 0
        or policy["accuracyFloor"] > 1
    ):
        return invalid_input()

    if not isinstance(
        policy["requiredSlices"],
        dict
    ):
        return invalid_input()

    if (
        not finite(policy["maxLatencyMs"])
        or policy["maxLatencyMs"] < 0
    ):
        return invalid_input()

    if not nonnegative_safe_integer(
        policy["maxSizeBytes"]
    ):
        return invalid_input()

    if (
        not finite(policy["minImprovement"])
        or policy["minImprovement"] < 0
        or policy["minImprovement"] > 1
    ):
        return invalid_input()

    for name, floor in policy[
        "requiredSlices"
    ].items():

        if not isinstance(name, str):
            return invalid_input()

        if (
            not finite(floor)
            or floor < 0
            or floor > 1
        ):
            return invalid_input()

    # =========================================================
    # INITIAL FAILED GATES
    # =========================================================

    failed = {}

    # =========================================================
    # CANONICAL VERSION CHECK
    # =========================================================

    counts = {}

    valid_items = []

    for item in versions:

        if not isinstance(item, dict):

            add_gate(
                failed,
                "<invalid>",
                "INVALID_VERSION"
            )

            continue

        version = item.get("version")

        if not canonical_version(version):

            key = (
                version
                if isinstance(version, str)
                else "<invalid>"
            )

            add_gate(
                failed,
                key,
                "INVALID_VERSION"
            )

            continue

        counts[version] = (
            counts.get(version, 0) + 1
        )

        valid_items.append(item)

    # =========================================================
    # DUPLICATES
    # =========================================================

    duplicates = {
        version
        for version, count in counts.items()
        if count > 1
    }

    for version in duplicates:

        add_gate(
            failed,
            version,
            "DUPLICATE_VERSION"
        )

    # =========================================================
    # CHAMPION MUST BE VALID AND UNIQUE
    # =========================================================

    if not canonical_version(champion):

        add_gate(
            failed,
            champion,
            "INVALID_VERSION"
        )

        return make_response(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    if champion not in counts:

        return make_response(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    if champion in duplicates:

        return make_response(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    # =========================================================
    # LOOKUP MAP
    # =========================================================

    lookup = {}

    for item in valid_items:

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
    # EVIDENCE GATE
    # =========================================================

    def evaluate(item):

        gates = set()

        # -----------------------------------------------------
        # EVALUATION
        # -----------------------------------------------------

        if "evaluation" not in item:
            gates.add("MISSING_EVALUATION")
            return gates

        evaluation = item["evaluation"]

        if not isinstance(evaluation, dict):
            gates.add("MISSING_EVALUATION")
            return gates

        # -----------------------------------------------------
        # TIMESTAMP
        # -----------------------------------------------------

        created_at = evaluation.get(
            "createdAt"
        )

        created = parse_instant(
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
        # METRICS
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

        # Accuracy
        if not finite(accuracy):

            gates.add("NON_FINITE")

        elif accuracy < 0 or accuracy > 1:

            gates.add("METRIC_RANGE")

        # Latency
        if not finite(latency):

            gates.add("NON_FINITE")

        elif latency < 0:

            gates.add("METRIC_RANGE")

        # Size
        if not finite(size):

            gates.add("NON_FINITE")

        elif (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size > MAX_SAFE_INT
        ):

            gates.add("METRIC_RANGE")

        # -----------------------------------------------------
        # ARTIFACT LINEAGE
        # -----------------------------------------------------

        if (
            evaluation.get("artifactDigest")
            != item.get("artifactDigest")
        ):

            gates.add(
                "ARTIFACT_MISMATCH"
            )

        # -----------------------------------------------------
        # DATASET LINEAGE
        # -----------------------------------------------------

        if (
            evaluation.get("datasetDigest")
            != policy["datasetDigest"]
        ):

            gates.add(
                "DATASET_MISMATCH"
            )

        # -----------------------------------------------------
        # SCHEMA LINEAGE
        # -----------------------------------------------------

        if (
            evaluation.get("schemaDigest")
            != policy["schemaDigest"]
        ):

            gates.add(
                "SCHEMA_MISMATCH"
            )

        # -----------------------------------------------------
        # ACCURACY FLOOR
        # -----------------------------------------------------

        if (
            finite(accuracy)
            and 0 <= accuracy <= 1
            and accuracy < policy["accuracyFloor"]
        ):

            gates.add(
                "ACCURACY_FLOOR"
            )

        # -----------------------------------------------------
        # LATENCY LIMIT
        # -----------------------------------------------------

        if (
            finite(latency)
            and latency >= 0
            and latency > policy["maxLatencyMs"]
        ):

            gates.add(
                "LATENCY_LIMIT"
            )

        # -----------------------------------------------------
        # SIZE LIMIT
        # -----------------------------------------------------

        if (
            isinstance(size, int)
            and not isinstance(size, bool)
            and 0 <= size <= MAX_SAFE_INT
            and size > policy["maxSizeBytes"]
        ):

            gates.add(
                "SIZE_LIMIT"
            )

        # -----------------------------------------------------
        # REQUIRED SLICES
        # -----------------------------------------------------

        slices = evaluation.get(
            "slices"
        )

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

            if not finite(value):

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
    # EVALUATE ALL UNIQUE VERSIONS
    # =========================================================

    eligible_items = []

    for version, item in lookup.items():

        gates = evaluate(item)

        if gates:

            add = failed.setdefault(
                version,
                set()
            )

            add.update(gates)

        else:

            eligible_items.append(item)

    # =========================================================
    # CHAMPION EVIDENCE
    # =========================================================

    champion_item = lookup.get(
        champion
    )

    if champion_item is None:

        return make_response(
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
        ).update(
            champion_gates
        )

        # Rank only eligible versions.
        ranked = sorted(
            eligible_items,
            key=lambda item: (
                -item["evaluation"]["accuracy"],
                item["evaluation"]["latencyMs"],
                item["evaluation"]["sizeBytes"],
                int(item["version"])
            )
        )

        return make_response(
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
    # DETERMINISTIC RANKING
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

    # Champion must itself be eligible here.
    if not ranked:

        return make_response(
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
    # CHAMPION WINS
    # =========================================================

    if winner["version"] == champion:

        return make_response(
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

        return make_response(
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

    return make_response(
        "retain",
        champion,
        champion,
        eligible_versions,
        failed,
        None,
        champion_evidence
    )
