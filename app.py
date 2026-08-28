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

    if TIMESTAMP_RE.fullmatch(value) is None:
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


def safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def canonical_version(value):
    if not isinstance(value, str):
        return False

    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        return False

    try:
        number = int(value)
    except Exception:
        return False

    return 1 <= number <= MAX_SAFE_INTEGER


def add_gate(failed, version, gate):
    failed.setdefault(version, set()).add(gate)


def utf8_sorted(values):
    return sorted(set(values), key=lambda x: x.encode("utf-8"))


def failed_output(failed):
    return {
        version: utf8_sorted(gates)
        for version, gates in failed.items()
        if gates
    }


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
        "failedGates": failed_output(failed),
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
    # FIRST PASS:
    # Reject every duplicate/noncanonical version BEFORE
    # constructing the lookup map.
    # =========================================================

    canonical_entries = []
    occurrences = {}

    # Keep all supplied version identifiers so INVALID_POLICY
    # can be attached to every actual version occurrence.
    supplied_version_keys = []

    for item in versions:

        if not isinstance(item, dict):
            add_gate(
                failed,
                "<invalid>",
                "INVALID_VERSION"
            )
            continue

        version = item.get("version")

        if isinstance(version, str):
            supplied_version_keys.append(version)
        else:
            supplied_version_keys.append("<invalid>")

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

        occurrences[version] = (
            occurrences.get(version, 0) + 1
        )

        canonical_entries.append(item)

    duplicate_versions = {
        version
        for version, count in occurrences.items()
        if count > 1
    }

    for version in duplicate_versions:
        add_gate(
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
        for key in required_policy:
            if key not in policy:
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

        if not safe_integer(policy["maxAgeSeconds"]):
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

        if not safe_integer(policy["maxSizeBytes"]):
            policy_valid = False

        if (
            not finite(policy["minImprovement"])
            or not 0 <= policy["minImprovement"] <= 1
        ):
            policy_valid = False

    if policy_valid:

        for name, floor in policy["requiredSlices"].items():

            if (
                not isinstance(name, str)
                or not finite(floor)
                or floor < 0
                or floor > 1
            ):
                policy_valid = False
                break

    # A semantically invalid policy is a promotion gate,
    # NOT an HTTP 400.
    if not policy_valid:

        # Every version supplied by the caller receives
        # INVALID_POLICY in addition to any version-specific
        # validation failure.
        for key in supplied_version_keys:
            add_gate(
                failed,
                key,
                "INVALID_POLICY"
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

    # =========================================================
    # CHAMPION VALIDATION
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

    if champion not in occurrences:

        return make_response(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    if champion in duplicate_versions:

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
    # Only unique canonical versions reach this map.
    # =========================================================

    lookup = {}

    for item in canonical_entries:

        version = item["version"]

        if version in duplicate_versions:
            continue

        lookup[version] = item

    # =========================================================
    # TIME WINDOW
    # =========================================================

    oldest_allowed = (
        as_of -
        timedelta(
            seconds=policy["maxAgeSeconds"]
        )
    )

    # =========================================================
    # EVIDENCE VALIDATION
    # =========================================================

    def validate(item):

        gates = set()

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

        created_at = parse_timestamp(
            evaluation.get("createdAt")
        )

        if created_at is None:

            gates.add("INVALID_TIMESTAMP")

        else:

            # Future has its own code.
            if created_at > as_of:
                gates.add("FUTURE_EVALUATION")

            # Boundary is valid:
            # asOf - maxAgeSeconds <= createdAt
            if created_at < oldest_allowed:
                gates.add("STALE_EVALUATION")

        # -----------------------------------------------------
        # ACCURACY
        # -----------------------------------------------------

        accuracy = evaluation.get("accuracy")

        if not finite(accuracy):

            gates.add("NON_FINITE")

        elif accuracy < 0 or accuracy > 1:

            gates.add("METRIC_RANGE")

        # -----------------------------------------------------
        # LATENCY
        # -----------------------------------------------------

        latency = evaluation.get("latencyMs")

        if not finite(latency):

            gates.add("NON_FINITE")

        elif latency < 0:

            gates.add("METRIC_RANGE")

        # -----------------------------------------------------
        # SIZE
        # -----------------------------------------------------

        size = evaluation.get("sizeBytes")

        if not finite(size):

            gates.add("NON_FINITE")

        elif not safe_integer(size):

            gates.add("METRIC_RANGE")

        # -----------------------------------------------------
        # ARTIFACT LINEAGE
        # -----------------------------------------------------

        if (
            not isinstance(item.get("artifactDigest"), str)
            or not isinstance(
                evaluation.get("artifactDigest"),
                str
            )
            or evaluation.get("artifactDigest")
            != item.get("artifactDigest")
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
            or evaluation.get("datasetDigest")
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
            or evaluation.get("schemaDigest")
            != policy["schemaDigest"]
        ):

            gates.add("SCHEMA_MISMATCH")

        # -----------------------------------------------------
        # ACCURACY FLOOR
        # -----------------------------------------------------

        if (
            finite(accuracy)
            and 0 <= accuracy <= 1
            and accuracy < policy["accuracyFloor"]
        ):

            gates.add("ACCURACY_FLOOR")

        # -----------------------------------------------------
        # LATENCY LIMIT
        # -----------------------------------------------------

        if (
            finite(latency)
            and latency >= 0
            and latency > policy["maxLatencyMs"]
        ):

            gates.add("LATENCY_LIMIT")

        # -----------------------------------------------------
        # SIZE LIMIT
        # -----------------------------------------------------

        if (
            safe_integer(size)
            and size > policy["maxSizeBytes"]
        ):

            gates.add("SIZE_LIMIT")

        # -----------------------------------------------------
        # REQUIRED SLICES
        # -----------------------------------------------------

        slices = evaluation.get("slices")

        if not isinstance(slices, dict):
            slices = {}

        for name, floor in policy["requiredSlices"].items():

            if name not in slices:

                gates.add(
                    "MISSING_SLICE:" + name
                )
                continue

            value = slices[name]

            if (
                not finite(value)
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
    # EVALUATE EVERY UNIQUE CANONICAL VERSION
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
    # DETERMINISTIC RANKING
    #
    # accuracy DESC
    # latency ASC
    # size ASC
    # numeric version ASC
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
    # CHAMPION EVIDENCE
    # =========================================================

    champion_item = lookup.get(champion)

    if champion_item is None:

        return make_response(
            "block",
            champion,
            None,
            eligible_versions,
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
        ).update(champion_gates)

        return make_response(
            "block",
            champion,
            None,
            eligible_versions,
            failed,
            None,
            None
        )

    champion_evidence = (
        champion_item["evaluation"]
    )

    # =========================================================
    # NO ELIGIBLE CHALLENGER
    # =========================================================

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
    # CHAMPION IS ALREADY TOP RANKED
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
