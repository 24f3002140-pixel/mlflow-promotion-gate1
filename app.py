from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
import math
import re

app = FastAPI()

SAFE_INT_MAX = 9007199254740991

TS_RE = re.compile(
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

    if not TS_RE.fullmatch(value):
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


def add_gate(failed, version, code):
    failed.setdefault(version, set()).add(code)


def sorted_utf8(values):
    return sorted(
        set(values),
        key=lambda x: x.encode("utf-8")
    )


def make_failed(failed, input_versions):
    """
    The response must contain every input version.
    Versions with no failures have [].
    """
    output = {}

    for version in input_versions:
        if version not in output:
            output[version] = []

        if version in failed:
            output[version] = sorted_utf8(
                failed[version]
            )

    return output


def response(
    action,
    champion,
    selected,
    eligible,
    failed,
    input_versions,
    mutation,
    evidence
):
    return {
        "action": action,
        "championVersion": champion,
        "selectedVersion": selected,
        "eligibleVersions": eligible,
        "failedGates": make_failed(
            failed,
            input_versions
        ),
        "aliasMutation": mutation,
        "evidence": evidence
    }


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "MLflow Model Promotion Gate"
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

    # =========================================================
    # PRESERVE EVERY INPUT VERSION FOR failedGates
    # =========================================================

    input_versions = []

    for item in versions:
        if isinstance(item, dict):
            v = item.get("version")

            if isinstance(v, str):
                input_versions.append(v)
            else:
                input_versions.append("<invalid>")
        else:
            input_versions.append("<invalid>")

    failed = {}

    # =========================================================
    # VERSION VALIDATION
    # =========================================================

    counts = {}
    canonical_items = []

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

        counts[version] = counts.get(
            version,
            0
        ) + 1

        canonical_items.append(item)

    # Every occurrence of a duplicate is invalid.
    duplicate_versions = {
        version
        for version, count in counts.items()
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

    policy_valid = isinstance(policy, dict)

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
            not isinstance(
                policy["datasetDigest"],
                str
            )
            or policy["datasetDigest"] == ""
        ):
            policy_valid = False

        if (
            not isinstance(
                policy["schemaDigest"],
                str
            )
            or policy["schemaDigest"] == ""
        ):
            policy_valid = False

        if not safe_integer(
            policy["maxAgeSeconds"]
        ):
            policy_valid = False

        if (
            not finite_number(
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
            not finite_number(
                policy["maxLatencyMs"]
            )
            or policy["maxLatencyMs"] < 0
        ):
            policy_valid = False

        if not safe_integer(
            policy["maxSizeBytes"]
        ):
            policy_valid = False

        if (
            not finite_number(
                policy["minImprovement"]
            )
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
                not finite_number(floor)
                or floor < 0
                or floor > 1
            ):
                policy_valid = False
                break

    if not policy_valid:

        for version in counts:
            add_gate(
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
            input_versions,
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

        return response(
            "block",
            champion,
            None,
            [],
            failed,
            input_versions,
            None,
            None
        )

    if champion not in counts:

        return response(
            "block",
            champion,
            None,
            [],
            failed,
            input_versions,
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
            input_versions,
            None,
            None
        )

    # =========================================================
    # LOOKUP MAP ONLY AFTER DUPLICATE VALIDATION
    # =========================================================

    lookup = {}

    for item in canonical_items:

        version = item["version"]

        if version in duplicate_versions:
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

        accuracy_finite = finite_number(
            accuracy
        )

        latency_finite = finite_number(
            latency
        )

        size_finite = finite_number(
            size
        )

        if not accuracy_finite:
            gates.add("NON_FINITE")

        if not latency_finite:
            gates.add("NON_FINITE")

        if not size_finite:
            gates.add("NON_FINITE")

        # -----------------------------------------------------
        # RANGE VALIDATION
        # -----------------------------------------------------

        if accuracy_finite:

            if accuracy < 0 or accuracy > 1:
                gates.add("METRIC_RANGE")

        if latency_finite:

            if latency < 0:
                gates.add("METRIC_RANGE")

        if size_finite:

            if (
                not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or size > SAFE_INT_MAX
            ):
                gates.add("METRIC_RANGE")

        # -----------------------------------------------------
        # IMMUTABLE ARTIFACT LINEAGE
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
        # DATASET LINEAGE
        # -----------------------------------------------------

        if (
            evaluation.get("datasetDigest")
            != policy["datasetDigest"]
        ):
            gates.add("DATASET_MISMATCH")

        # -----------------------------------------------------
        # SCHEMA LINEAGE
        # -----------------------------------------------------

        if (
            evaluation.get("schemaDigest")
            != policy["schemaDigest"]
        ):
            gates.add("SCHEMA_MISMATCH")

        # -----------------------------------------------------
        # ACCURACY
        # -----------------------------------------------------

        if (
            accuracy_finite
            and 0 <= accuracy <= 1
            and accuracy < policy["accuracyFloor"]
        ):
            gates.add("ACCURACY_FLOOR")

        # -----------------------------------------------------
        # LATENCY
        # -----------------------------------------------------

        if (
            latency_finite
            and latency >= 0
            and latency > policy["maxLatencyMs"]
        ):
            gates.add("LATENCY_LIMIT")

        # -----------------------------------------------------
        # SIZE
        # -----------------------------------------------------

        if (
            isinstance(size, int)
            and not isinstance(size, bool)
            and 0 <= size <= SAFE_INT_MAX
            and size > policy["maxSizeBytes"]
        ):
            gates.add("SIZE_LIMIT")

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
                    "MISSING_SLICE:" + name
                )

                continue

            value = slices[name]

            if (
                not finite_number(value)
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
    # EVALUATE ALL UNIQUE CANONICAL VERSIONS
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
    # CHAMPION EVIDENCE MUST BE VALID
    # =========================================================

    champion_item = lookup.get(
        champion
    )

    if champion_item is None:

        return response(
            "block",
            champion,
            None,
            [],
            failed,
            input_versions,
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
            input_versions,
            None,
            None
        )

    champion_evidence = (
        champion_item["evaluation"]
    )

    # =========================================================
    # DETERMINISTIC ELIGIBLE RANKING
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
    # CHAMPION IS BEST
    # =========================================================

    if ranked and ranked[0]["version"] == champion:

        return response(
            "retain",
            champion,
            champion,
            eligible_versions,
            failed,
            input_versions,
            None,
            champion_evidence
        )

    # =========================================================
    # NO OTHER ELIGIBLE MODEL
    # =========================================================

    if not ranked:

        return response(
            "retain",
            champion,
            champion,
            [],
            failed,
            input_versions,
            None,
            champion_evidence
        )

    winner = ranked[0]

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

        return response(
            "promote",
            champion,
            winner["version"],
            eligible_versions,
            failed,
            input_versions,
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
        input_versions,
        None,
        champion_evidence
    )
