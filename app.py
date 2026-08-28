from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
import math
import re

app = FastAPI()

MAX_SAFE = 9007199254740991

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


def parse_time(value):
    if not isinstance(value, str):
        return None

    if TS_RE.fullmatch(value) is None:
        return None

    try:
        s = value[:-1] + "+00:00" if value.endswith("Z") else value
        dt = datetime.fromisoformat(s)

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
        and 0 <= value <= MAX_SAFE
    )


def canonical_version(value):
    if not isinstance(value, str):
        return False

    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        return False

    try:
        n = int(value)
        return 1 <= n <= MAX_SAFE
    except Exception:
        return False


def add_failure(failed, version, code):
    failed.setdefault(version, set()).add(code)


def sorted_codes(codes):
    return sorted(set(codes), key=lambda x: x.encode("utf-8"))


def output_failures(failed):
    return {
        k: sorted_codes(v)
        for k, v in failed.items()
        if v
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
        "failedGates": output_failures(failed),
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

    # ---------------------------------------------------------
    # HTTP-INVALID INPUTS
    # ---------------------------------------------------------

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

    as_of = parse_time(body["asOf"])

    if as_of is None:
        return invalid_input()

    policy = body["policy"]
    versions = body["versions"]
    champion = body["championVersion"]

    failed = {}

    # ---------------------------------------------------------
    # CANONICAL / DUPLICATE VERSION PROCESSING
    # ---------------------------------------------------------

    canonical_entries = []
    counts = {}

    for item in versions:

        if not isinstance(item, dict):
            add_failure(
                failed,
                "<invalid>",
                "INVALID_VERSION"
            )
            continue

        version = item.get("version")

        if not canonical_version(version):

            # Preserve the supplied string exactly when possible.
            key = version if isinstance(version, str) else "<invalid>"

            add_failure(
                failed,
                key,
                "INVALID_VERSION"
            )
            continue

        counts[version] = counts.get(version, 0) + 1
        canonical_entries.append(item)

    duplicate_versions = {
        v for v, count in counts.items()
        if count > 1
    }

    # Every occurrence of a duplicated version is rejected.
    for version in duplicate_versions:
        add_failure(
            failed,
            version,
            "DUPLICATE_VERSION"
        )

    # ---------------------------------------------------------
    # POLICY
    # ---------------------------------------------------------

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
            or len(policy["datasetDigest"]) == 0
        ):
            policy_valid = False

        if (
            not isinstance(policy["schemaDigest"], str)
            or len(policy["schemaDigest"]) == 0
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

            if (
                not isinstance(name, str)
                or not finite(floor)
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

        return make_response(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    # ---------------------------------------------------------
    # CHAMPION ID
    # ---------------------------------------------------------

    if not canonical_version(champion):

        add_failure(
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

    # ---------------------------------------------------------
    # ONLY NOW CREATE LOOKUP MAP
    # ---------------------------------------------------------

    lookup = {}

    for item in canonical_entries:

        version = item["version"]

        if version in duplicate_versions:
            continue

        lookup[version] = item

    # ---------------------------------------------------------
    # AGE
    # ---------------------------------------------------------

    minimum_time = (
        as_of -
        timedelta(
            seconds=policy["maxAgeSeconds"]
        )
    )

    # ---------------------------------------------------------
    # EVIDENCE GATE FUNCTION
    # ---------------------------------------------------------

    def evidence_gates(item):

        gates = set()

        if "evaluation" not in item:
            gates.add("MISSING_EVALUATION")
            return gates

        evaluation = item["evaluation"]

        if not isinstance(evaluation, dict):
            gates.add("MISSING_EVALUATION")
            return gates

        # =====================================================
        # TIMESTAMP
        # =====================================================

        created_at = parse_time(
            evaluation.get("createdAt")
        )

        if created_at is None:

            gates.add("INVALID_TIMESTAMP")

        else:

            if created_at > as_of:
                gates.add("FUTURE_EVALUATION")

            if created_at < minimum_time:
                gates.add("STALE_EVALUATION")

        # =====================================================
        # METRICS
        # =====================================================

        accuracy = evaluation.get("accuracy")
        latency = evaluation.get("latencyMs")
        size = evaluation.get("sizeBytes")

        if not finite(accuracy):
            gates.add("NON_FINITE")
        elif accuracy < 0 or accuracy > 1:
            gates.add("METRIC_RANGE")

        if not finite(latency):
            gates.add("NON_FINITE")
        elif latency < 0:
            gates.add("METRIC_RANGE")

        if not finite(size):
            gates.add("NON_FINITE")
        elif (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size > MAX_SAFE
        ):
            gates.add("METRIC_RANGE")

        # =====================================================
        # IMMUTABLE ARTIFACT DIGEST
        # =====================================================

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

        # =====================================================
        # DATASET DIGEST
        # =====================================================

        if (
            not isinstance(
                evaluation.get("datasetDigest"),
                str
            )
            or evaluation.get("datasetDigest")
            != policy["datasetDigest"]
        ):
            gates.add("DATASET_MISMATCH")

        # =====================================================
        # SCHEMA DIGEST
        # =====================================================

        if (
            not isinstance(
                evaluation.get("schemaDigest"),
                str
            )
            or evaluation.get("schemaDigest")
            != policy["schemaDigest"]
        ):
            gates.add("SCHEMA_MISMATCH")

        # =====================================================
        # ACCURACY FLOOR
        # =====================================================

        if (
            finite(accuracy)
            and 0 <= accuracy <= 1
            and accuracy < policy["accuracyFloor"]
        ):
            gates.add("ACCURACY_FLOOR")

        # =====================================================
        # LATENCY
        # =====================================================

        if (
            finite(latency)
            and latency >= 0
            and latency > policy["maxLatencyMs"]
        ):
            gates.add("LATENCY_LIMIT")

        # =====================================================
        # SIZE
        # =====================================================

        if (
            isinstance(size, int)
            and not isinstance(size, bool)
            and 0 <= size <= MAX_SAFE
            and size > policy["maxSizeBytes"]
        ):
            gates.add("SIZE_LIMIT")

        # =====================================================
        # REQUIRED SLICES
        # =====================================================

        slices = evaluation.get("slices")

        if not isinstance(slices, dict):
            slices = {}

        for name, floor in policy["requiredSlices"].items():

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

        return gates

    # ---------------------------------------------------------
    # EVALUATE ALL UNIQUE VERSIONS
    # ---------------------------------------------------------

    eligible = []

    for version, item in lookup.items():

        gates = evidence_gates(item)

        if gates:

            for gate in gates:
                add_failure(
                    failed,
                    version,
                    gate
                )

        else:
            eligible.append(item)

    # ---------------------------------------------------------
    # CHAMPION EVIDENCE
    # ---------------------------------------------------------

    champion_item = lookup.get(champion)

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

    champion_gates = evidence_gates(
        champion_item
    )

    if champion_gates:

        for gate in champion_gates:
            add_failure(
                failed,
                champion,
                gate
            )

        # Even when champion is invalid, eligible versions
        # are still ranked deterministically.
        ranked = sorted(
            eligible,
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

    # ---------------------------------------------------------
    # RANK ELIGIBLE VERSIONS
    # ---------------------------------------------------------

    ranked = sorted(
        eligible,
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

    # ---------------------------------------------------------
    # NO ELIGIBLE VERSIONS
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # CHAMPION IS BEST
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # IMPROVEMENT
    # ---------------------------------------------------------

    improvement = round(
        winner["evaluation"]["accuracy"]
        - champion_evidence["accuracy"],
        12
    )

    # ---------------------------------------------------------
    # PROMOTE
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # RETAIN
    # ---------------------------------------------------------

    return make_response(
        "retain",
        champion,
        champion,
        eligible_versions,
        failed,
        None,
        champion_evidence
    )
