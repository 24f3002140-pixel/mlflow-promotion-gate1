from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
import math
import re

app = FastAPI()

MAX_SAFE_INT = 9007199254740991

TS_RE = re.compile(
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
        and 0 <= value <= MAX_SAFE_INT
    )


def valid_version(value):
    if not isinstance(value, str):
        return False

    if not re.fullmatch(r"[1-9][0-9]*", value):
        return False

    try:
        return 1 <= int(value) <= MAX_SAFE_INT
    except Exception:
        return False


def add_gate(failed, version, gate):
    failed.setdefault(version, set()).add(gate)


def sort_codes(codes):
    return sorted(
        set(codes),
        key=lambda x: x.encode("utf-8")
    )


def failed_output(failed):
    out = {}

    for version, codes in failed.items():
        if codes:
            out[version] = sort_codes(codes)

    return out


def make_result(
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
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/promote")
async def promote(request: Request):

    # ========================================================
    # BASIC INPUT
    # ========================================================

    try:
        body = await request.json()
    except Exception:
        return bad_input()

    if not isinstance(body, dict):
        return bad_input()

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

    as_of = parse_time(body["asOf"])

    if as_of is None:
        return bad_input()

    policy = body["policy"]
    versions = body["versions"]
    champion = body["championVersion"]

    # ========================================================
    # FAILED GATES
    # ========================================================

    failed = {}

    # ========================================================
    # VERSION VALIDATION
    # ========================================================

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

        if not valid_version(version):

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

        valid_items.append(item)

    # ========================================================
    # DUPLICATE VERSIONS
    # ========================================================

    duplicates = {
        v for v, count in counts.items()
        if count > 1
    }

    for v in duplicates:
        add_gate(
            failed,
            v,
            "DUPLICATE_VERSION"
        )

    # ========================================================
    # POLICY VALIDATION
    #
    # Missing policy itself was already handled above.
    # Invalid policy contents are gate failures.
    # ========================================================

    policy_valid = True

    required = [
        "datasetDigest",
        "schemaDigest",
        "maxAgeSeconds",
        "accuracyFloor",
        "requiredSlices",
        "maxLatencyMs",
        "maxSizeBytes",
        "minImprovement"
    ]

    if not isinstance(policy, dict):
        policy_valid = False

    if policy_valid:

        for field in required:
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

        if not safe_int(
            policy["maxAgeSeconds"]
        ):
            policy_valid = False

        if (
            not finite(policy["accuracyFloor"])
            or not 0 <= policy["accuracyFloor"] <= 1
        ):
            policy_valid = False

        if not isinstance(
            policy["requiredSlices"],
            dict
        ):
            policy_valid = False

        if (
            not finite(policy["maxLatencyMs"])
            or policy["maxLatencyMs"] < 0
        ):
            policy_valid = False

        if not safe_int(
            policy["maxSizeBytes"]
        ):
            policy_valid = False

        if (
            not finite(policy["minImprovement"])
            or not 0 <= policy["minImprovement"] <= 1
        ):
            policy_valid = False

    if policy_valid:

        for name, floor in policy[
            "requiredSlices"
        ].items():

            if (
                not isinstance(name, str)
                or not finite(floor)
                or floor < 0
                or floor > 1
            ):
                policy_valid = False
                break

    # ========================================================
    # INVALID POLICY = BLOCK + INVALID_POLICY GATES
    # ========================================================

    if not policy_valid:

        for version in counts:
            add_gate(
                failed,
                version,
                "INVALID_POLICY"
            )

        return make_result(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    # ========================================================
    # CHAMPION VALIDATION
    # ========================================================

    if not valid_version(champion):

        add_gate(
            failed,
            champion,
            "INVALID_VERSION"
        )

        return make_result(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    if champion not in counts:

        return make_result(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    if champion in duplicates:

        return make_result(
            "block",
            champion,
            None,
            [],
            failed,
            None,
            None
        )

    # ========================================================
    # LOOKUP AFTER DUPLICATE REJECTION
    # ========================================================

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

    # ========================================================
    # EVIDENCE VALIDATION
    # ========================================================

    def validate(item):

        gates = set()

        if "evaluation" not in item:
            gates.add("MISSING_EVALUATION")
            return gates

        ev = item["evaluation"]

        if not isinstance(ev, dict):
            gates.add("MISSING_EVALUATION")
            return gates

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        created = parse_time(
            ev.get("createdAt")
        )

        if created is None:

            gates.add("INVALID_TIMESTAMP")

        else:

            if created > as_of:
                gates.add("FUTURE_EVALUATION")

            if created < cutoff:
                gates.add("STALE_EVALUATION")

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = ev.get("accuracy")
        latency = ev.get("latencyMs")
        size = ev.get("sizeBytes")

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

        elif not safe_int(size):

            gates.add("METRIC_RANGE")

        # ----------------------------------------------------
        # ARTIFACT
        # ----------------------------------------------------

        if (
            ev.get("artifactDigest")
            != item.get("artifactDigest")
        ):

            gates.add(
                "ARTIFACT_MISMATCH"
            )

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        if (
            ev.get("datasetDigest")
            != policy["datasetDigest"]
        ):

            gates.add(
                "DATASET_MISMATCH"
            )

        # ----------------------------------------------------
        # SCHEMA
        # ----------------------------------------------------

        if (
            ev.get("schemaDigest")
            != policy["schemaDigest"]
        ):

            gates.add(
                "SCHEMA_MISMATCH"
            )

        # ----------------------------------------------------
        # ACCURACY FLOOR
        # ----------------------------------------------------

        if (
            finite(accuracy)
            and 0 <= accuracy <= 1
            and accuracy < policy["accuracyFloor"]
        ):

            gates.add(
                "ACCURACY_FLOOR"
            )

        # ----------------------------------------------------
        # LATENCY LIMIT
        # ----------------------------------------------------

        if (
            finite(latency)
            and latency >= 0
            and latency > policy["maxLatencyMs"]
        ):

            gates.add(
                "LATENCY_LIMIT"
            )

        # ----------------------------------------------------
        # SIZE LIMIT
        # ----------------------------------------------------

        if (
            safe_int(size)
            and size > policy["maxSizeBytes"]
        ):

            gates.add(
                "SIZE_LIMIT"
            )

        # ----------------------------------------------------
        # SLICES
        # ----------------------------------------------------

        slices = ev.get("slices")

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

    # ========================================================
    # VALIDATE ALL UNIQUE VERSIONS
    # ========================================================

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

    # ========================================================
    # CHAMPION EVIDENCE
    # ========================================================

    champion_item = lookup.get(
        champion
    )

    if champion_item is None:

        return make_result(
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
        ).update(champion_gates)

        ranked = sorted(
            eligible_items,
            key=lambda x: (
                -x["evaluation"]["accuracy"],
                x["evaluation"]["latencyMs"],
                x["evaluation"]["sizeBytes"],
                int(x["version"])
            )
        )

        return make_result(
            "block",
            champion,
            None,
            [x["version"] for x in ranked],
            failed,
            None,
            None
        )

    champion_evidence = (
        champion_item["evaluation"]
    )

    # ========================================================
    # RANKING
    # ========================================================

    ranked = sorted(
        eligible_items,
        key=lambda x: (
            -x["evaluation"]["accuracy"],
            x["evaluation"]["latencyMs"],
            x["evaluation"]["sizeBytes"],
            int(x["version"])
        )
    )

    eligible_versions = [
        x["version"]
        for x in ranked
    ]

    # ========================================================
    # CHAMPION RETAIN
    # ========================================================

    if ranked and ranked[0]["version"] == champion:

        return make_result(
            "retain",
            champion,
            champion,
            eligible_versions,
            failed,
            None,
            champion_evidence
        )

    # ========================================================
    # SAFETY
    # ========================================================

    if not ranked:

        return make_result(
            "retain",
            champion,
            champion,
            [],
            failed,
            None,
            champion_evidence
        )

    # ========================================================
    # BEST CHALLENGER
    # ========================================================

    challenger = ranked[0]

    improvement = round(
        challenger["evaluation"]["accuracy"]
        - champion_evidence["accuracy"],
        12
    )

    # ========================================================
    # PROMOTE
    # ========================================================

    if improvement >= policy["minImprovement"]:

        return make_result(
            "promote",
            champion,
            challenger["version"],
            eligible_versions,
            failed,
            {
                "alias": "champion",
                "version": challenger["version"]
            },
            challenger["evaluation"]
        )

    # ========================================================
    # RETAIN
    # ========================================================

    return make_result(
        "retain",
        champion,
        champion,
        eligible_versions,
        failed,
        None,
        champion_evidence
    )
