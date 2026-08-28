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


def bad():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


def parse_ts(value):
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


def safe_nonnegative_int(value):
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
        n = int(value)
        return 1 <= n <= MAX_SAFE_INT
    except Exception:
        return False


def add(failed, version, code):
    if version not in failed:
        failed[version] = set()
    failed[version].add(code)


def code_sort(codes):
    return sorted(
        codes,
        key=lambda x: x.encode("utf-8")
    )


def failed_json(failed):
    return {
        version: code_sort(codes)
        for version, codes in failed.items()
        if codes
    }


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
        "failedGates": failed_json(failed),
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
    # TOP-LEVEL INPUT
    # =========================================================

    try:
        data = await request.json()
    except Exception:
        return bad()

    if not isinstance(data, dict):
        return bad()

    if "policy" not in data:
        return bad()

    if "versions" not in data:
        return bad()

    if not isinstance(data["versions"], list):
        return bad()

    if "championVersion" not in data:
        return bad()

    if not isinstance(data["championVersion"], str):
        return bad()

    if "asOf" not in data:
        return bad()

    if not isinstance(data["asOf"], str):
        return bad()

    as_of = parse_ts(data["asOf"])

    if as_of is None:
        return bad()

    policy = data["policy"]
    versions = data["versions"]
    champion = data["championVersion"]

    # =========================================================
    # POLICY
    # =========================================================

    policy_valid = True

    if not isinstance(policy, dict):
        policy_valid = False

    required_policy = (
        "datasetDigest",
        "schemaDigest",
        "maxAgeSeconds",
        "accuracyFloor",
        "requiredSlices",
        "maxLatencyMs",
        "maxSizeBytes",
        "minImprovement",
    )

    if policy_valid:
        for key in required_policy:
            if key not in policy:
                policy_valid = False
                break

    if policy_valid:

        if (
            not isinstance(policy["datasetDigest"], str)
            or not policy["datasetDigest"]
        ):
            policy_valid = False

        if (
            not isinstance(policy["schemaDigest"], str)
            or not policy["schemaDigest"]
        ):
            policy_valid = False

        if not safe_nonnegative_int(
            policy["maxAgeSeconds"]
        ):
            policy_valid = False

        if (
            not finite_number(policy["accuracyFloor"])
            or not 0 <= policy["accuracyFloor"] <= 1
        ):
            policy_valid = False

        if not isinstance(
            policy["requiredSlices"],
            dict
        ):
            policy_valid = False

        if (
            not finite_number(policy["maxLatencyMs"])
            or policy["maxLatencyMs"] < 0
        ):
            policy_valid = False

        if not safe_nonnegative_int(
            policy["maxSizeBytes"]
        ):
            policy_valid = False

        if (
            not finite_number(policy["minImprovement"])
            or not 0 <= policy["minImprovement"] <= 1
        ):
            policy_valid = False

    if policy_valid:

        for name, floor in policy[
            "requiredSlices"
        ].items():

            if (
                not isinstance(name, str)
                or not finite_number(floor)
                or floor < 0
                or floor > 1
            ):
                policy_valid = False
                break

    # =========================================================
    # FAILED GATES
    # =========================================================

    failed = {}

    # =========================================================
    # CANONICAL + UNIQUE VERSIONS
    # =========================================================

    occurrences = {}
    valid_items = []

    for item in versions:

        if not isinstance(item, dict):

            add(
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

            add(
                failed,
                key,
                "INVALID_VERSION"
            )

            continue

        occurrences[version] = (
            occurrences.get(version, 0) + 1
        )

        valid_items.append(item)

    duplicates = {
        version
        for version, count in occurrences.items()
        if count > 1
    }

    for version in duplicates:
        add(
            failed,
            version,
            "DUPLICATE_VERSION"
        )

    # =========================================================
    # INVALID POLICY IS A GATE
    # =========================================================

    if not policy_valid:

        for version in occurrences:
            add(
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

    if not canonical_version(champion):

        add(
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

    if champion not in occurrences:

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
    # LOOKUP
    # =========================================================

    lookup = {}

    for item in valid_items:

        version = item["version"]

        if version in duplicates:
            continue

        lookup[version] = item

    # =========================================================
    # TIME WINDOW
    # =========================================================

    oldest_allowed = (
        as_of
        - timedelta(
            seconds=policy["maxAgeSeconds"]
        )
    )

    # =========================================================
    # EVALUATE ONE VERSION
    # =========================================================

    def gates_for(item):

        gates = set()

        if "evaluation" not in item:
            gates.add("MISSING_EVALUATION")
            return gates

        ev = item["evaluation"]

        if not isinstance(ev, dict):
            gates.add("MISSING_EVALUATION")
            return gates

        # -----------------------------------------------------
        # TIMESTAMP
        # -----------------------------------------------------

        created = parse_ts(
            ev.get("createdAt")
        )

        if created is None:

            gates.add("INVALID_TIMESTAMP")

        else:

            if created > as_of:
                gates.add("FUTURE_EVALUATION")

            elif created < oldest_allowed:
                gates.add("STALE_EVALUATION")

        # -----------------------------------------------------
        # ACCURACY
        # -----------------------------------------------------

        accuracy = ev.get("accuracy")

        if not finite_number(accuracy):

            gates.add("NON_FINITE")

        elif accuracy < 0 or accuracy > 1:

            gates.add("METRIC_RANGE")

        # -----------------------------------------------------
        # LATENCY
        # -----------------------------------------------------

        latency = ev.get("latencyMs")

        if not finite_number(latency):

            gates.add("NON_FINITE")

        elif latency < 0:

            gates.add("METRIC_RANGE")

        # -----------------------------------------------------
        # SIZE
        # -----------------------------------------------------

        size = ev.get("sizeBytes")

        if not finite_number(size):

            gates.add("NON_FINITE")

        elif not safe_nonnegative_int(size):

            gates.add("METRIC_RANGE")

        # -----------------------------------------------------
        # ARTIFACT
        # -----------------------------------------------------

        if (
            not isinstance(
                item.get("artifactDigest"),
                str
            )
            or not isinstance(
                ev.get("artifactDigest"),
                str
            )
            or ev.get("artifactDigest")
            != item.get("artifactDigest")
        ):

            gates.add(
                "ARTIFACT_MISMATCH"
            )

        # -----------------------------------------------------
        # DATASET
        # -----------------------------------------------------

        if (
            not isinstance(
                ev.get("datasetDigest"),
                str
            )
            or ev.get("datasetDigest")
            != policy["datasetDigest"]
        ):

            gates.add(
                "DATASET_MISMATCH"
            )

        # -----------------------------------------------------
        # SCHEMA
        # -----------------------------------------------------

        if (
            not isinstance(
                ev.get("schemaDigest"),
                str
            )
            or ev.get("schemaDigest")
            != policy["schemaDigest"]
        ):

            gates.add(
                "SCHEMA_MISMATCH"
            )

        # -----------------------------------------------------
        # ACCURACY FLOOR
        # -----------------------------------------------------

        if (
            finite_number(accuracy)
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
            finite_number(latency)
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
            safe_nonnegative_int(size)
            and size > policy["maxSizeBytes"]
        ):

            gates.add(
                "SIZE_LIMIT"
            )

        # -----------------------------------------------------
        # SLICES
        # -----------------------------------------------------

        slices = ev.get("slices")

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
    # EVALUATE
    # =========================================================

    eligible_items = []

    for version, item in lookup.items():

        gates = gates_for(item)

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

    champion_item = lookup.get(champion)

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

    champion_gates = gates_for(
        champion_item
    )

    if champion_gates:

        failed.setdefault(
            champion,
            set()
        ).update(champion_gates)

        # Only eligible versions are ranked.
        ranked = sorted(
            eligible_items,
            key=lambda x: (
                -x["evaluation"]["accuracy"],
                x["evaluation"]["latencyMs"],
                x["evaluation"]["sizeBytes"],
                int(x["version"])
            )
        )

        return result(
            "block",
            champion,
            None,
            [x["version"] for x in ranked],
            failed,
            None,
            None
        )

    champion_ev = champion_item["evaluation"]

    # =========================================================
    # RANK ALL ELIGIBLE
    # =========================================================

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

    # =========================================================
    # NO ELIGIBLE CHALLENGER
    # =========================================================

    if not ranked:

        return result(
            "retain",
            champion,
            champion,
            [],
            failed,
            None,
            champion_ev
        )

    winner = ranked[0]

    # =========================================================
    # CHAMPION IS ALREADY WINNER
    # =========================================================

    if winner["version"] == champion:

        return result(
            "retain",
            champion,
            champion,
            eligible_versions,
            failed,
            None,
            champion_ev
        )

    # =========================================================
    # IMPROVEMENT
    # =========================================================

    improvement = round(
        winner["evaluation"]["accuracy"]
        - champion_ev["accuracy"],
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
        champion_ev
    )
