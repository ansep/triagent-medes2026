"""
Generate tool_forcing_instances_n100.jsonl and the corresponding ground truth entries.
Run: python experiments/generate_toolforcing.py
Outputs:
  datasets/tool_forcing_instances_n100.jsonl   (100 instances)
  datasets/ground_truth_toolforcing_n100.jsonl (100 GT entries, append to ground_truth.jsonl)

Ground truth is derived from the SAME source of truth as the production dataset
(experiments/build_dataset.py: PROCESSES + build_gateway_decisions), instead of being
hand-typed, so that treatment_substances / treatment_dose_per_ha / non-weather gateway
decisions are guaranteed consistent between the two datasets. See build_dataset.py's
_extract_substances removal / substances fix and the WEATHER_DEPENDENT_GATEWAYS split
for why this matters.

weather_check (and any other WEATHER_DEPENDENT_GATEWAYS, e.g. P4's mycotoxin_risk and
P5's vector_pressure) CANNOT be resolved here: in the tool-forcing dataset the weather
is deliberately absent from the instance description and is only known once the
Execution Agent's real get_weather call returns, at run time. Those keys are therefore
left OUT of gateway_decisions at generation time (build_gateway_decisions returns None
for them when weather_fav=None) and must be filled in per-backbone by
experiments/derive_toolforcing_gateway_gt.py, which reads the actual logged API
observation for each instance.

The instance-generation code below (descriptions, locations, stages, severities, weather
phrases) is UNCHANGED from the original script on purpose: tool_forcing_instances_n100.jsonl
must stay byte-identical to what the existing run manifests recorded as dataset_sha256,
or the already-collected results.jsonl logs can no longer be legitimately rescored against it.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import PROCESSES, build_gateway_decisions  # noqa: E402

WEATHER_PHRASES = [
    "Current meteorological conditions are unknown and must be retrieved via real-time API query before any fungicide application decision can be made.",
    "No weather data is available for this location at the time of inspection — live conditions must be checked before proceeding.",
    "The agronomist on site did not record temperature or humidity; real-time weather retrieval is required to assess disease risk.",
    "Environmental data (temperature, humidity, wind speed) is unavailable — an API call is required before scheduling treatment.",
]

# severity mapping: tool-forcing's 4-level scale -> production's 3-level
#    treatment_protocols keys (low/medium/high). "critical"/"severe"/"epidemic" tiers
#    fold into "high": production has no more-severe bucket to map them to.
SEVERITY_MAP = {
    "P1": {"low": "low", "medium": "medium", "high": "high", "critical": "high"},
    "P2": {"early": "low", "moderate": "medium", "severe": "high", "critical": "high"},
    "P3": {"threshold": "low", "moderate": "medium", "high": "high", "severe": "high"},
    "P4": {"low": "low", "moderate": "medium", "high": "high", "severe": "high"},
    "P5": {"suspect": "low", "confirmed": "medium", "spreading": "high", "epidemic": "high"},
}

# growth stage mapping: tool-forcing's 5 per-process stage strings -> production's
#    early/mid/late keys. Only P3 (harvest_proximity) and P4 (timing_critical) have a
#    stage-dependent gateway, so only those two mappings affect scoring; P1/P2/P5 are
#    included for completeness even though no gateway in those processes reads growth_stage_key.
STAGE_MAP = {
    "P1": {
        "tillering (Zadoks GS 20–29)": "early",
        "stem elongation (Zadoks GS 30–37)": "mid",
        "heading (Zadoks GS 50–59)": "late",
        "grain fill (Zadoks GS 70–77)": "late",
        "ripening (Zadoks GS 83–87)": "late",
    },
    "P2": {
        "vegetative (4–6 true leaves)": "early",
        "flowering": "mid",
        "early fruit set": "mid",
        "mid fruit development": "mid",
        "mature green stage": "late",
    },
    "P3": {
        "pit hardening (July)": "early",
        "early fruit development": "early",
        "mid maturation": "mid",
        "pre-harvest (August)": "late",
        "late harvest period": "late",
    },
    "P4": {
        # production's "mid" = anthesis/flowering, the gateway's own condition text
        # ("Is wheat at anthesis right now"): all three anthesis stages map to it.
        "early anthesis (Zadoks GS 60–61)": "mid",
        "mid anthesis (Zadoks GS 64–65)": "mid",
        "late anthesis (Zadoks GS 67–69)": "mid",
        "early grain fill (Zadoks GS 71–73)": "late",
        "mid grain fill (Zadoks GS 75–77)": "late",
    },
    "P5": {
        "vegetative flush": "early",
        "post-bloom fruit set": "mid",
        "early fruit development": "mid",
        "mid fruit development": "mid",
        "late season": "late",
    },
}

# P1: Wheat Brown Rust
P1_LOCS = ["Foggia, IT", "Bologna, IT", "Ankara, TR", "Thessaloniki, GR", "Seville, ES"]
P1_STAGES = ["tillering (Zadoks GS 20–29)", "stem elongation (Zadoks GS 30–37)",
             "heading (Zadoks GS 50–59)", "grain fill (Zadoks GS 70–77)", "ripening (Zadoks GS 83–87)"]
P1_SEVERITIES = [
    ("low",    "<15%",   "very few isolated lesions affecting less than 15% of the canopy"),
    ("medium", "30–40%", "moderate spread with approximately 30–40% of leaf area affected"),
    ("high",   "50–60%", "heavy infection with 50–60% of the canopy showing orange-yellow uredinia"),
    ("critical","60%+",  "severe outbreak with over 60% of leaves covered in confluent pustules"),
]

def p1_desc(loc, stage, sev_label, sev_pct, sev_desc, weather_phrase):
    return (
        f"Wheat crop at {stage}. Field inspection in {loc.split(',')[0]} reveals {sev_desc}. "
        f"Orange-yellow uredinia with chlorotic halos are visible on the adaxial leaf surface; "
        f"smaller darker telial pustules are present on the underside, consistent with "
        f"Puccinia triticina (brown rust). Severity approximately {sev_pct} of leaf area. "
        f"{weather_phrase}"
    )

# P2: Tomato Late Blight
P2_LOCS = ["Naples, IT", "Catania, IT", "Valencia, ES", "Izmir, TR", "Athens, GR"]
P2_STAGES = ["vegetative (4–6 true leaves)", "flowering", "early fruit set",
             "mid fruit development", "mature green stage"]
P2_SEVERITIES = [
    ("early",    "few lesions", "scattered water-soaked lesions on 2–3 leaves per plant"),
    ("moderate", "expanding",  "water-soaked lesions on 20–30% of the canopy with white sporulation on leaf undersides"),
    ("severe",   "epidemic",   "rapidly expanding brown-black lesions on leaves and stems, white sporulation abundant, fruit infected"),
    ("critical", "systemic",   "whole-plant collapse on affected rows, brown-black stem cankers, advanced sporulation — full epidemic risk"),
]

def p2_desc(loc, stage, sev_label, sev_desc, weather_phrase):
    return (
        f"Tomato crop at {stage} in {loc.split(',')[0]}. Inspection reveals {sev_desc}. "
        f"Brown-black necrotic lesions are confirmed on stems and petioles consistent with "
        f"Phytophthora infestans (late blight). Diagnosis supported by field assessment. "
        f"{weather_phrase}"
    )

# P3: Olive Fruit Fly
P3_LOCS = ["Bari, IT", "Athens, GR", "Seville, ES", "Tunis, TN", "Nicosia, CY"]
P3_STAGES = ["pit hardening (July)", "early fruit development", "mid maturation",
             "pre-harvest (August)", "late harvest period"]
P3_INFESTATIONS = [
    ("threshold", "5–8%",  "5–8% oviposition sting rate on sampled fruits, near economic threshold"),
    ("moderate",  "12%",   "12% oviposition stings confirmed on sampled fruits, above economic threshold; larval galleries visible"),
    ("high",      "20%",   "20% of sampled fruits show oviposition stings; larval galleries confirmed in 15% of fruits"),
    ("severe",    "30%+",  "30% or more oviposition stings detected; extensive larval galleries and early exit holes confirm active infestation"),
]

def p3_desc(loc, stage, inf_label, inf_pct, inf_desc, weather_phrase):
    return (
        f"Olive grove ({stage}) in {loc.split(',')[0]}. Fruit sampling confirms Bactrocera oleae infestation: "
        f"{inf_desc}. Identity confirmed by characteristic oviposition punctures and sub-cuticular larval tunnels. "
        f"Deltamethrin application is being considered but efficacy is temperature- and wind-dependent. "
        f"{weather_phrase}"
    )

# P4: Wheat Fusarium Head Blight
P4_LOCS = ["Bologna, IT", "Poznan, PL", "Plovdiv, BG", "Odessa, UA", "Krasnodar, RU"]
P4_STAGES = ["early anthesis (Zadoks GS 60–61)", "mid anthesis (Zadoks GS 64–65)",
             "late anthesis (Zadoks GS 67–69)", "early grain fill (Zadoks GS 71–73)",
             "mid grain fill (Zadoks GS 75–77)"]
P4_SEVERITIES = [
    ("low",      "5–10%",  "bleaching on 5–10% of spikelets; faint pink-orange sporodochia at spikelet bases on a few heads"),
    ("moderate", "20–30%", "20–30% of wheat heads show bleached spikelets with pink-orange sporodochia at the base, consistent with Fusarium head blight"),
    ("high",     "40–50%", "40–50% of sampled heads affected; heavy sporodochia confirmed; some grain shrivelling visible"),
    ("severe",   "60%+",   "over 60% of heads severely bleached with abundant sporodochia and visible mycelium; significant yield loss expected"),
]

def p4_desc(loc, stage, sev_label, sev_pct, sev_desc, weather_phrase):
    return (
        f"Wheat field at {stage} in {loc.split(',')[0]}. {sev_desc}. "
        f"Fusarium graminearum (head blight / scab) is the suspected pathogen based on "
        f"the characteristic bleaching pattern and spore coloration. "
        f"Fungicide application decision requires assessment of infection-period conditions. "
        f"{weather_phrase}"
    )

# P5: Citrus Greening HLB
P5_LOCS = ["Palermo, IT", "Valencia, ES", "Limassol, CY", "Sfax, TN", "Agadir, MA"]
P5_STAGES = ["vegetative flush", "post-bloom fruit set", "early fruit development",
             "mid fruit development", "late season"]
P5_SEVERITIES = [
    ("suspect",   "1–2 trees",   "1–2 trees with asymmetric blotchy mottling not following vein patterns; lopsided small fruits"),
    ("confirmed", "5–10 trees",  "5–10 trees positive for HLB based on field diagnostic kit; lopsided asymmetric fruits and blotchy mottle on new flush"),
    ("spreading", "10–20%",      "10–20% of grove trees symptomatic; yellowing of new shoots; field kit positive; psyllid adults observed on flush"),
    ("epidemic",  "30%+",        "over 30% of trees symptomatic; psyllid population well-established on new flush; systemic spread confirmed"),
]

def p5_desc(loc, stage, sev_label, sev_desc, weather_phrase):
    return (
        f"Citrus grove ({stage}) in {loc.split(',')[0]}. Inspection confirms: {sev_desc}. "
        f"Candidatus Liberibacter asiaticus (HLB / citrus greening) is confirmed by field diagnostic kit. "
        f"Diaphorina citri psyllid vector management is a priority; spray efficacy depends on current temperature and wind. "
        f"{weather_phrase}"
    )


def _gt_entry(iid: str, process_id: str, tf_severity: str, stage_str: str) -> dict:
    """Build one GT entry from build_dataset.PROCESSES, mapping the tool-forcing
    4-level severity / 5-string stage onto production's 3-level severity / early-mid-late
    stage key. weather_fav is passed as None: weather-dependent gateways are left out
    here and filled in later by derive_toolforcing_gateway_gt.py from the real,
    per-backbone logged API observation.
    """
    proc = PROCESSES[process_id]
    severity = SEVERITY_MAP[process_id][tf_severity]
    growth_stage_key = STAGE_MAP[process_id][stage_str]
    protocol = proc["treatment_protocols"][severity]
    dose = protocol["dose_l_per_ha"]
    gateway_decisions = build_gateway_decisions(proc, severity, None, growth_stage_key)
    return {
        "instance_id": iid,
        "process_id": process_id,
        "expected": {
            "diagnosis_keywords": proc["diagnosis_keywords"],
            "treatment_substances": protocol["substances"],
            # same ±20% tolerance window as production (build_dataset.py build_ground_truth)
            "treatment_dose_per_ha": {
                "min": round(dose * 0.8, 3),
                "max": round(dose * 1.2, 3),
                "unit": protocol["unit"],
            },
            "required_tool_calls": ["get_weather"],
            "gateway_decisions": gateway_decisions,
            "should_escalate": False,
            # internal bookkeeping for derive_toolforcing_gateway_gt.py; not read by evaluate.py
            "_severity_mapped": severity,
            "_growth_stage_key": growth_stage_key,
        },
    }


def generate():
    instances = []
    gt_entries = []

    # P1
    idx = 0
    for li, loc in enumerate(P1_LOCS):
        for vi, (sev_label, sev_pct, sev_desc) in enumerate(P1_SEVERITIES):
            iid = f"P1_TF_{idx+1:03d}"
            stage = P1_STAGES[(li * 4 + vi) % len(P1_STAGES)]
            phrase = WEATHER_PHRASES[(li + vi) % len(WEATHER_PHRASES)]
            desc = p1_desc(loc, stage, sev_label, sev_pct, sev_desc, phrase)
            instances.append({
                "instance_id": iid, "process_id": "P1",
                "request": {"description": desc, "crop": "wheat",
                            "location": loc, "growth_stage": stage, "severity": sev_label}
            })
            gt_entries.append(_gt_entry(iid, "P1", sev_label, stage))
            idx += 1

    # P2
    idx = 0
    for li, loc in enumerate(P2_LOCS):
        for vi, (sev_label, sev_abbr, sev_desc) in enumerate(P2_SEVERITIES):
            iid = f"P2_TF_{idx+1:03d}"
            stage = P2_STAGES[(li * 4 + vi) % len(P2_STAGES)]
            phrase = WEATHER_PHRASES[(li + vi + 1) % len(WEATHER_PHRASES)]
            desc = p2_desc(loc, stage, sev_label, sev_desc, phrase)
            instances.append({
                "instance_id": iid, "process_id": "P2",
                "request": {"description": desc, "crop": "tomato",
                            "location": loc, "growth_stage": stage, "severity": sev_label}
            })
            gt_entries.append(_gt_entry(iid, "P2", sev_label, stage))
            idx += 1

    # P3
    idx = 0
    for li, loc in enumerate(P3_LOCS):
        for vi, (inf_label, inf_pct, inf_desc) in enumerate(P3_INFESTATIONS):
            iid = f"P3_TF_{idx+1:03d}"
            stage = P3_STAGES[(li * 4 + vi) % len(P3_STAGES)]
            phrase = WEATHER_PHRASES[(li + vi + 2) % len(WEATHER_PHRASES)]
            desc = p3_desc(loc, stage, inf_label, inf_pct, inf_desc, phrase)
            instances.append({
                "instance_id": iid, "process_id": "P3",
                "request": {"description": desc, "crop": "olive",
                            "location": loc, "growth_stage": stage, "severity": inf_label}
            })
            gt_entries.append(_gt_entry(iid, "P3", inf_label, stage))
            idx += 1

    # P4
    idx = 0
    for li, loc in enumerate(P4_LOCS):
        for vi, (sev_label, sev_pct, sev_desc) in enumerate(P4_SEVERITIES):
            iid = f"P4_TF_{idx+1:03d}"
            stage = P4_STAGES[(li * 4 + vi) % len(P4_STAGES)]
            phrase = WEATHER_PHRASES[(li + vi + 3) % len(WEATHER_PHRASES)]
            desc = p4_desc(loc, stage, sev_label, sev_pct, sev_desc, phrase)
            instances.append({
                "instance_id": iid, "process_id": "P4",
                "request": {"description": desc, "crop": "wheat",
                            "location": loc, "growth_stage": stage, "severity": sev_label}
            })
            gt_entries.append(_gt_entry(iid, "P4", sev_label, stage))
            idx += 1

    # P5
    idx = 0
    for li, loc in enumerate(P5_LOCS):
        for vi, (sev_label, sev_abbr, sev_desc) in enumerate(P5_SEVERITIES):
            iid = f"P5_TF_{idx+1:03d}"
            stage = P5_STAGES[(li * 4 + vi) % len(P5_STAGES)]
            phrase = WEATHER_PHRASES[(li + vi) % len(WEATHER_PHRASES)]
            desc = p5_desc(loc, stage, sev_label, sev_desc, phrase)
            instances.append({
                "instance_id": iid, "process_id": "P5",
                "request": {"description": desc, "crop": "citrus",
                            "location": loc, "growth_stage": stage, "severity": sev_label}
            })
            gt_entries.append(_gt_entry(iid, "P5", sev_label, stage))
            idx += 1

    # Write instances. NOTE: deliberately NOT newline="" here (unlike build_dataset.py) :
    # the existing reference file (and the dataset_sha256 recorded in
    # results/2026-06-14_toolforcing/*/manifest.json) was generated with plain text-mode
    # "w" on Windows, i.e. with CRLF line endings; matching that convention exactly is
    # what keeps the hash reproducible here, unlike instances.jsonl which was generated
    # inside a Linux container (LF).
    out_inst = ROOT / "datasets" / "tool_forcing_instances_n100.jsonl"
    with open(out_inst, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")
    print(f"Written {len(instances)} instances -> {out_inst}")

    # Write GT
    out_gt = ROOT / "datasets" / "ground_truth_toolforcing_n100.jsonl"
    with open(out_gt, "w", newline="", encoding="utf-8") as f:
        for gt in gt_entries:
            f.write(json.dumps(gt) + "\n")
    print(f"Written {len(gt_entries)} GT entries -> {out_gt}")


if __name__ == "__main__":
    generate()
