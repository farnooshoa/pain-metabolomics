#!/usr/bin/env python3
"""Generate interactive HTML visualization for Pain HeteroGNN Top-250 Differed-Metabolite model.
   Includes ALL 250 selected metabolites, ALL enzymes linked to them, ALL sub-pathways.
"""

import json
import csv
import os
import sys

# Paths
BASE_TOP250 = "/home/ryu11/PPI_GNN/pain_hetero_pipeline/out_hetero_top250"
BASE_IDENT = "/home/ryu11/PPI_GNN/pain_hetero_pipeline/out_hetero_endogenous_identified"
EXCEL_PATH = "/home/ryu11/PPI_GNN/metabolites names and pathways.xlsx"
OUTPUT_HTML = "/home/ryu11/PPI_GNN/attachments_for_collaborator_2026-04-20/interactive_graph_top250.html"

# ---------- Read data ----------

# metabolite_rank.csv
metabolites = []
with open(os.path.join(BASE_TOP250, "metabolite_rank.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        metabolites.append({
            "name": row["metabolite"],
            "importance_score": float(row["importance_score"]),
            "importance_std": float(row["importance_std"]),
            "selection_frequency": int(row["selection_frequency"]),
        })
metabolites.sort(key=lambda x: x["importance_score"], reverse=True)

# Build a lookup: metabolite name -> rank (1-based) and data
met_rank_lookup = {}
for i, m in enumerate(metabolites):
    met_rank_lookup[m["name"]] = {"rank": i + 1, **m}

# enzyme_rank.csv (top-250 schema: enzyme_id, enzyme_name, enzyme_score, score_std, degree,
# score_MEAN_importance, score_MAX_importance, supporting_metabolites, selection_frequency)
enzymes = []
with open(os.path.join(BASE_TOP250, "enzyme_rank.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        enzymes.append({
            "enzyme_id": row["enzyme_id"],
            "enzyme_name": row.get("enzyme_name", "") or "",
            "enzyme_score": float(row["enzyme_score"]),
            "score_std": float(row["score_std"]),
            "degree": int(row["degree"]),
            "score_MEAN_importance": float(row["score_MEAN_importance"]),
            "score_MAX_importance": float(row["score_MAX_importance"]),
            "supporting_metabolites": row["supporting_metabolites"],
            "selection_frequency": int(row["selection_frequency"]),
        })
enzymes.sort(key=lambda x: x["enzyme_score"], reverse=True)

# cv_metrics_overall.csv (no cv_metrics_by_group.csv for top-250 model)
cv_overall = []
with open(os.path.join(BASE_TOP250, "cv_metrics_overall.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        cv_overall.append(row)

# selected_metabolites.csv (p_value, q_value, log2_fc, median_high, median_low)
selected_metabolites = {}
with open(os.path.join(BASE_TOP250, "selected_metabolites.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["metabolite"]
        def _pf(v):
            if v is None or v == "":
                return None
            try:
                return float(v)
            except ValueError:
                return None
        selected_metabolites[name] = {
            "p_value": _pf(row.get("p_value")),
            "q_value": _pf(row.get("q_value")),
            "log2_fc": _pf(row.get("log2_fc")),
            "median_high": _pf(row.get("median_high")),
            "median_low": _pf(row.get("median_low")),
        }

# met2ec.json
with open(os.path.join(BASE_TOP250, "met2ec.json")) as f:
    met2ec = json.load(f)

# met2pathway.json
with open(os.path.join(BASE_TOP250, "met2pathway.json")) as f:
    met2pathway = json.load(f)

# ec_info.json (from identified model - KEGG enzyme info)
with open(os.path.join(BASE_IDENT, "ec_info.json")) as f:
    ec_info = json.load(f)

# rxn_equations.json (from identified model)
with open(os.path.join(BASE_IDENT, "rxn_equations.json")) as f:
    rxn_equations = json.load(f)

# Excel: Chemical Annotation
import openpyxl
wb = openpyxl.load_workbook(EXCEL_PATH, read_only=True)
ws = wb["Chemical Annotation"]
rows = list(ws.iter_rows(values_only=True))
header = list(rows[0])
chem_name_idx = header.index("CHEMICAL_NAME")
super_pw_idx = header.index("SUPER_PATHWAY")
sub_pw_idx = header.index("SUB_PATHWAY")

chem_annotation = {}
for row in rows[1:]:
    name = row[chem_name_idx]
    if name:
        chem_annotation[name] = {
            "super_pathway": row[super_pw_idx] or "Unknown",
            "sub_pathway": row[sub_pw_idx] or "Unknown",
        }
wb.close()

# ---------- Use ALL 250 metabolites and ALL enzymes linked to them ----------
all_metabolites = metabolites  # ALL 250
all_enzymes = enzymes          # All enzymes in the filtered model

all_met_names = set(m["name"] for m in all_metabolites)
all_enz_ids = set(e["enzyme_id"] for e in all_enzymes)

print(f"  All metabolites: {len(all_metabolites)}")
print(f"  All enzymes: {len(all_enzymes)}")

# Annotate metabolites with chemical annotation + Wilcoxon stats
for m in all_metabolites:
    ann = chem_annotation.get(m["name"], {"super_pathway": "Unknown", "sub_pathway": "Unknown"})
    m["super_pathway"] = ann["super_pathway"]
    m["sub_pathway"] = ann["sub_pathway"]
    stats = selected_metabolites.get(m["name"], {})
    m["p_value"] = stats.get("p_value")
    m["q_value"] = stats.get("q_value")
    m["log2_fc"] = stats.get("log2_fc")
    m["median_high"] = stats.get("median_high")
    m["median_low"] = stats.get("median_low")

# Annotate enzymes with ec_info (KEGG name, systematic name, reactions)
EXCLUDED_PATHWAYS = {"path:map01100"}
for e in all_enzymes:
    info = ec_info.get(e["enzyme_id"], {})
    # Prefer our enzyme_name column from top-250 CSV if set, else KEGG name
    kegg_name = info.get("name", "")
    e["name"] = e["enzyme_name"] or kegg_name or e["enzyme_id"]
    e["sysname"] = info.get("sysname", "")
    raw_rxns = info.get("reactions", [])
    rxn_ids = []
    for r in raw_rxns:
        for token in r.replace(">", " ").split():
            token = token.strip()
            if token.startswith("R") and len(token) == 6 and token[1:].isdigit():
                rxn_ids.append(token)
    e["reaction_ids"] = rxn_ids[:10]
    e["reaction_equations"] = {}
    for rid in e["reaction_ids"]:
        if rid in rxn_equations:
            e["reaction_equations"][rid] = rxn_equations[rid]
    # Compute enzyme's connected pathways (excluding map01100) via supporting metabolites
    supp_mets = [s.strip() for s in e["supporting_metabolites"].split(";") if s.strip()]
    pw_set = []
    seen = set()
    for sm in supp_mets:
        for pw in met2pathway.get(sm, []):
            if pw in EXCLUDED_PATHWAYS:
                continue
            if pw not in seen:
                seen.add(pw)
                pw_set.append(pw)
    e["pathways"] = "; ".join(pw_set)

# Color scheme
SUPER_PATHWAY_COLORS = {
    "Lipid": "#f59e0b",
    "Amino Acid": "#10b981",
    "Nucleotide": "#8b5cf6",
    "Peptide": "#ec4899",
    "Cofactors and Vitamins": "#06b6d4",
    "Carbohydrate": "#f97316",
    "Energy": "#ef4444",
    "Partially Characterized Molecules": "#6b7280",
    "Xenobiotics": "#a78bfa",
    "Unknown": "#94a3b8",
}

def get_color(sp):
    return SUPER_PATHWAY_COLORS.get(sp, "#94a3b8")

# ---------- Build Cytoscape elements ----------
nodes = []
edges = []
sub_pathway_set = {}  # sub_pathway -> super_pathway

# Metabolite nodes - ALL 250 metabolites with tiered sizing
n_met = len(all_metabolites)
for i, m in enumerate(all_metabolites):
    rank = i + 1
    sp = m["super_pathway"]
    sub = m["sub_pathway"]
    if sub and sub != "Unknown":
        sub_pathway_set[sub] = sp
    score = m["importance_score"]

    # Tiered sizing and opacity (scaled to 250 metabolites)
    if rank <= 50:
        # Top 50: larger (20-45px), full color
        min_s = all_metabolites[min(49, n_met-1)]["importance_score"]
        max_s = all_metabolites[0]["importance_score"]
        norm = (score - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        size = 20 + norm * 25  # 20-45px
        opacity = 1.0
        tier = "top50"
    elif rank <= 150:
        # Rank 51-150: medium (14-20px), full color
        min_s = all_metabolites[min(149, n_met-1)]["importance_score"]
        max_s = all_metabolites[50]["importance_score"]
        norm = (score - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        size = 14 + norm * 6  # 14-20px
        opacity = 1.0
        tier = "mid"
    else:
        # Rank 151+: small (8-12px), lighter fill (60% opacity)
        min_s = all_metabolites[-1]["importance_score"]
        max_s = all_metabolites[min(150, n_met-1)]["importance_score"]
        norm = (score - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        size = 8 + norm * 4  # 8-12px
        opacity = 0.6
        tier = "low"

    nodes.append({
        "data": {
            "id": "met_" + m["name"],
            "label": m["name"],
            "type": "metabolite",
            "tier": tier,
            "super_pathway": sp,
            "sub_pathway": sub,
            "importance_score": score,
            "importance_std": m["importance_std"],
            "selection_frequency": m["selection_frequency"],
            "p_value": m["p_value"],
            "q_value": m["q_value"],
            "log2_fc": m["log2_fc"],
            "median_high": m["median_high"],
            "median_low": m["median_low"],
            "rank": rank,
            "color": get_color(sp),
            "size": round(size, 1),
            "bg_opacity": opacity,
        }
    })

# Enzyme nodes - ALL enzymes with tiered sizing
n_enz = len(all_enzymes)
for i, e in enumerate(all_enzymes):
    rank = i + 1
    score = e["enzyme_score"]

    if rank <= 30:
        # Top 30: larger (25-45px), bright orange
        min_s = all_enzymes[min(29, n_enz-1)]["enzyme_score"]
        max_s = all_enzymes[0]["enzyme_score"]
        norm = (score - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        size = 25 + norm * 20  # 25-45px
        color = "#fb923c"
        tier = "top30"
    elif rank <= 100:
        # Rank 31-100: medium (15-25px)
        min_s = all_enzymes[min(99, n_enz-1)]["enzyme_score"]
        max_s = all_enzymes[30]["enzyme_score"]
        norm = (score - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        size = 15 + norm * 10  # 15-25px
        color = "#fb923c"
        tier = "mid"
    else:
        # Rank 101+: small (10-15px), lighter fill
        min_s = all_enzymes[-1]["enzyme_score"]
        max_s = all_enzymes[min(100, n_enz-1)]["enzyme_score"]
        norm = (score - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        size = 10 + norm * 5  # 10-15px
        color = "#fb923c"
        tier = "low"

    nodes.append({
        "data": {
            "id": "enz_" + e["enzyme_id"],
            "label": e["name"] if e["name"] != e["enzyme_id"] else e["enzyme_id"],
            "ec_id": e["enzyme_id"],
            "type": "enzyme",
            "tier": tier,
            "enzyme_score": e["enzyme_score"],
            "score_std": e["score_std"],
            "degree": e["degree"],
            "score_MEAN_importance": e["score_MEAN_importance"],
            "score_MAX_importance": e["score_MAX_importance"],
            "supporting_metabolites": e["supporting_metabolites"],
            "pathways": e["pathways"],
            "selection_frequency": e["selection_frequency"],
            "name": e["name"],
            "sysname": e["sysname"],
            "reaction_ids": e["reaction_ids"],
            "reaction_equations": e["reaction_equations"],
            "rank": rank,
            "color": color,
            "size": round(size, 1),
        }
    })

# Sub-pathway met counts (for ALL metabolites)
sub_pw_met_count = {}
for m in all_metabolites:
    sub = m["sub_pathway"]
    if sub and sub != "Unknown" and sub in sub_pathway_set:
        sub_pw_met_count[sub] = sub_pw_met_count.get(sub, 0) + 1

# Sub Pathway nodes - sized by metabolite count (15-50px)
for sub, sp in sub_pathway_set.items():
    count = sub_pw_met_count.get(sub, 1)
    max_count = max(sub_pw_met_count.values()) if sub_pw_met_count else 1
    norm = (count - 1) / (max_count - 1) if max_count > 1 else 0.5
    size = 15 + norm * 35  # 15-50px
    nodes.append({
        "data": {
            "id": "pw_" + sub,
            "label": sub,
            "type": "sub_pathway",
            "super_pathway": sp,
            "color": get_color(sp),
            "size": round(size, 1),
            "met_count": count,
        }
    })

# Edges: metabolite -> enzyme (for ALL metabolites to ALL enzymes)
edge_id = 0
edge_set = set()  # track (source, target) to avoid duplicates

for mname in all_met_names:
    ec_list = met2ec.get(mname, [])
    for ec in ec_list:
        if ec in all_enz_ids:
            key = ("met_" + mname, "enz_" + ec)
            if key not in edge_set:
                edge_set.add(key)
                edges.append({
                    "data": {
                        "id": f"e{edge_id}",
                        "source": key[0],
                        "target": key[1],
                        "type": "met_enz",
                    }
                })
                edge_id += 1

# Also add edges from enzyme supporting_metabolites directly
for e in all_enzymes:
    supp_mets = [s.strip() for s in e["supporting_metabolites"].split(";") if s.strip()]
    for sm in supp_mets:
        if sm in all_met_names:
            key = ("met_" + sm, "enz_" + e["enzyme_id"])
            if key not in edge_set:
                edge_set.add(key)
                edges.append({
                    "data": {
                        "id": f"e{edge_id}",
                        "source": key[0],
                        "target": key[1],
                        "type": "met_enz",
                    }
                })
                edge_id += 1

# Edges: metabolite -> sub_pathway (for ALL metabolites)
for m in all_metabolites:
    sub = m["sub_pathway"]
    if sub and sub != "Unknown" and sub in sub_pathway_set:
        edges.append({
            "data": {
                "id": f"e{edge_id}",
                "source": "met_" + m["name"],
                "target": "pw_" + sub,
                "type": "met_pw",
            }
        })
        edge_id += 1

# Edges: enzyme -> sub_pathway (via supporting metabolites)
for e in all_enzymes:
    supp_mets = [s.strip() for s in e["supporting_metabolites"].split(";")]
    connected_subs = set()
    for sm in supp_mets:
        ann = chem_annotation.get(sm.strip(), {})
        sub = ann.get("sub_pathway")
        if sub and sub in sub_pathway_set:
            connected_subs.add(sub)
    for sub in connected_subs:
        edges.append({
            "data": {
                "id": f"e{edge_id}",
                "source": "enz_" + e["enzyme_id"],
                "target": "pw_" + sub,
                "type": "enz_pw",
            }
        })
        edge_id += 1

elements = nodes + edges

# ---------- Prepare data for JS ----------
# CV metrics
mean_row = None
std_row = None
fold_rows = []
for row in cv_overall:
    if row["fold"] == "mean":
        mean_row = row
    elif row["fold"] == "std":
        std_row = row
    else:
        fold_rows.append(row)

# Metabolite list for sidebar (top 50 only)
met_list_js = []
for i, m in enumerate(all_metabolites[:50]):
    met_list_js.append({
        "name": m["name"],
        "score": m["importance_score"],
        "rank": i + 1,
        "super_pathway": m["super_pathway"],
    })

# Enzyme list for sidebar (top 50 only)
enz_list_js = []
for i, e in enumerate(all_enzymes[:50]):
    enz_list_js.append({
        "ec_id": e["enzyme_id"],
        "name": e["name"],
        "score": e["enzyme_score"],
        "rank": i + 1,
    })

# Pathway list for sidebar
pw_list_js = []
for sub, sp in sorted(sub_pathway_set.items()):
    count = sub_pw_met_count.get(sub, 0)
    pw_list_js.append({
        "name": sub,
        "super_pathway": sp,
        "count": count,
    })
pw_list_js.sort(key=lambda x: x["count"], reverse=True)

# Collect unique super pathways for legend
sp_legend = {}
for m in all_metabolites:
    sp = m["super_pathway"]
    if sp not in sp_legend:
        sp_legend[sp] = get_color(sp)
# Add enzyme
sp_legend["Enzyme"] = "#fb923c"
# Add sub_pathway marker
sp_legend["Sub Pathway"] = "#64748b"

# ---------- Generate HTML ----------

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pain HeteroGNN (Top-250 Differed Model) Network Explorer</title>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background:#0f172a; color:#e2e8f0; display:flex; height:100vh; overflow:hidden; }}

#sidebar {{ width:380px; min-width:380px; background:#1e293b; display:flex; flex-direction:column; overflow:hidden; border-right:1px solid #334155; }}
#sidebar-header {{ padding:16px 20px; border-bottom:1px solid #334155; }}
#sidebar-header h1 {{ font-size:17px; color:#f8fafc; margin-bottom:4px; line-height:1.25; }}
#sidebar-header p {{ font-size:12px; color:#94a3b8; }}
#sidebar-header .variant-note {{ margin-top:6px; padding:6px 8px; background:#1e3a5f; border-left:3px solid #38bdf8; border-radius:4px; font-size:11px; color:#cbd5e1; line-height:1.35; }}

.metrics-section {{ padding:12px 20px; border-bottom:1px solid #334155; }}
.metrics-section h3 {{ font-size:13px; color:#94a3b8; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.5px; }}
.metrics-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }}
.metric-box {{ background:#0f172a; border-radius:8px; padding:8px; text-align:center; border:1px solid #334155; }}
.metric-box .val {{ font-size:18px; font-weight:700; color:#38bdf8; }}
.metric-box .lbl {{ font-size:10px; color:#94a3b8; margin-top:2px; }}

#search-box {{ padding:8px 20px; border-bottom:1px solid #334155; }}
#search-box input {{ width:100%; padding:8px 12px; border-radius:6px; border:1px solid #475569; background:#0f172a; color:#e2e8f0; font-size:13px; outline:none; }}
#search-box input:focus {{ border-color:#38bdf8; }}

#legend {{ padding:8px 20px; border-bottom:1px solid #334155; }}
#legend h3 {{ font-size:12px; color:#94a3b8; margin-bottom:6px; }}
.legend-items {{ display:flex; flex-wrap:wrap; gap:4px 10px; }}
.legend-item {{ display:flex; align-items:center; gap:5px; font-size:11px; cursor:pointer; padding:2px 4px; border-radius:4px; }}
.legend-item:hover {{ background:#334155; }}
.legend-item.dimmed {{ opacity:0.3; }}
.legend-dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}
.legend-dot.diamond {{ border-radius:2px; transform:rotate(45deg); width:9px; height:9px; }}
.legend-dot.hexagon {{ border-radius:2px; }}

.edge-legend {{ margin-top:8px; padding-top:6px; border-top:1px solid #334155; }}
.edge-legend h4 {{ font-size:11px; color:#64748b; margin-bottom:4px; }}
.edge-legend-item {{ display:flex; align-items:center; gap:8px; font-size:11px; color:#94a3b8; margin-bottom:3px; }}
.edge-legend-line {{ width:30px; height:0; flex-shrink:0; }}

.size-legend {{ margin-top:8px; padding-top:6px; border-top:1px solid #334155; }}
.size-legend h4 {{ font-size:11px; color:#64748b; margin-bottom:6px; }}
.size-legend-row {{ display:flex; align-items:center; gap:6px; margin-bottom:4px; font-size:10px; color:#94a3b8; }}
.size-legend-circle {{ border-radius:50%; background:#475569; flex-shrink:0; }}

.tabs {{ display:flex; border-bottom:1px solid #334155; }}
.tab-btn {{ flex:1; padding:8px; text-align:center; font-size:12px; cursor:pointer; border:none; background:transparent; color:#94a3b8; }}
.tab-btn.active {{ color:#38bdf8; border-bottom:2px solid #38bdf8; }}

#tab-content {{ flex:1; overflow-y:auto; }}
.tab-pane {{ display:none; padding:8px 12px; }}
.tab-pane.active {{ display:block; }}

.list-item {{ padding:6px 8px; border-radius:6px; cursor:pointer; display:flex; justify-content:space-between; align-items:center; font-size:12px; border-bottom:1px solid #1e293b; }}
.list-item:hover {{ background:#334155; }}
.list-item .rank {{ color:#64748b; min-width:24px; }}
.list-item .name {{ flex:1; margin:0 6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.list-item .score {{ color:#38bdf8; font-weight:600; font-size:11px; white-space:nowrap; }}

.fold-table {{ width:100%; border-collapse:collapse; font-size:11px; }}
.fold-table th {{ background:#0f172a; padding:6px 4px; text-align:right; color:#94a3b8; position:sticky; top:0; }}
.fold-table td {{ padding:5px 4px; text-align:right; border-bottom:1px solid #1e293b; }}
.fold-table tr.mean-row {{ background:#1e3a5f; font-weight:700; }}

#main {{ flex:1; position:relative; }}
#cy {{ width:100%; height:100%; }}

#detail-panel {{ position:absolute; top:0; right:0; width:340px; height:100%; background:#1e293b; border-left:1px solid #334155; overflow-y:auto; padding:16px; display:none; z-index:10; }}
#detail-panel.open {{ display:block; }}
#detail-close {{ position:absolute; top:8px; right:12px; background:none; border:none; color:#94a3b8; font-size:20px; cursor:pointer; }}
#detail-panel h2 {{ font-size:16px; color:#f8fafc; margin-bottom:12px; padding-right:24px; }}
.detail-row {{ margin-bottom:8px; }}
.detail-row .dlbl {{ font-size:11px; color:#64748b; text-transform:uppercase; }}
.detail-row .dval {{ font-size:13px; color:#e2e8f0; margin-top:2px; word-wrap:break-word; }}
.detail-list {{ list-style:none; padding:0; }}
.detail-list li {{ font-size:12px; padding:2px 0; color:#cbd5e1; }}
.detail-badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:600; margin-bottom:6px; }}
.detail-badge.top-importance {{ background:#1e3a5f; color:#38bdf8; }}
.detail-badge.enzyme-support {{ background:#3b2f1e; color:#f59e0b; }}
.detail-section-title {{ font-size:11px; color:#fbbf24; text-transform:uppercase; letter-spacing:0.5px; margin:10px 0 4px 0; border-top:1px solid #334155; padding-top:8px; }}

#toolbar {{ position:absolute; bottom:16px; left:50%; transform:translateX(-50%); display:flex; gap:6px; z-index:5; }}
.tb-btn {{ padding:6px 14px; border-radius:6px; border:1px solid #475569; background:#1e293b; color:#e2e8f0; font-size:12px; cursor:pointer; }}
.tb-btn:hover {{ background:#334155; }}
.tb-btn.active {{ background:#38bdf8; color:#0f172a; border-color:#38bdf8; }}

#tooltip {{ position:absolute; background:#0f172a; border:1px solid #475569; padding:6px 10px; border-radius:6px; font-size:12px; pointer-events:none; display:none; z-index:20; color:#e2e8f0; }}
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h1>Pain HeteroGNN (Top-250 Differed Model) &mdash; Network Explorer</h1>
    <p>Interactive metabolite-enzyme-pathway network from 6-fold CV</p>
    <div class="variant-note">Top-250 most differentially-abundant metabolites (by Mann-Whitney U p-value) version</div>
  </div>

  <div class="metrics-section">
    <h3>Model Performance (6-Fold CV Mean)</h3>
    <div class="metrics-grid">
      <div class="metric-box"><div class="val">{float(mean_row['auc']):.3f}</div><div class="lbl">AUC</div></div>
      <div class="metric-box"><div class="val">{float(mean_row['f1']):.3f}</div><div class="lbl">F1</div></div>
      <div class="metric-box"><div class="val">{float(mean_row['accuracy']):.3f}</div><div class="lbl">Accuracy</div></div>
      <div class="metric-box"><div class="val">{float(mean_row['precision']):.3f}</div><div class="lbl">Precision</div></div>
      <div class="metric-box"><div class="val">{float(mean_row['recall']):.3f}</div><div class="lbl">Recall</div></div>
    </div>
  </div>

  <div id="search-box">
    <input type="text" id="search-input" placeholder="Search nodes..." />
  </div>

  <div id="legend">
    <h3>Legend (click to toggle)</h3>
    <div class="legend-items" id="legend-items"></div>

    <div class="edge-legend">
      <h4>Edge Types</h4>
      <div class="edge-legend-item">
        <svg class="edge-legend-line" viewBox="0 0 30 4" xmlns="http://www.w3.org/2000/svg">
          <line x1="0" y1="2" x2="30" y2="2" stroke="#38bdf8" stroke-width="2"/>
        </svg>
        <span>Metabolite - Enzyme</span>
      </div>
      <div class="edge-legend-item">
        <svg class="edge-legend-line" viewBox="0 0 30 4" xmlns="http://www.w3.org/2000/svg">
          <line x1="0" y1="2" x2="30" y2="2" stroke="#475569" stroke-width="1" stroke-dasharray="3,2"/>
        </svg>
        <span>Metabolite - Sub-pathway</span>
      </div>
      <div class="edge-legend-item">
        <svg class="edge-legend-line" viewBox="0 0 30 4" xmlns="http://www.w3.org/2000/svg">
          <line x1="0" y1="2" x2="30" y2="2" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="1,3"/>
        </svg>
        <span>Enzyme - Sub-pathway</span>
      </div>
    </div>

    <div class="size-legend">
      <h4>Node Size</h4>
      <div class="size-legend-row">
        <div class="size-legend-circle" style="width:10px;height:10px;"></div>
        <div class="size-legend-circle" style="width:18px;height:18px;"></div>
        <div class="size-legend-circle" style="width:28px;height:28px;"></div>
        <span style="margin-left:4px;">Metabolites: importance score</span>
      </div>
      <div class="size-legend-row">
        <div class="size-legend-circle" style="width:10px;height:10px;background:#fb923c;border-radius:2px;transform:rotate(45deg);"></div>
        <div class="size-legend-circle" style="width:18px;height:18px;background:#fb923c;border-radius:2px;transform:rotate(45deg);"></div>
        <div class="size-legend-circle" style="width:28px;height:28px;background:#fb923c;border-radius:2px;transform:rotate(45deg);"></div>
        <span style="margin-left:4px;">Enzymes: enzyme score</span>
      </div>
      <div class="size-legend-row">
        <div class="size-legend-circle" style="width:10px;height:10px;background:#64748b;border-radius:2px;"></div>
        <div class="size-legend-circle" style="width:18px;height:18px;background:#64748b;border-radius:2px;"></div>
        <div class="size-legend-circle" style="width:28px;height:28px;background:#64748b;border-radius:2px;"></div>
        <span style="margin-left:4px;">Sub-pathways: connected metabolites</span>
      </div>
    </div>
  </div>

  <div class="tabs">
    <div class="tab-btn active" data-tab="tab-met">Metabolites</div>
    <div class="tab-btn" data-tab="tab-enz">Enzymes</div>
    <div class="tab-btn" data-tab="tab-pw">Pathways</div>
    <div class="tab-btn" data-tab="tab-fold">Fold Results</div>
  </div>

  <div id="tab-content">
    <div class="tab-pane active" id="tab-met"></div>
    <div class="tab-pane" id="tab-enz"></div>
    <div class="tab-pane" id="tab-pw"></div>
    <div class="tab-pane" id="tab-fold"></div>
  </div>
</div>

<div id="main">
  <div id="cy"></div>
  <div id="tooltip"></div>
  <div id="detail-panel">
    <button id="detail-close">&times;</button>
    <div id="detail-content"></div>
  </div>
  <div id="toolbar">
    <button class="tb-btn" id="btn-fit">Fit All</button>
    <button class="tb-btn" id="btn-zin">Zoom +</button>
    <button class="tb-btn" id="btn-zout">Zoom -</button>
    <button class="tb-btn" id="btn-relayout">Re-Layout</button>
    <button class="tb-btn" id="btn-top10">Top 10</button>
    <button class="tb-btn" id="btn-top20">Top 20</button>
    <button class="tb-btn" id="btn-top50">Top 50</button>
    <button class="tb-btn" id="btn-reset">Reset</button>
  </div>
</div>

<script>
const ELEMENTS = """ + json.dumps(elements) + """;
const MET_LIST = """ + json.dumps(met_list_js) + """;
const ENZ_LIST = """ + json.dumps(enz_list_js) + """;
const PW_LIST = """ + json.dumps(pw_list_js) + """;
const FOLD_ROWS = """ + json.dumps(fold_rows) + """;
const MEAN_ROW = """ + json.dumps(mean_row) + """;
const STD_ROW = """ + json.dumps(std_row) + """;
const SP_COLORS = """ + json.dumps(sp_legend) + """;

let cy;
let hiddenTypes = new Set();

document.addEventListener('DOMContentLoaded', function() {
  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: ELEMENTS,
    style: [
      // Metabolite nodes - top50 tier (full opacity, larger)
      { selector: 'node[type="metabolite"][tier="top50"]', style: {
        'background-color': 'data(color)', 'background-opacity': 1.0,
        'label': 'data(label)', 'width': 'data(size)', 'height': 'data(size)',
        'font-size': 8, 'color': '#e2e8f0', 'text-valign': 'bottom', 'text-margin-y': 4,
        'text-max-width': 80, 'text-wrap': 'ellipsis', 'border-width': 1, 'border-color': '#475569',
        'min-zoomed-font-size': 10
      }},
      // Metabolite nodes - mid tier (full opacity, medium)
      { selector: 'node[type="metabolite"][tier="mid"]', style: {
        'background-color': 'data(color)', 'background-opacity': 1.0,
        'label': 'data(label)', 'width': 'data(size)', 'height': 'data(size)',
        'font-size': 7, 'color': '#cbd5e1', 'text-valign': 'bottom', 'text-margin-y': 4,
        'text-max-width': 80, 'text-wrap': 'ellipsis', 'border-width': 1, 'border-color': '#475569',
        'min-zoomed-font-size': 10
      }},
      // Metabolite nodes - low tier (60% opacity, small)
      { selector: 'node[type="metabolite"][tier="low"]', style: {
        'background-color': 'data(color)', 'background-opacity': 0.6,
        'label': 'data(label)', 'width': 'data(size)', 'height': 'data(size)',
        'font-size': 6, 'color': '#94a3b8', 'text-valign': 'bottom', 'text-margin-y': 4,
        'text-max-width': 70, 'text-wrap': 'ellipsis', 'border-width': 0.5, 'border-color': '#334155',
        'min-zoomed-font-size': 10
      }},
      // Enzyme nodes - top30
      { selector: 'node[type="enzyme"][tier="top30"]', style: {
        'background-color': '#fb923c', 'background-opacity': 1.0,
        'label': 'data(label)', 'shape': 'diamond',
        'width': 'data(size)', 'height': 'data(size)',
        'font-size': 7, 'color': '#fed7aa', 'text-valign': 'bottom', 'text-margin-y': 4,
        'text-max-width': 80, 'text-wrap': 'ellipsis', 'border-width': 1, 'border-color': '#92400e',
        'min-zoomed-font-size': 10
      }},
      // Enzyme nodes - mid
      { selector: 'node[type="enzyme"][tier="mid"]', style: {
        'background-color': '#fb923c', 'background-opacity': 0.85,
        'label': 'data(label)', 'shape': 'diamond',
        'width': 'data(size)', 'height': 'data(size)',
        'font-size': 7, 'color': '#fed7aa', 'text-valign': 'bottom', 'text-margin-y': 4,
        'text-max-width': 80, 'text-wrap': 'ellipsis', 'border-width': 1, 'border-color': '#92400e',
        'min-zoomed-font-size': 10
      }},
      // Enzyme nodes - low
      { selector: 'node[type="enzyme"][tier="low"]', style: {
        'background-color': '#fb923c', 'background-opacity': 0.5,
        'label': 'data(label)', 'shape': 'diamond',
        'width': 'data(size)', 'height': 'data(size)',
        'font-size': 6, 'color': '#94a3b8', 'text-valign': 'bottom', 'text-margin-y': 4,
        'text-max-width': 70, 'text-wrap': 'ellipsis', 'border-width': 0.5, 'border-color': '#92400e',
        'min-zoomed-font-size': 10
      }},
      { selector: 'node[type="sub_pathway"]', style: {
        'background-color': 'data(color)', 'label': 'data(label)', 'shape': 'hexagon',
        'width': 'data(size)', 'height': 'data(size)',
        'font-size': 7, 'color': '#cbd5e1', 'text-valign': 'bottom', 'text-margin-y': 4,
        'text-max-width': 90, 'text-wrap': 'ellipsis', 'border-width': 1, 'border-color': '#475569',
        'min-zoomed-font-size': 10
      }},
      { selector: 'edge[type="met_enz"]', style: {
        'line-color': '#38bdf8', 'width': 2, 'opacity': 0.7, 'curve-style': 'bezier', 'line-style': 'solid'
      }},
      { selector: 'edge[type="met_pw"]', style: {
        'line-color': '#475569', 'width': 1, 'opacity': 0.4, 'curve-style': 'bezier', 'line-style': 'dashed',
        'line-dash-pattern': [4, 3]
      }},
      { selector: 'edge[type="enz_pw"]', style: {
        'line-color': '#f59e0b', 'width': 1.5, 'opacity': 0.5, 'curve-style': 'bezier', 'line-style': 'dotted'
      }},
      { selector: '.highlighted', style: {
        'border-width': 3, 'border-color': '#38bdf8', 'z-index': 999
      }},
      { selector: '.dimmed', style: { 'opacity': 0.1 }},
      { selector: '.search-match', style: { 'border-width': 3, 'border-color': '#fbbf24', 'z-index': 999 }},
      { selector: 'edge.highlighted', style: { 'width': 4, 'opacity': 1, 'z-index': 998 }},
      { selector: 'edge.highlighted[type="met_enz"]', style: { 'line-color': '#7dd3fc' }},
      { selector: 'edge.highlighted[type="met_pw"]', style: { 'line-color': '#94a3b8' }},
      { selector: 'edge.highlighted[type="enz_pw"]', style: { 'line-color': '#fbbf24' }},
    ],
    layout: { name: 'cose', animate: false, nodeRepulsion: 8000, idealEdgeLength: 80, gravity: 0.3, numIter: 500 },
    minZoom: 0.05, maxZoom: 5,
    textureOnViewport: true,
    hideEdgesOnViewport: true,
    pixelRatio: 1,
  });

  // Fit to all nodes after layout
  cy.fit();

  // Legend
  const legendEl = document.getElementById('legend-items');
  for (const [name, color] of Object.entries(SP_COLORS)) {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.dataset.sp = name;
    let dotClass = 'legend-dot';
    if (name === 'Enzyme') dotClass += ' diamond';
    else if (name === 'Sub Pathway') dotClass += ' hexagon';
    item.innerHTML = '<div class="' + dotClass + '" style="background:' + color + '"></div><span>' + name + '</span>';
    item.addEventListener('click', function() {
      const sp = this.dataset.sp;
      if (hiddenTypes.has(sp)) {
        hiddenTypes.delete(sp);
        this.classList.remove('dimmed');
      } else {
        hiddenTypes.add(sp);
        this.classList.add('dimmed');
      }
      applyVisibility();
    });
    legendEl.appendChild(item);
  }

  function applyVisibility() {
    cy.batch(function() {
      cy.nodes().forEach(n => {
        let cat;
        const t = n.data('type');
        if (t === 'enzyme') cat = 'Enzyme';
        else if (t === 'sub_pathway') cat = 'Sub Pathway';
        else cat = n.data('super_pathway') || 'Unknown';
        if (hiddenTypes.has(cat)) { n.style('display', 'none'); }
        else { n.style('display', 'element'); }
      });
      cy.edges().forEach(e => {
        const src = e.source(); const tgt = e.target();
        if (src.style('display') === 'none' || tgt.style('display') === 'none') {
          e.style('display', 'none');
        } else {
          e.style('display', 'element');
        }
      });
    });
  }

  // Tabs
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      this.classList.add('active');
      document.getElementById(this.dataset.tab).classList.add('active');
    });
  });

  // Populate metabolites tab (top 50)
  const metPane = document.getElementById('tab-met');
  MET_LIST.forEach(m => {
    const d = document.createElement('div');
    d.className = 'list-item';
    d.innerHTML = '<span class="rank">#' + m.rank + '</span><span class="name" title="' + escHtml(m.name) + '">' + escHtml(m.name) + '</span><span class="score">' + m.score.toExponential(2) + '</span>';
    d.addEventListener('click', () => highlightNode('met_' + m.name));
    metPane.appendChild(d);
  });

  // Populate enzymes tab (top 50)
  const enzPane = document.getElementById('tab-enz');
  ENZ_LIST.forEach(e => {
    const d = document.createElement('div');
    d.className = 'list-item';
    d.innerHTML = '<span class="rank">#' + e.rank + '</span><span class="name" title="' + escHtml(e.name) + '">' + escHtml(e.ec_id) + ' ' + escHtml(e.name) + '</span><span class="score">' + e.score.toExponential(2) + '</span>';
    d.addEventListener('click', () => highlightNode('enz_' + e.ec_id));
    enzPane.appendChild(d);
  });

  // Populate pathways tab
  const pwPane = document.getElementById('tab-pw');
  PW_LIST.forEach(p => {
    const d = document.createElement('div');
    d.className = 'list-item';
    d.innerHTML = '<span class="name" title="' + escHtml(p.name) + '">' + escHtml(p.name) + '</span><span class="score">' + p.count + ' metabolites</span>';
    d.addEventListener('click', () => highlightNode('pw_' + p.name));
    pwPane.appendChild(d);
  });

  // Populate fold results tab
  const foldPane = document.getElementById('tab-fold');
  let thtml = '<table class="fold-table"><thead><tr><th>Fold</th><th>Acc</th><th>AUC</th><th>F1</th><th>Prec</th><th>Rec</th></tr></thead><tbody>';
  FOLD_ROWS.forEach(r => {
    thtml += '<tr><td>' + r.fold + '</td><td>' + parseFloat(r.accuracy).toFixed(3) + '</td><td>' + parseFloat(r.auc).toFixed(3) + '</td><td>' + parseFloat(r.f1).toFixed(3) + '</td><td>' + parseFloat(r.precision).toFixed(3) + '</td><td>' + parseFloat(r.recall).toFixed(3) + '</td></tr>';
  });
  thtml += '<tr class="mean-row"><td>Mean</td><td>' + parseFloat(MEAN_ROW.accuracy).toFixed(3) + '</td><td>' + parseFloat(MEAN_ROW.auc).toFixed(3) + '</td><td>' + parseFloat(MEAN_ROW.f1).toFixed(3) + '</td><td>' + parseFloat(MEAN_ROW.precision).toFixed(3) + '</td><td>' + parseFloat(MEAN_ROW.recall).toFixed(3) + '</td></tr>';
  thtml += '<tr><td>Std</td><td>' + parseFloat(STD_ROW.accuracy).toFixed(3) + '</td><td>' + parseFloat(STD_ROW.auc).toFixed(3) + '</td><td>' + parseFloat(STD_ROW.f1).toFixed(3) + '</td><td>' + parseFloat(STD_ROW.precision).toFixed(3) + '</td><td>' + parseFloat(STD_ROW.recall).toFixed(3) + '</td></tr>';
  thtml += '</tbody></table>';
  foldPane.innerHTML = thtml;

  // Search
  document.getElementById('search-input').addEventListener('input', function() {
    const q = this.value.toLowerCase().trim();
    cy.batch(function() {
      cy.nodes().removeClass('search-match');
      if (!q) return;
      cy.nodes().forEach(n => {
        const lbl = (n.data('label') || '').toLowerCase();
        const eid = (n.data('ec_id') || '').toLowerCase();
        if (lbl.includes(q) || eid.includes(q)) n.addClass('search-match');
      });
    });
  });

  // Tooltip
  const tooltip = document.getElementById('tooltip');
  cy.on('mouseover', 'node', function(evt) {
    const n = evt.target;
    let extra = '';
    const tier = n.data('tier');
    if (tier) extra = '<br><span style="color:#94a3b8;font-size:10px;">tier: ' + tier + '</span>';
    tooltip.innerHTML = '<strong>' + escHtml(n.data('label')) + '</strong><br><span style="color:#94a3b8">' + n.data('type') + '</span>' + extra;
    tooltip.style.display = 'block';
  });
  cy.on('mousemove', 'node', function(evt) {
    const pos = evt.renderedPosition;
    tooltip.style.left = (pos.x + document.getElementById('sidebar').offsetWidth + 12) + 'px';
    tooltip.style.top = (pos.y + 12) + 'px';
  });
  cy.on('mouseout', 'node', function() { tooltip.style.display = 'none'; });

  // Click node -> detail panel
  cy.on('tap', 'node', function(evt) { showDetail(evt.target); });
  cy.on('tap', function(evt) {
    if (evt.target === cy) {
      document.getElementById('detail-panel').classList.remove('open');
      resetHighlight();
    }
  });
  document.getElementById('detail-close').addEventListener('click', function() {
    document.getElementById('detail-panel').classList.remove('open');
    resetHighlight();
  });

  // Toolbar
  document.getElementById('btn-fit').addEventListener('click', () => cy.fit());
  document.getElementById('btn-zin').addEventListener('click', () => cy.zoom(cy.zoom() * 1.3));
  document.getElementById('btn-zout').addEventListener('click', () => cy.zoom(cy.zoom() / 1.3));
  document.getElementById('btn-relayout').addEventListener('click', () => {
    cy.layout({ name: 'cose', animate: false, nodeRepulsion: 8000, idealEdgeLength: 80, gravity: 0.3, numIter: 500 }).run();
    cy.fit();
  });
  document.getElementById('btn-top10').addEventListener('click', () => highlightTopN(10));
  document.getElementById('btn-top20').addEventListener('click', () => highlightTopN(20));
  document.getElementById('btn-top50').addEventListener('click', () => highlightTopN(50));
  document.getElementById('btn-reset').addEventListener('click', () => {
    resetHighlight();
    document.getElementById('search-input').value = '';
    cy.batch(function() {
      cy.nodes().removeClass('search-match');
    });
    hiddenTypes.clear();
    document.querySelectorAll('.legend-item').forEach(i => i.classList.remove('dimmed'));
    cy.batch(function() {
      cy.elements().style('display', 'element');
    });
    cy.fit();
  });
});

function escHtml(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function fmtNum(v, mode) {
  if (v === null || v === undefined || v === '' || (typeof v === 'number' && isNaN(v))) return 'N/A';
  const num = Number(v);
  if (!isFinite(num)) return 'N/A';
  if (mode === 'exp') return num.toExponential(3);
  if (mode === 'fix3') return num.toFixed(3);
  return String(num);
}

function highlightNode(id) {
  resetHighlight();
  const n = cy.getElementById(id);
  if (!n || n.length === 0) return;
  cy.batch(function() {
    cy.elements().addClass('dimmed');
    n.removeClass('dimmed').addClass('highlighted');
    const connected = n.connectedEdges();
    connected.removeClass('dimmed').addClass('highlighted');
    connected.connectedNodes().removeClass('dimmed').addClass('highlighted');
  });
  cy.animate({ center: { eles: n }, zoom: Math.max(cy.zoom(), 1.5) }, { duration: 400 });
  showDetail(n);
}

function resetHighlight() {
  cy.batch(function() {
    cy.elements().removeClass('dimmed highlighted search-match');
  });
}

function highlightTopN(n) {
  resetHighlight();
  const topMetIds = MET_LIST.slice(0, n).map(m => 'met_' + m.name);
  cy.batch(function() {
    cy.elements().addClass('dimmed');
    topMetIds.forEach(id => {
      const node = cy.getElementById(id);
      if (node.length) {
        node.removeClass('dimmed').addClass('highlighted');
        const conn = node.connectedEdges();
        conn.removeClass('dimmed').addClass('highlighted');
        conn.connectedNodes().removeClass('dimmed').addClass('highlighted');
      }
    });
  });
  // Active buttons
  document.querySelectorAll('.tb-btn').forEach(b => b.classList.remove('active'));
  if (n === 10) document.getElementById('btn-top10').classList.add('active');
  if (n === 20) document.getElementById('btn-top20').classList.add('active');
  if (n === 50) document.getElementById('btn-top50').classList.add('active');
}

function showDetail(node) {
  const panel = document.getElementById('detail-panel');
  const content = document.getElementById('detail-content');
  const d = node.data();
  let html = '';

  if (d.type === 'metabolite') {
    html = '<h2>' + escHtml(d.label) + '</h2>';
    html += '<div class="detail-badge top-importance">Rank #' + d.rank + ' Metabolite (tier: ' + d.tier + ')</div>';
    html += detailRow('Overall Importance Rank', '#' + d.rank);
    html += detailRow('Importance Score', fmtNum(d.importance_score, 'exp'));
    html += detailRow('Score Std', fmtNum(d.importance_std, 'exp'));
    html += detailRow('Selection Frequency', d.selection_frequency + '/6');
    html += detailRow('Super Pathway', d.super_pathway);
    html += detailRow('Sub Pathway', d.sub_pathway);
    // Differential-abundance statistics (Mann-Whitney U / Wilcoxon rank-sum test)
    html += '<div class="detail-section-title">Differential Abundance (high-pain vs low-pain)</div>';
    html += detailRow('Wilcoxon p-value', fmtNum(d.p_value, 'exp'));
    html += detailRow('q-value (BH-FDR)', fmtNum(d.q_value, 'exp'));
    html += detailRow('log2 Fold-Change', fmtNum(d.log2_fc, 'fix3'));
    html += detailRow('Median (high-pain)', fmtNum(d.median_high, 'fix3'));
    html += detailRow('Median (low-pain)', fmtNum(d.median_low, 'fix3'));
    // Connected enzymes
    const enzNeighbors = node.neighborhood('node[type="enzyme"]');
    if (enzNeighbors.length) {
      html += '<div class="detail-row"><div class="dlbl">Connected Enzymes (' + enzNeighbors.length + ')</div><ul class="detail-list">';
      enzNeighbors.forEach(en => { html += '<li>' + escHtml(en.data('ec_id')) + ' - ' + escHtml(en.data('label')) + '</li>'; });
      html += '</ul></div>';
    }
    const pwNeighbors = node.neighborhood('node[type="sub_pathway"]');
    if (pwNeighbors.length) {
      html += '<div class="detail-row"><div class="dlbl">Connected Pathways (' + pwNeighbors.length + ')</div><ul class="detail-list">';
      pwNeighbors.forEach(pn => { html += '<li>' + escHtml(pn.data('label')) + '</li>'; });
      html += '</ul></div>';
    }
  } else if (d.type === 'enzyme') {
    html = '<h2>' + escHtml(d.ec_id) + '</h2>';
    html += detailRow('Enzyme Name', d.name);
    if (d.sysname) html += detailRow('Systematic Name', d.sysname);
    html += detailRow('Rank', '#' + d.rank);
    html += detailRow('Degree (# connected metabolites)', d.degree);
    html += detailRow('Enzyme Score (primary)', fmtNum(d.enzyme_score, 'exp'));
    html += detailRow('Score Std', fmtNum(d.score_std, 'exp'));
    html += detailRow('Selection Frequency', d.selection_frequency + '/6');
    // Alternative scores
    html += '<div class="detail-section-title">Alternative Score Formulations</div>';
    html += detailRow('MEAN_importance', fmtNum(d.score_MEAN_importance, 'exp'));
    html += detailRow('MAX_importance', fmtNum(d.score_MAX_importance, 'exp'));
    html += detailRow('Supporting Metabolites', d.supporting_metabolites);
    if (d.pathways) html += detailRow('KEGG Pathways (excl. map01100)', d.pathways);
    if (d.reaction_ids && d.reaction_ids.length) {
      html += '<div class="detail-row"><div class="dlbl">Reactions (' + d.reaction_ids.length + ')</div><ul class="detail-list">';
      d.reaction_ids.forEach(rid => {
        const eq = d.reaction_equations[rid] || '';
        html += '<li><strong>' + escHtml(rid) + '</strong>: ' + escHtml(eq) + '</li>';
      });
      html += '</ul></div>';
    }
    const metNeighbors = node.neighborhood('node[type="metabolite"]');
    if (metNeighbors.length) {
      html += '<div class="detail-row"><div class="dlbl">Connected Metabolites (' + metNeighbors.length + ')</div><ul class="detail-list">';
      metNeighbors.forEach(mn => { html += '<li>' + escHtml(mn.data('label')) + '</li>'; });
      html += '</ul></div>';
    }
  } else if (d.type === 'sub_pathway') {
    html = '<h2>' + escHtml(d.label) + '</h2>';
    html += detailRow('Type', 'Sub Pathway');
    html += detailRow('Super Pathway', d.super_pathway);
    const metNeighbors = node.neighborhood('node[type="metabolite"]');
    if (metNeighbors.length) {
      html += '<div class="detail-row"><div class="dlbl">Metabolites (' + metNeighbors.length + ')</div><ul class="detail-list">';
      metNeighbors.forEach(mn => { html += '<li>' + escHtml(mn.data('label')) + '</li>'; });
      html += '</ul></div>';
    }
    const enzNeighbors = node.neighborhood('node[type="enzyme"]');
    if (enzNeighbors.length) {
      html += '<div class="detail-row"><div class="dlbl">Connected Enzymes (' + enzNeighbors.length + ')</div><ul class="detail-list">';
      enzNeighbors.forEach(en => { html += '<li>' + escHtml(en.data('ec_id')) + ' - ' + escHtml(en.data('label')) + '</li>'; });
      html += '</ul></div>';
    }
  }

  content.innerHTML = html;
  panel.classList.add('open');
}

function detailRow(label, value) {
  return '<div class="detail-row"><div class="dlbl">' + label + '</div><div class="dval">' + escHtml(String(value)) + '</div></div>';
}
</script>
</body>
</html>"""

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"HTML written to {OUTPUT_HTML}")
print(f"  Nodes: {len(nodes)}")
print(f"  Edges: {len(edges)}")
print(f"  File size: {os.path.getsize(OUTPUT_HTML) / 1024:.1f} KB")

# Breakdown
met_nodes = sum(1 for n in nodes if n["data"]["type"] == "metabolite")
enz_nodes = sum(1 for n in nodes if n["data"]["type"] == "enzyme")
pw_nodes = sum(1 for n in nodes if n["data"]["type"] == "sub_pathway")
met_enz_edges = sum(1 for e in edges if e["data"]["type"] == "met_enz")
met_pw_edges = sum(1 for e in edges if e["data"]["type"] == "met_pw")
enz_pw_edges = sum(1 for e in edges if e["data"]["type"] == "enz_pw")
print(f"  Metabolite nodes: {met_nodes}")
print(f"  Enzyme nodes: {enz_nodes}")
print(f"  Sub-pathway nodes: {pw_nodes}")
print(f"  Met-Enz edges: {met_enz_edges}")
print(f"  Met-PW edges: {met_pw_edges}")
print(f"  Enz-PW edges: {enz_pw_edges}")
