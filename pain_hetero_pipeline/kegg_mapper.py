"""
KEGG REST API mapping module.

This module queries KEGG REST API to map metabolites to compounds,
reactions, enzymes, and pathways.
"""

import os
import json
import time
import logging
import hashlib
import requests
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class KEGGMapper:
    """Map metabolites to KEGG compounds, reactions, enzymes, and pathways."""

    BASE_URL = "https://rest.kegg.jp"
    REQUEST_DELAY = 0.15  # Polite delay between requests (seconds)

    def __init__(
        self,
        cache_dir: str = "kegg_cache",
        request_delay: float = 0.15
    ):
        """
        Initialize the KEGG mapper.

        Args:
            cache_dir: Directory for caching KEGG API responses
            request_delay: Delay between API requests (seconds)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay

        # Mapping results
        self.met2cpd: Dict[str, Optional[str]] = {}
        self.met2rxn: Dict[str, List[str]] = {}
        self.met2pathway: Dict[str, List[str]] = {}
        self.met2ec: Dict[str, List[str]] = {}
        self.mapping_report: List[Dict] = []

        # Compound info cache
        self.cpd_info: Dict[str, Dict] = {}

    def _cache_path(self, url: str) -> Path:
        """Get cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.json"

    def _fetch(self, endpoint: str, use_cache: bool = True) -> Optional[str]:
        """
        Fetch data from KEGG REST API with caching.

        Args:
            endpoint: API endpoint (e.g., "/find/compound/glucose")
            use_cache: Whether to use cached responses

        Returns:
            Response text or None on failure
        """
        url = f"{self.BASE_URL}{endpoint}"
        cache_file = self._cache_path(url)

        # Check cache
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                    # Return cached content (may be None for cached 400 errors)
                    if "content" in cached:
                        return cached["content"]
            except (json.JSONDecodeError, IOError):
                pass

        # Make request
        try:
            time.sleep(self.request_delay)
            logger.debug(f"Fetching: {url}")

            response = requests.get(url, timeout=30)

            if response.status_code == 404:
                # Not found - cache empty result
                content = ""
            elif response.status_code == 200:
                content = response.text
            elif response.status_code == 400:
                # Bad request (usually URL encoding issue) - cache as error
                logger.warning(f"KEGG API error 400 for {endpoint}")
                content = None  # Mark as cached error
            else:
                logger.warning(f"KEGG API error {response.status_code} for {endpoint}")
                return None

            # Cache response (including 400 errors to avoid retry)
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({"url": url, "content": content}, f)
            except IOError as e:
                logger.warning(f"Could not cache response: {e}")

            return content

        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {endpoint}: {e}")
            return None

    def _parse_find_results(self, response: str) -> List[Tuple[str, str]]:
        """
        Parse KEGG /find results.

        Args:
            response: Raw response text

        Returns:
            List of (compound_id, names_string) tuples
        """
        results = []
        if not response:
            return results

        for line in response.strip().split('\n'):
            if '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    cpd_id = parts[0].strip()
                    names = parts[1].strip()
                    results.append((cpd_id, names))

        return results

    def _parse_link_results(self, response: str) -> List[str]:
        """
        Parse KEGG /link results.

        Args:
            response: Raw response text

        Returns:
            List of linked IDs
        """
        results = []
        if not response:
            return results

        for line in response.strip().split('\n'):
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    # Second column is the linked ID
                    linked_id = parts[1].strip()
                    results.append(linked_id)

        return results

    def _score_match(self, query: str, names: str) -> float:
        """
        Score how well a query matches compound names.

        Args:
            query: Search query
            names: Compound names string (semicolon-separated)

        Returns:
            Match score (higher is better)
        """
        query_lower = query.lower().strip()
        names_lower = names.lower()
        name_list = [n.strip() for n in names_lower.split(';')]

        # Exact match with first name
        if name_list and name_list[0] == query_lower:
            return 1.0

        # Exact match with any name
        if query_lower in name_list:
            return 0.9

        # First name contains query
        if name_list and query_lower in name_list[0]:
            return 0.7

        # Any name contains query
        for name in name_list:
            if query_lower in name:
                return 0.5

        # Query contains first name
        if name_list and name_list[0] in query_lower:
            return 0.3

        # Fallback - some overlap
        return 0.1

    def find_compound(
        self,
        metabolite: str,
        queries: List[str]
    ) -> Tuple[Optional[str], str, float]:
        """
        Find KEGG compound for a metabolite using candidate queries.

        Args:
            metabolite: Original metabolite name
            queries: List of search queries to try

        Returns:
            Tuple of (compound_id, evidence_line, confidence_score)
        """
        best_cpd = None
        best_score = 0.0
        best_evidence = ""

        for query in queries:
            # Clean query
            query_clean = query.strip()
            if not query_clean:
                continue

            # URL encode query
            query_encoded = requests.utils.quote(query_clean)
            endpoint = f"/find/compound/{query_encoded}"

            response = self._fetch(endpoint)
            if not response:
                continue

            candidates = self._parse_find_results(response)

            for cpd_id, names in candidates:
                score = self._score_match(query_clean, names)

                if score > best_score:
                    best_score = score
                    best_cpd = cpd_id
                    best_evidence = f"Query '{query_clean}' -> {cpd_id}: {names[:100]}"

                # Perfect match - stop searching
                if score >= 0.9:
                    return best_cpd, best_evidence, best_score

        return best_cpd, best_evidence, best_score

    def get_reactions(self, compound_id: str) -> List[str]:
        """Get reactions linked to a compound."""
        if not compound_id:
            return []

        # Extract just the ID part (e.g., "C00031" from "cpd:C00031")
        cpd_clean = compound_id.replace("cpd:", "")
        endpoint = f"/link/reaction/cpd:{cpd_clean}"

        response = self._fetch(endpoint)
        reactions = self._parse_link_results(response)

        return reactions

    def get_pathways(self, compound_id: str) -> List[str]:
        """Get pathways linked to a compound."""
        if not compound_id:
            return []

        cpd_clean = compound_id.replace("cpd:", "")
        endpoint = f"/link/pathway/cpd:{cpd_clean}"

        response = self._fetch(endpoint)
        pathways = self._parse_link_results(response)

        # Filter to keep only map pathways (remove organism-specific)
        pathways = [p for p in pathways if p.startswith("path:map") or p.startswith("map")]

        return pathways

    def get_enzymes_for_reaction(self, reaction_id: str) -> List[str]:
        """Get enzymes (EC numbers) linked to a reaction."""
        if not reaction_id:
            return []

        rxn_clean = reaction_id.replace("rn:", "")
        endpoint = f"/link/enzyme/rn:{rxn_clean}"

        response = self._fetch(endpoint)
        enzymes = self._parse_link_results(response)

        return enzymes

    def map_metabolite(
        self,
        metabolite: str,
        queries: List[str],
        confidence: str = "medium"
    ) -> Dict:
        """
        Fully map a metabolite to compounds, reactions, enzymes, pathways.

        Args:
            metabolite: Original metabolite name
            queries: List of KEGG queries to try
            confidence: LLM confidence level

        Returns:
            Mapping result dict
        """
        result = {
            "original": metabolite,
            "queries_tried": queries,
            "cpd": None,
            "reactions": [],
            "pathways": [],
            "enzymes": [],
            "evidence": "",
            "confidence": confidence,
            "mapping_score": 0.0
        }

        # Find compound
        cpd_id, evidence, score = self.find_compound(metabolite, queries)

        if cpd_id:
            result["cpd"] = cpd_id
            result["evidence"] = evidence
            result["mapping_score"] = score

            # Get reactions
            reactions = self.get_reactions(cpd_id)
            result["reactions"] = reactions

            # Get pathways
            pathways = self.get_pathways(cpd_id)
            result["pathways"] = pathways

            # Get enzymes from reactions
            enzymes = set()
            for rxn in reactions[:20]:  # Limit to avoid too many requests
                rxn_enzymes = self.get_enzymes_for_reaction(rxn)
                enzymes.update(rxn_enzymes)
            result["enzymes"] = list(enzymes)

        return result

    def map_all_metabolites(
        self,
        alias_results: Dict[str, Dict],
        progress_callback: Optional[callable] = None
    ):
        """
        Map all metabolites using their LLM-generated queries.

        Args:
            alias_results: Dict from OpenRouterAliasGenerator.generate_aliases()
            progress_callback: Optional callback(current, total) for progress
        """
        total = len(alias_results)
        logger.info(f"Mapping {total} metabolites to KEGG...")

        for i, (met, alias) in enumerate(alias_results.items()):
            if progress_callback:
                progress_callback(i, total)

            queries = alias.get("queries", [])
            confidence = alias.get("confidence", "medium")

            if not queries:
                # Generate basic fallback queries
                queries = [met.lower()]

            logger.debug(f"[{i+1}/{total}] Mapping: {met}")

            result = self.map_metabolite(met, queries, confidence)

            # Store results
            self.met2cpd[met] = result["cpd"]
            self.met2rxn[met] = result["reactions"]
            self.met2pathway[met] = result["pathways"]
            self.met2ec[met] = result["enzymes"]
            self.mapping_report.append({
                "original_met_name": met,
                "llm_queries": "; ".join(queries[:5]),
                "chosen_kegg_cpd": result["cpd"],
                "confidence": confidence,
                "mapping_score": result["mapping_score"],
                "evidence_line": result["evidence"],
                "n_reactions": len(result["reactions"]),
                "n_enzymes": len(result["enzymes"]),
                "n_pathways": len(result["pathways"])
            })

        if progress_callback:
            progress_callback(total, total)

        # Log summary
        mapped = sum(1 for v in self.met2cpd.values() if v is not None)
        with_enzymes = sum(1 for v in self.met2ec.values() if v)
        logger.info(f"Mapping complete: {mapped}/{total} compounds found, "
                   f"{with_enzymes} with enzymes")

    def load_kegg_excel(self, excel_path: str) -> Dict[str, Dict]:
        """
        Load supplementary KEGG mappings from Excel file.

        Args:
            excel_path: Path to KEGG Excel file

        Returns:
            Dict mapping metabolite names to KEGG info
        """
        import pandas as pd

        try:
            df = pd.read_excel(excel_path)
            logger.info(f"Loaded KEGG Excel with {len(df)} rows")

            mappings = {}
            for _, row in df.iterrows():
                # Try different column names
                met_name = None
                for col in ['BIOCHEMICAL', 'Metabolite', 'Name', 'metabolite']:
                    if col in df.columns and pd.notna(row.get(col)):
                        met_name = str(row[col]).strip()
                        break

                if not met_name:
                    continue

                kegg_id = None
                for col in ['KEGG', 'KEGG_ID', 'kegg_id', 'KEGG ID']:
                    if col in df.columns and pd.notna(row.get(col)):
                        kegg_id = str(row[col]).strip()
                        break

                pathway = None
                for col in ['SuperPathway', 'SUPER_PATHWAY', 'Pathway', 'pathway']:
                    if col in df.columns and pd.notna(row.get(col)):
                        pathway = str(row[col]).strip()
                        break

                sub_pathway = None
                for col in ['sub_pathway', 'SUB_PATHWAY', 'SubPathway']:
                    if col in df.columns and pd.notna(row.get(col)):
                        sub_pathway = str(row[col]).strip()
                        break

                mappings[met_name] = {
                    "kegg_id": kegg_id,
                    "pathway": pathway,
                    "sub_pathway": sub_pathway
                }

            return mappings

        except Exception as e:
            logger.warning(f"Could not load KEGG Excel: {e}")
            return {}

    def merge_excel_mappings(self, excel_mappings: Dict[str, Dict]):
        """
        Merge Excel mappings into current results.
        Excel mappings are used as supplementary info, not override.

        Args:
            excel_mappings: Dict from load_kegg_excel()
        """
        for met, excel_info in excel_mappings.items():
            if met not in self.met2cpd:
                continue

            # If we don't have KEGG cpd but Excel does, use it
            if not self.met2cpd[met] and excel_info.get("kegg_id"):
                kegg_id = excel_info["kegg_id"]
                if kegg_id.startswith("C"):
                    self.met2cpd[met] = f"cpd:{kegg_id}"
                    logger.debug(f"Added Excel mapping for {met}: {kegg_id}")

            # Add pathway info if available
            if excel_info.get("pathway") and not self.met2pathway.get(met):
                self.met2pathway[met] = [excel_info["pathway"]]

    def save_mappings(self, output_dir: str):
        """
        Save all mapping results to files.

        Args:
            output_dir: Output directory
        """
        import pandas as pd

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON mappings
        with open(output_path / "met2cpd.json", 'w') as f:
            json.dump(self.met2cpd, f, indent=2)

        with open(output_path / "met2rxn.json", 'w') as f:
            json.dump(self.met2rxn, f, indent=2)

        with open(output_path / "met2pathway.json", 'w') as f:
            json.dump(self.met2pathway, f, indent=2)

        with open(output_path / "met2ec.json", 'w') as f:
            json.dump(self.met2ec, f, indent=2)

        # Save mapping report CSV
        if self.mapping_report:
            df = pd.DataFrame(self.mapping_report)
            df.to_csv(output_path / "mapping_report.csv", index=False)

        logger.info(f"Saved mappings to {output_path}")

    def get_all_enzymes(self) -> Set[str]:
        """Get set of all unique enzymes."""
        enzymes = set()
        for ecs in self.met2ec.values():
            enzymes.update(ecs)
        return enzymes

    def get_enzyme_metabolites(self) -> Dict[str, List[str]]:
        """Get mapping of enzymes to their metabolites."""
        ec2met = defaultdict(list)
        for met, ecs in self.met2ec.items():
            for ec in ecs:
                ec2met[ec].append(met)
        return dict(ec2met)


def main():
    """Test the KEGG mapper."""
    logging.basicConfig(level=logging.INFO)

    # Test with a few metabolites
    test_aliases = {
        "glucose": {
            "queries": ["glucose", "D-glucose", "dextrose"],
            "confidence": "high"
        },
        "pyruvate": {
            "queries": ["pyruvate", "pyruvic acid"],
            "confidence": "high"
        },
        "unknown_metabolite_xyz": {
            "queries": ["unknown_metabolite_xyz"],
            "confidence": "low"
        }
    }

    mapper = KEGGMapper(cache_dir="test_kegg_cache")
    mapper.map_all_metabolites(test_aliases)

    print("\n=== Mapping Results ===")
    for met in test_aliases:
        print(f"\n{met}:")
        print(f"  Compound: {mapper.met2cpd.get(met)}")
        print(f"  Reactions: {len(mapper.met2rxn.get(met, []))}")
        print(f"  Enzymes: {mapper.met2ec.get(met, [])[:5]}")
        print(f"  Pathways: {mapper.met2pathway.get(met, [])[:3]}")

    mapper.save_mappings("test_kegg_output")


if __name__ == "__main__":
    main()
