"""
OpenRouter LLM-based metabolite name normalization module.

This module uses OpenRouter API with google/gemini-2.5-flash to normalize
metabolite names and generate KEGG query candidates.
"""

import os
import json
import time
import logging
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class OpenRouterAliasGenerator:
    """Generate KEGG query aliases for metabolites using OpenRouter LLM."""

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = "google/gemini-2.5-flash"

    SYSTEM_PROMPT = (
        "You normalize metabolomics feature names for KEGG mapping. "
        "Output strictly valid JSON and nothing else."
    )

    USER_PROMPT_TEMPLATE = """For each metabolite string below, output a JSON list of objects with keys:
  original, description, queries, confidence.
Rules:
- Provide 5-10 query strings to use with KEGG REST: /find/compound/<query>
- Include variants: remove punctuation, remove stereochemistry prefixes (S-/R-), add 'acid' when appropriate,
  expand abbreviations (e.g., DiHOME) into plausible full names, include common synonyms
  (e.g., alpha-ketoglutarate vs 2-oxoglutarate).
- confidence must be one of: high, medium, low
Return JSON only.

Metabolites:
{metabolites}"""

    def __init__(
        self,
        cache_path: str = "aliases_cache.json",
        batch_size: int = 20,
        max_retries: int = 2,
        retry_delay: float = 2.0
    ):
        """
        Initialize the alias generator.

        Args:
            cache_path: Path to cache file for storing LLM results
            batch_size: Number of metabolites per LLM request
            max_retries: Maximum retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Please set it with: export OPENROUTER_API_KEY='your-key-here'"
            )

        self.cache_path = Path(cache_path)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Load cached aliases from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.info(f"Loaded {len(cache)} cached metabolite aliases")
                return cache
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load cache: {e}")
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved cache with {len(self.cache)} entries")
        except IOError as e:
            logger.error(f"Could not save cache: {e}")

    def _call_openrouter(self, metabolites: List[str]) -> Optional[List[Dict]]:
        """
        Call OpenRouter API to normalize metabolite names.

        Args:
            metabolites: List of metabolite names to normalize

        Returns:
            List of alias objects or None on failure
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/pain-gnn-pipeline",
            "X-Title": "Pain GNN Pipeline"
        }

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            metabolites="\n".join(f"- {m}" for m in metabolites)
        )

        payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling OpenRouter (attempt {attempt + 1}/{self.max_retries}) "
                           f"for {len(metabolites)} metabolites...")

                response = requests.post(
                    self.OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()

                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Parse JSON response
                try:
                    # Try to extract JSON from response
                    content = content.strip()
                    if content.startswith("```"):
                        # Remove markdown code blocks
                        lines = content.split("\n")
                        content = "\n".join(
                            l for l in lines
                            if not l.strip().startswith("```")
                        )

                    parsed = json.loads(content)

                    # Handle both list and dict with 'metabolites' key
                    if isinstance(parsed, dict):
                        if "metabolites" in parsed:
                            parsed = parsed["metabolites"]
                        elif "results" in parsed:
                            parsed = parsed["results"]
                        else:
                            # Try to extract list from dict values
                            for v in parsed.values():
                                if isinstance(v, list):
                                    parsed = v
                                    break

                    if isinstance(parsed, list):
                        logger.info(f"Successfully parsed {len(parsed)} metabolite aliases")
                        return parsed
                    else:
                        logger.warning(f"Unexpected response format: {type(parsed)}")

                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error: {e}")
                    logger.debug(f"Raw content: {content[:500]}...")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error: {e}")

            if attempt < self.max_retries - 1:
                logger.info(f"Retrying in {self.retry_delay}s...")
                time.sleep(self.retry_delay)

        return None

    def _generate_fallback_queries(self, metabolite: str) -> Dict:
        """
        Generate fallback queries without LLM.

        Args:
            metabolite: Metabolite name

        Returns:
            Alias object with basic query variants
        """
        queries = []
        name = metabolite.strip()

        # Original name
        queries.append(name.lower())

        # Remove common suffixes/prefixes
        clean = name
        for prefix in ["S-", "R-", "L-", "D-", "alpha-", "beta-", "gamma-",
                       "cis-", "trans-", "N-", "O-"]:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                queries.append(clean.lower())

        # Remove asterisks and numbers in parentheses
        import re
        clean = re.sub(r'\*+$', '', name)
        clean = re.sub(r'\s*\(\d+\)\s*$', '', clean)
        clean = re.sub(r'\s*\[\d+\]\s*$', '', clean)
        if clean != name:
            queries.append(clean.lower())

        # Remove stereochemistry numbers
        clean = re.sub(r'\d+-', '', name)
        if clean != name:
            queries.append(clean.lower())

        # Add 'acid' if might be an acid
        if 'ate' in name.lower() and 'acid' not in name.lower():
            acid_form = re.sub(r'ate\b', 'ic acid', name, flags=re.IGNORECASE)
            queries.append(acid_form.lower())

        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_clean = q.strip()
            if q_clean and q_clean not in seen:
                seen.add(q_clean)
                unique_queries.append(q_clean)

        return {
            "original": metabolite,
            "description": "Fallback (no LLM)",
            "queries": unique_queries[:10],
            "confidence": "low"
        }

    def generate_aliases(
        self,
        metabolites: List[str],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Dict]:
        """
        Generate KEGG query aliases for a list of metabolites.

        Args:
            metabolites: List of metabolite names
            use_llm: Whether to use LLM (if False, only use cache/fallback)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Dict mapping metabolite name to alias object
        """
        results = {}
        uncached = []

        # Check cache first
        for met in metabolites:
            if met in self.cache:
                results[met] = self.cache[met]
            else:
                uncached.append(met)

        logger.info(f"Found {len(results)} metabolites in cache, "
                   f"{len(uncached)} need processing")

        if not uncached:
            return results

        if use_llm:
            # Process in batches
            total_batches = (len(uncached) + self.batch_size - 1) // self.batch_size

            for i in range(0, len(uncached), self.batch_size):
                batch = uncached[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1

                logger.info(f"Processing batch {batch_num}/{total_batches} "
                           f"({len(batch)} metabolites)")

                if progress_callback:
                    progress_callback(i, len(uncached))

                llm_results = self._call_openrouter(batch)

                if llm_results:
                    # Match results to metabolites
                    for alias_obj in llm_results:
                        original = alias_obj.get("original", "")
                        # Find matching metabolite (case-insensitive)
                        matched = None
                        for met in batch:
                            if met.lower() == original.lower() or met == original:
                                matched = met
                                break

                        if matched:
                            results[matched] = alias_obj
                            self.cache[matched] = alias_obj

                # Generate fallback for any missing
                for met in batch:
                    if met not in results:
                        logger.warning(f"No LLM result for '{met}', using fallback")
                        fallback = self._generate_fallback_queries(met)
                        results[met] = fallback
                        self.cache[met] = fallback

                # Save cache after each batch
                self._save_cache()

                # Rate limiting
                if i + self.batch_size < len(uncached):
                    time.sleep(1.0)
        else:
            # Fallback only
            for met in uncached:
                fallback = self._generate_fallback_queries(met)
                results[met] = fallback
                self.cache[met] = fallback
            self._save_cache()

        if progress_callback:
            progress_callback(len(uncached), len(uncached))

        return results


def main():
    """Test the alias generator."""
    import sys

    logging.basicConfig(level=logging.INFO)

    test_metabolites = [
        "glucose",
        "alpha-ketoglutarate",
        "S-adenosylmethionine",
        "12,13-DiHOME",
        "palmitate (16:0)"
    ]

    try:
        generator = OpenRouterAliasGenerator(cache_path="test_aliases_cache.json")
        aliases = generator.generate_aliases(test_metabolites)

        print("\n=== Alias Results ===")
        for met, alias in aliases.items():
            print(f"\n{met}:")
            print(f"  Description: {alias.get('description', 'N/A')}")
            print(f"  Queries: {alias.get('queries', [])[:5]}")
            print(f"  Confidence: {alias.get('confidence', 'N/A')}")

    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
