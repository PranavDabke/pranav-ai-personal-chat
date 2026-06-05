"""
build_kb.py — Run ONCE locally before deploying.
Fetches Pranav's public GitHub repos + READMEs and writes them into data/
as committed knowledge-base files. The chat app reads only data/*.md at runtime,
so serving has no live GitHub dependency (faster, no rate-limit risk).

Usage:
    python build_kb.py
"""

import os
import requests

GITHUB_USER = "PranavDabke"
OUT_DIR = "data"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Fetching repos for {GITHUB_USER}...")
    repos = requests.get(
        f"https://api.github.com/users/{GITHUB_USER}/repos?per_page=100",
        timeout=15,
    ).json()

    if not isinstance(repos, list):
        print("ERROR: GitHub API did not return a repo list. "
              "You may be rate-limited (60 req/hr unauthenticated). "
              "Wait a few minutes and retry. Response:", repos)
        return

    written = 0
    for repo in repos:
        name = repo.get("name", "")
        desc = repo.get("description") or "No description provided."
        lang = repo.get("language") or "N/A"
        url = repo.get("html_url", "")

        readme = ""
        try:
            r = requests.get(
                f"https://api.github.com/repos/{GITHUB_USER}/{name}/readme",
                headers={"Accept": "application/vnd.github.raw+json"},
                timeout=15,
            )
            if r.status_code == 200:
                readme = r.text
        except Exception as e:
            print(f"  ! README fetch failed for {name}: {e}")

        content = (
            f"# GitHub Repository: {name}\n\n"
            f"- Primary language: {lang}\n"
            f"- URL: {url}\n"
            f"- Description: {desc}\n\n"
        )
        if readme.strip():
            content += "## README\n\n" + readme

        path = os.path.join(OUT_DIR, f"repo_{name}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        written += 1
        print(f"  wrote {path} ({'README included' if readme.strip() else 'description only'})")

    print(f"\nDone. Wrote {written} repo files into {OUT_DIR}/.")
    print("Commit these files so the chatbot stays grounded without a live GitHub call.")


if __name__ == "__main__":
    main()
