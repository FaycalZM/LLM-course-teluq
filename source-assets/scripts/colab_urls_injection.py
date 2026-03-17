"""
inject_colab_urls.py
─────────────────────────────────────────────────────────────────────────────
Fetches .ipynb notebook URLs from Google Drive and patches them into the
# COLAB_NOTEBOOKS: [] placeholder that generate_front_matter.py left in
each raw-content week*.md file.

Run this script AFTER generate_front_matter.py.

Setup
─────
  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

  Google Drive credentials:
    1. Go to https://console.cloud.google.com/
    2. Create a project -> enable "Google Drive API"
    3. Create OAuth 2.0 credentials (Desktop app) -> download as credentials.json
    4. Place credentials.json next to this script.
    On first run, a browser window opens to authorise once.
    A token.json is then saved so all future runs are silent.

  DRIVE_ROOT_FOLDER_ID:
    Open your root Drive folder in the browser. The URL looks like:
      https://drive.google.com/drive/folders/1ABCdef1234567890XYZ
    Copy the last segment (1ABCdef1234567890XYZ) and paste it below.

    Your Drive structure must follow this pattern:
      <root folder>/
        week-1/   notebook-a.ipynb  notebook-b.ipynb ...
        week-2/   ...
        ...
        week-14/  ...
    Folder names are matched case-insensitively and accept any separator
    between "week" and the number (week1, week-1, week_1 all work).

Result (replaces placeholder in each week*.md)
───────────────────────────────────────────────
  # COLAB_NOTEBOOKS:
  #   - url: "https://colab.research.google.com/drive/FILE_ID"
  #     label: "Week 1 Intro Notebook"
  #   - url: "https://colab.research.google.com/drive/FILE_ID_2"
  #     label: "Week 1 Exercises"
"""

import json
import re
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

RAW_CONTENT_DIR = Path("../source-assets/raw-content")

# Paste your Google Drive root folder ID here
DRIVE_ROOT_FOLDER_ID = "YOUR_ROOT_FOLDER_ID_HERE"

CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE       = Path("token.json")
SCOPES           = ["https://www.googleapis.com/auth/drive.readonly"]

# ─────────────────────────────────────────────────────────────────────────────


# ── Google Drive helpers ──────────────────────────────────────────────────────

def get_drive_service():
    """Authenticate once, cache token, return Drive v3 service."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                print(
                    "\nERROR: credentials.json not found.\n"
                    "Download it from Google Cloud Console -> APIs & Services -> Credentials\n"
                    "(OAuth 2.0, Desktop app) and place it next to this script."
                )
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


def fetch_all_pages(service, **list_kwargs) -> list:
    """Calls files().list() and follows nextPageToken to get all results."""
    results = []
    page_token = None
    while True:
        if page_token:
            list_kwargs["pageToken"] = page_token
        response = service.files().list(**list_kwargs).execute()
        results.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return results


def fetch_colab_urls(root_folder_id: str) -> dict:
    """
    Returns { week_number: [ {"url": ..., "label": ...}, ... ] }
    by scanning week-N subfolders inside root_folder_id for .ipynb files.
    """
    print("  Authenticating with Google Drive ...")
    service = get_drive_service()
    print("  Authenticated.")

    # List all immediate subfolders
    subfolders = fetch_all_pages(
        service,
        q=(
            f"'{root_folder_id}' in parents"
            " and mimeType='application/vnd.google-apps.folder'"
            " and trashed=false"
        ),
        fields="files(id, name)",
        pageSize=100,
    )

    if not subfolders:
        print(
            "\n  ERROR: No subfolders found in the Drive root folder.\n"
            "  Double-check DRIVE_ROOT_FOLDER_ID."
        )
        sys.exit(1)

    week_folder_re = re.compile(r"^week[-_]?(\d+)$", re.IGNORECASE)
    week_notebooks: dict = {}

    for folder in sorted(subfolders, key=lambda f: f["name"]):
        m = week_folder_re.match(folder["name"].strip())
        if not m:
            continue  # ignore non-week subfolders silently

        week_num = int(m.group(1))

        notebooks = fetch_all_pages(
            service,
            q=(
                f"'{folder['id']}' in parents"
                " and name contains '.ipynb'"
                " and trashed=false"
            ),
            fields="files(id, name)",
            pageSize=100,
        )

        week_notebooks[week_num] = [
            {
                "url":   f"https://colab.research.google.com/drive/{nb['id']}",
                "label": (
                    nb["name"]
                    .replace(".ipynb", "")
                    .replace("_", " ")
                    .replace("-", " ")
                    .strip()
                ),
            }
            for nb in sorted(notebooks, key=lambda x: x["name"])
        ]

        count = len(notebooks)
        print(f"  Week {week_num:02d}: {count} notebook(s) found  ({folder['name']}/)")

    return week_notebooks


# ── Front matter patching ─────────────────────────────────────────────────────

def make_colab_block(notebooks: list) -> str:
    """
    Renders the COLAB_NOTEBOOKS block.

    Single notebook  ->  # COLAB_NOTEBOOKS:
                         #   - url: "..."
                         #     label: "..."

    No notebooks     ->  # COLAB_NOTEBOOKS: []
    """
    if not notebooks:
        return "# COLAB_NOTEBOOKS: []"

    lines = ["# COLAB_NOTEBOOKS:"]
    for nb in notebooks:
        lines.append(f'#   - url: "{nb["url"]}"')
        lines.append(f'#     label: "{nb["label"]}"')
    return "\n".join(lines)


def patch_colab_in_file(md_file: Path, notebooks: list) -> bool:
    """
    Replaces the '# COLAB_NOTEBOOKS: []' placeholder with the full block.
    Returns True if the file was updated, False if skipped.
    """
    content = md_file.read_text(encoding="utf-8")

    # Match the current placeholder (either empty list or already-patched block)
    # We replace only the single-line placeholder so re-running is safe.
    placeholder_re = re.compile(r"^# COLAB_NOTEBOOKS: \[\]$", re.MULTILINE)

    if not placeholder_re.search(content):
        # Already patched or front matter not yet generated
        if "# COLAB_NOTEBOOKS:" in content:
            return False  # already has entries — skip silently
        return False      # generate_front_matter.py hasn't run yet

    new_block = make_colab_block(notebooks)
    updated   = placeholder_re.sub(new_block, content, count=1)
    md_file.write_text(updated, encoding="utf-8")
    return True


def patch_all_files(raw_content_dir: Path, week_notebooks: dict) -> None:
    week_file_re = re.compile(r"week(\d+)", re.IGNORECASE)

    for md_file in sorted(raw_content_dir.glob("week*.md")):
        m = week_file_re.search(md_file.stem)
        if not m:
            continue
        week      = int(m.group(1))
        notebooks = week_notebooks.get(week, [])
        updated   = patch_colab_in_file(md_file, notebooks)

        if updated:
            print(f"  OK    Week {week:02d}: {len(notebooks)} notebook(s) injected -> {md_file.name}")
        elif not notebooks:
            print(f"  Warn  Week {week:02d}: no notebooks found on Drive for this week.")
        else:
            print(f"  Skip  Week {week:02d}: placeholder not found (already patched or front matter missing).")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" inject_colab_urls.py")
    print("=" * 60)

    if DRIVE_ROOT_FOLDER_ID == "YOUR_ROOT_FOLDER_ID_HERE":
        print(
            "\nERROR: DRIVE_ROOT_FOLDER_ID is not set.\n"
            "Open this script and paste your Google Drive folder ID.\n"
            "Find it in the URL when you open the folder:\n"
            "  https://drive.google.com/drive/folders/PASTE_THIS_PART"
        )
        sys.exit(1)

    print("\nStep 1 -- Fetching notebook URLs from Google Drive ...")
    week_notebooks = fetch_colab_urls(DRIVE_ROOT_FOLDER_ID)
    total_nb = sum(len(v) for v in week_notebooks.values())
    print(f"\n  {total_nb} notebooks found across {len(week_notebooks)} week folder(s).")

    print("\nStep 2 -- Patching COLAB_NOTEBOOKS into week files ...")
    patch_all_files(RAW_CONTENT_DIR, week_notebooks)

    print("\nDone.")