"""The release inventory must detect files that hashes alone cannot catch."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_release_manifest import release_files
from verify_release import check_files
from export_public_release import export_release


def make_release(root):
    source = root / "source.txt"
    source.write_text("reviewed source\n")
    (root / "release_manifest.json").write_text(json.dumps({
        "files": {"source.txt": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
    }))
    return source


def test_inventory_checks_extra_missing_and_changed_files(tmp_path):
    source = make_release(tmp_path)
    check_files(tmp_path)
    extra = tmp_path / "unreviewed.txt"
    extra.write_text("must not silently join a release")
    with pytest.raises(ValueError, match="extra=.*unreviewed.txt"):
        check_files(tmp_path)
    extra.unlink()
    source.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        check_files(tmp_path)
    source.unlink()
    with pytest.raises(ValueError, match="missing=.*source.txt"):
        check_files(tmp_path)


@pytest.mark.parametrize("directory", [".aws", ".agents", ".codex", ".ssh"])
def test_manifest_builder_rejects_private_configuration(tmp_path, directory):
    private = tmp_path / directory
    private.mkdir()
    (private / "config").write_text("private configuration")
    with pytest.raises(ValueError, match="Private configuration"):
        list(release_files(tmp_path))


def test_inventory_excludes_generated_outputs_and_rejects_symlinks(tmp_path):
    source = make_release(tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "log.txt").write_text("local output")
    check_files(tmp_path)
    (tmp_path / "linked.txt").symlink_to(source)
    with pytest.raises(ValueError, match="Symlink"):
        check_files(tmp_path)


@pytest.mark.parametrize("relative", [
    "study_posts.jsonl", "data/private_posts.csv", "results/reference/individual_predictions.json",
])
def test_manifest_builder_rejects_post_level_and_unapproved_research_artifacts(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("private research data")
    with pytest.raises(ValueError, match="Post-level|Unapproved research artifact"):
        list(release_files(tmp_path))


def test_export_excludes_private_outputs_and_copies_verified_files(tmp_path):
    root = tmp_path / "working"
    root.mkdir()
    make_release(root)
    private = root / "outputs" / "private_release_archive"
    private.mkdir(parents=True)
    (private / "posts.jsonl").write_text("private records")
    destination = tmp_path / "public"
    assert export_release(root, destination) == 2
    assert {path.name for path in destination.iterdir()} == {"source.txt", "release_manifest.json"}
    check_files(destination)
    with pytest.raises(ValueError, match="already exist"):
        export_release(root, destination)
    with pytest.raises(ValueError, match="outside"):
        export_release(root, root / "nested")


def test_export_fails_before_copying_unreviewed_files(tmp_path):
    root = tmp_path / "working"
    root.mkdir()
    make_release(root)
    (root / "unreviewed.txt").write_text("must not export")
    destination = tmp_path / "public"
    with pytest.raises(ValueError, match="inventory mismatch"):
        export_release(root, destination)
    assert not destination.exists()
