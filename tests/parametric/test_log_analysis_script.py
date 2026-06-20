from pathlib import Path
import subprocess
import zipfile

from torch.utils.tensorboard import SummaryWriter


def test_analyze_training_log_script_summarizes_zip(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "analyze_training_log.py"
    event_dir = tmp_path / "events"
    event_dir.mkdir()
    writer = SummaryWriter(log_dir=event_dir)
    writer.add_scalar("ESR", 0.25, 0)
    writer.add_scalar("ESR", 0.125, 1)
    writer.flush()
    writer.close()

    archive_path = event_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w") as archive:
        for event_path in event_dir.rglob("*"):
            archive.write(event_path, event_path.relative_to(tmp_path))

    result = subprocess.run(
        ["conda", "run", "-n", "nam", "python", str(script_path), str(archive_path)],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert "# events" in result.stdout
    assert "ESR: best epoch_idx=" in result.stdout
