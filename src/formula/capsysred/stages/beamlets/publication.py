"""Stage 11 result publication, rollback and leftover detection."""

import os
import shutil


STAGING_PREFIX = "stage11-staging-"
TMP_PREFIX = "stage11-tmp-"
BACKUP_SUFFIX = ".prev"


def refuse_stage11_leftovers(out_dir):
    """Leftovers of an earlier run (backups, staging, tmp) stop the stage:
    a complete set of final names does not prove a completed publication,
    so nothing is removed automatically."""
    found = []
    try:
        names = os.listdir(out_dir)
    except FileNotFoundError:
        return
    for name in sorted(names):
        if (name.endswith(BACKUP_SUFFIX) or name.startswith(STAGING_PREFIX)
                or name.startswith(TMP_PREFIX)):
            found.append(name)
    if found:
        raise ValueError(
            f"{out_dir}: leftovers of an earlier stage-11 run: {', '.join(found)}. "
            f"Restore or remove them manually (a complete *{BACKUP_SUFFIX} set is "
            "the previous result; a staging directory holds an unpublished one), "
            "then rerun")


def _file_state(path) -> str:
    try:
        return f"{os.path.getsize(path)} B"
    except OSError:
        return "absent"


def _publication_state(staging, out_dir, names) -> str:
    """One line per file: final / backup / staged sizes, for recovery."""
    lines = []
    for name in names:
        final = os.path.join(out_dir, name)
        lines.append(f"{name}: final {_file_state(final)}, backup "
                     f"{_file_state(final + BACKUP_SUFFIX)}, staged "
                     f"{_file_state(os.path.join(staging, name))}")
    return "; ".join(lines)


def publish_stage11(staging, out_dir, names, replace=os.replace):
    """Install the stage files from staging into out_dir with per-file
    backups: phase 1 moves existing files to *.prev, phase 2 installs the new
    ones. Installing the last file is the boundary: before it any error rolls
    back (new files out, backups restored); after it only the cleanup of the
    backups and staging runs and its failures are reported, never rolled back."""
    backed, installed = [], []
    try:
        for name in names:
            final = os.path.join(out_dir, name)
            if os.path.lexists(final):
                replace(final, final + BACKUP_SUFFIX)
                backed.append(name)
        for name in names:
            replace(os.path.join(staging, name), os.path.join(out_dir, name))
            installed.append(name)
    except BaseException as exc:
        failures = []
        for name in names:
            final = os.path.join(out_dir, name)
            try:
                if name in backed:
                    replace(final + BACKUP_SUFFIX, final)
                elif name in installed:
                    os.remove(final)
            except OSError as err:
                failures.append(f"{name}: {err}")
        if failures:
            raise ValueError(
                f"stage 11 publication failed ({exc}) and the rollback did not "
                f"complete; {staging} and the *{BACKUP_SUFFIX} files are kept for "
                "manual recovery. Rollback errors: " + "; ".join(failures)
                + ". File state: " + _publication_state(staging, out_dir, names)) from exc
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"stage 11 publication failed before completion; the "
                         f"previous result was restored: {exc}") from exc
    leftovers = []
    for name in backed:
        try:
            os.remove(os.path.join(out_dir, name + BACKUP_SUFFIX))
        except OSError as err:
            leftovers.append(f"{name}{BACKUP_SUFFIX}: {err}")
    try:
        shutil.rmtree(staging)
    except OSError as err:
        leftovers.append(f"{staging}: {err}")
    if leftovers:
        raise ValueError("stage 11 published completely, but the cleanup left "
                         "remnants to remove manually: " + "; ".join(leftovers))
