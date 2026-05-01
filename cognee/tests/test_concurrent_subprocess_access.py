import asyncio
import os
import pathlib
import subprocess

import cognee
from cognee.shared.logging_utils import get_logger


logger = get_logger()


def _shared_kuzu_lock_enabled() -> bool:
    return os.getenv("SHARED_KUZU_LOCK", "").lower() in {"true", "1", "yes", "on"}


def _local_skip_enabled() -> bool:
    return os.getenv("COGNEE_SKIP_SHARED_KUZU_LOCK_TEST", "").lower() in {
        "true",
        "1",
        "yes",
        "on",
    }


def _child_env(data_directory_path: str, cognee_directory_path: str, kuzu_db_path: str) -> dict:
    env = os.environ.copy()
    env["COGNEE_TEST_DATA_ROOT"] = data_directory_path
    env["COGNEE_TEST_SYSTEM_ROOT"] = cognee_directory_path
    env["COGNEE_TEST_KUZU_DB_PATH"] = kuzu_db_path
    return env


def _wait_for_success(*processes: subprocess.Popen) -> None:
    failures = []

    for process in processes:
        returncode = process.wait()
        if returncode != 0:
            failures.append((process.args, returncode))

    if failures:
        args, returncode = failures[0]
        raise subprocess.CalledProcessError(returncode, args)


async def concurrent_subprocess_access():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/concurrent_tasks")
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/concurrent_tasks")
        ).resolve()
    )
    subprocess_directory_path = str(
        pathlib.Path(os.path.join(pathlib.Path(__file__).parent, "subprocesses/")).resolve()
    )
    kuzu_db_path = os.path.join(cognee_directory_path, "databases", "subprocess_access_graph")
    child_env = _child_env(data_directory_path, cognee_directory_path, kuzu_db_path)

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    if not _shared_kuzu_lock_enabled():
        if not _local_skip_enabled():
            raise RuntimeError(
                "Concurrent subprocess access test requires SHARED_KUZU_LOCK=true. "
                "Set COGNEE_SKIP_SHARED_KUZU_LOCK_TEST=true only for explicit local smoke runs."
            )
        logger.info("Skipping concurrent subprocess access test: SHARED_KUZU_LOCK is not enabled")
        return

    writer_process = subprocess.Popen(
        [os.sys.executable, os.path.join(subprocess_directory_path, "writer.py")],
        env=child_env,
    )
    reader_process = subprocess.Popen(
        [os.sys.executable, os.path.join(subprocess_directory_path, "reader.py")],
        env=child_env,
    )

    _wait_for_success(writer_process, reader_process)

    logger.info("Basic write read subprocess example finished")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(
        """
        This is the text of the first cognify subprocess
        """,
        dataset_name="first_cognify_dataset",
    )
    await cognee.add(
        """
        This is the text of the second cognify subprocess
        """,
        dataset_name="second_cognify_dataset",
    )

    first_cognify_process = subprocess.Popen(
        [os.sys.executable, os.path.join(subprocess_directory_path, "simple_cognify_1.py")],
        env=child_env,
    )
    second_cognify_process = subprocess.Popen(
        [os.sys.executable, os.path.join(subprocess_directory_path, "simple_cognify_2.py")],
        env=child_env,
    )

    _wait_for_success(first_cognify_process, second_cognify_process)

    logger.info("Database concurrent subprocess example finished")


if __name__ == "__main__":
    asyncio.run(concurrent_subprocess_access())
