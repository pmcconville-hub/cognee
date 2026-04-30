"""
Cognee Load Tests using LocustIO

Usage examples:
    # Note: ENV variables must be set before usage.

    # Multi-dataset test through the web UI
    locust -f locustfile.py --host http://your-cognee-instance.com MultiDatasetCogneeTest

    # Single-dataset test through the web UI
    locust -f locustfile.py --host http://your-cognee-instance.com SingleDatasetCogneeTest

    # Both simultaneously
    locust -f locustfile.py --host http://your-cognee-instance.com

    # Headles without UI, 10 Locust users, 5 minutes
    locust -f locustfile.py --host http://your-cognee-instance.com \
    --headless -u 10 -r 2 --run-time 5m --csv results

Environment variables:
    COGNEE_API_KEY          - API key for authentication (required)
"""

import io
import os
import random
import uuid

from pathlib import Path
from datetime import datetime
from locust import HttpUser, SequentialTaskSet, between, events, tag, task

API_KEY = os.environ.get("COGNEE_API_KEY", "")
SEARCH_TYPE = os.environ.get("COGNEE_SEARCH_TYPE", "GRAPH_COMPLETION")

TOPICS = [
    "quantum computing",
    "machine learning",
    "climate change",
    "renewable energy",
    "space exploration",
    "genetic engineering",
    "blockchain technology",
    "artificial intelligence",
    "ocean conservation",
    "urban planning",
    "medieval history",
    "philosophy of mind",
    "distributed systems",
    "neuroscience",
    "economic theory",
]

SUBTOPICS = [
    "data analysis",
    "pattern recognition",
    "resource allocation",
    "risk assessment",
    "optimization algorithms",
    "predictive modeling",
    "system integration",
    "scalability",
    "error correction",
    "signal processing",
    "network topology",
    "feedback loops",
    "energy efficiency",
    "material science",
    "behavioral adaptation",
    "information theory",
]

SENTENCE_TEMPLATES = [
    "The field of {topic} has seen remarkable advances in recent years, particularly in the area of {subtopic}.",
    "Researchers studying {topic} have discovered that {subtopic} plays a crucial role in understanding the broader implications.",
    "A comprehensive review of {topic} literature reveals that {subtopic} remains one of the most debated aspects.",
    "Recent experiments in {topic} demonstrate a strong correlation between {subtopic} and observed outcomes.",
    "The intersection of {topic} and {subtopic} opens new possibilities for practical applications.",
    "Experts in {topic} argue that {subtopic} will be the defining challenge of the next decade.",
    "Historical analysis shows that {topic} has always been influenced by developments in {subtopic}.",
    "New computational models for {topic} suggest that {subtopic} can be optimized through iterative approaches.",
    "The economic impact of {topic} is closely tied to advancements in {subtopic}, according to recent studies.",
    "Collaborative efforts in {topic} have led to breakthroughs in {subtopic} that were previously thought impossible.",
    "Understanding {topic} requires a deep appreciation of how {subtopic} interacts with existing frameworks.",
    "Policy makers are increasingly turning to {topic} research to inform decisions about {subtopic}.",
]

SEARCH_QUERIES = [
    "What are the main findings about {topic}?",
    "How does {topic} relate to recent developments?",
    "What are the key challenges in {topic}?",
    "Summarize the information about {topic}.",
    "What practical applications exist for {topic}?",
    "What is the current state of research in {topic}?",
]


def generate_paragraph(topic: str, num_sentences: int = 5) -> str:
    sentences = []
    for _ in range(num_sentences):
        template = random.choice(SENTENCE_TEMPLATES)
        subtopic = random.choice(SUBTOPICS)
        sentences.append(template.format(topic=topic, subtopic=subtopic))
    return " ".join(sentences)


def generate_document(num_paragraphs: int = 3) -> tuple[str, str]:
    """Returns (text, primary_topic) for a randomly generated document. Along with random UUID to ensure uniqueness across test runs."""
    topic = random.choice(TOPICS)
    paragraphs = [
        generate_paragraph(topic, num_sentences=random.randint(3, 7)) for _ in range(num_paragraphs)
    ]
    paragraphs.append(str(uuid.uuid4()))  # Ensure uniqueness of document text across test runs
    return "\n\n".join(paragraphs), topic


def generate_search_query(topic: str) -> str:
    template = random.choice(SEARCH_QUERIES)
    return template.format(topic=topic)


@events.init_command_line_parser.add_listener
def add_custom_arguments(parser):
    parser.add_argument(
        "--cognee-api-key",
        type=str,
        default="",
        help="API key for Cognee (overrides COGNEE_API_KEY env var)",
    )


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    api_key = environment.parsed_options.cognee_api_key or API_KEY
    if not api_key:
        environment.runner.quit()
        raise SystemExit(
            "No API key provided. Set COGNEE_API_KEY env var or pass --cognee-api-key."
        )


class AddCognifySearchFlow(SequentialTaskSet):
    """
    Sequential flow: Add text → Cognify → Search.
    Subclasses set `dataset_name` and `api_key`.
    """

    dataset_name: str = ""
    api_key: str = ""
    topic: str = ""

    def _headers(self) -> dict:
        return {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }

    @tag("add")
    @task
    def add_text(self):
        # 1. Generate the text content
        text, self.topic = generate_document(num_paragraphs=random.randint(2, 5))

        # 2. Prepare the form data
        form_data = {
            "datasetName": self.dataset_name,
        }

        # 3. Prepare the file data
        files = [("data", ("document.txt", io.BytesIO(text.encode("utf-8")), "text/plain"))]

        # 4. Make the request
        with self.client.post(
            "/api/v1/add",
            data=form_data,
            files=files,
            headers=self._headers(),
            name="/api/v1/add",
            catch_response=True,
            timeout=60,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                # Enhanced error logging to capture the ErrorResponse model from FastAPI
                resp.failure(f"Add failed: {resp.status_code} - {resp.text[:300]}")

    @tag("cognify")
    @task
    def cognify(self):
        payload = {
            "datasets": [self.dataset_name],
            "runInBackground": False,
        }
        with self.client.post(
            "/api/v1/cognify",
            json=payload,
            headers=self._headers(),
            name="/api/v1/cognify",
            catch_response=True,
            timeout=300,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Cognify failed: {resp.status_code} - {resp.text[:300]}")

    @tag("search")
    @task
    def search(self):
        query = generate_search_query(self.topic or random.choice(TOPICS))
        payload = {
            "searchType": SEARCH_TYPE,
            "query": query,
            "datasets": [self.dataset_name],
        }
        with self.client.post(
            "/api/search",
            json=payload,
            headers=self._headers(),
            name="/api/v1/search",
            catch_response=True,
            timeout=120,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Search failed: {resp.status_code} - {resp.text[:300]}")
        self.interrupt()


# ---------------------------------------------------------------------------
# Scenario 1: Multiple datasets
# ---------------------------------------------------------------------------


class MultiDatasetFlow(AddCognifySearchFlow):
    """Each call a unique dataset."""

    def on_start(self):
        uid = uuid.uuid4().hex[:8]
        self.dataset_name = f"loadtest_user_{uid}"
        self.api_key = self.user.environment.parsed_options.cognee_api_key or API_KEY


class MultiDatasetCogneeTest(HttpUser):
    """
    Scenario 1 – Different virtual users (Locust users) making their own Add → Cognify → Search
    on different datasets.

    Each spawned user creates a unique dataset name and runs the full
    pipeline independently. They use the same API key, simulating multiple users (locust users) from the same account creating
    separate datasets and running operations on them.
    """

    tasks = [MultiDatasetFlow]
    wait_time = between(5, 10)
    weight = 2


# ---------------------------------------------------------------------------
# Scenario 2: Single user, single shared dataset
# ---------------------------------------------------------------------------

SHARED_DATASET_NAME = f"loadtest_shared_{uuid.uuid4().hex[:8]}"


class SingleDatasetFlow(AddCognifySearchFlow):
    """All virtual users share the same dataset and API key."""

    def on_start(self):
        self.dataset_name = SHARED_DATASET_NAME
        self.api_key = self.user.environment.parsed_options.cognee_api_key or API_KEY


class SingleDatasetCogneeTest(HttpUser):
    """
    Scenario 2. calling Add → Cognify → Search on a
    single dataset. All virtual users share one dataset name and API key,
    simulating concurrent calls from the same account.
    """

    tasks = [SingleDatasetFlow]
    wait_time = between(5, 10)
    weight = 1


if __name__ == "__main__":
    import asyncio
    import subprocess
    import sys
    import time
    import urllib.error
    import urllib.request

    import cognee
    from cognee.modules.users.api_key.create_api_key import create_api_key
    from cognee.modules.users.methods import create_default_user

    async def bootstrap() -> str:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        from cognee.modules.engine.operations.setup import setup

        await setup()

        user = await create_default_user()
        api_key_obj = await create_api_key(user, name="locust-loadtest")
        return api_key_obj.api_key

    def wait_for_server(url: str, timeout: float = 240.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError):
                pass
            time.sleep(0.5)
        raise SystemExit(f"Cognee server at {url} did not become ready in {timeout}s")

    api_key = asyncio.run(bootstrap())

    host = os.environ.get("HTTP_API_HOST", "localhost")
    port = os.environ.get("HTTP_API_PORT", "8000")
    base_url = f"http://{host}:{port}"

    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "cognee.api.client:app", "--host", host, "--port", port]
    )
    try:
        wait_for_server(f"{base_url}/health")
        env = {**os.environ, "COGNEE_API_KEY": api_key}
        # Timestamped results to avoid overwriting previous runs and for easier identification of test runs.
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        # Results will be saved in the `results` directory with a unique name based on the timestamp of the test run.
        result_folder = Path("results")
        result_folder.mkdir(exist_ok=True)
        result_location = result_folder / f"locust_run_{timestamp}"
        html_result_location = f"{result_location}.html"
        locust_log_location = Path(f"{result_location}.log")
        locust_log_location.touch()

        # Run locust command
        cmd = [
            "locust",
            "-f",
            __file__,
            "--host",
            base_url,
            "--csv",
            result_location,
            "--html",
            html_result_location,
            "--logfile",
            locust_log_location,
            "--run-time",
            "5m",
            *sys.argv[1:],
        ]

        rc = subprocess.run(cmd, env=env).returncode
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
    sys.exit(rc)
