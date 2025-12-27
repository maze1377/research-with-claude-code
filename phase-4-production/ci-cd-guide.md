# CI/CD for AI Agent Systems

**Last Updated:** December 2025
**Difficulty:** Intermediate to Advanced

Deploying AI agents requires specialized CI/CD practices that account for non-deterministic behavior, model dependencies, and evaluation-driven workflows. This guide covers pipeline design, testing strategies, and deployment patterns for production agent systems.

---

## Table of Contents

1. [CI/CD Challenges for Agents](#cicd-challenges-for-agents)
2. [Pipeline Architecture](#pipeline-architecture)
3. [GitHub Actions Workflows](#github-actions-workflows)
4. [Testing in CI](#testing-in-ci)
5. [Deployment Strategies](#deployment-strategies)
6. [Model and Prompt Management](#model-and-prompt-management)
7. [Monitoring and Rollback](#monitoring-and-rollback)

---

## CI/CD Challenges for Agents

### Unique Considerations

| Challenge | Traditional CI/CD | Agent CI/CD |
|-----------|------------------|-------------|
| **Test Determinism** | Reproducible outputs | Non-deterministic LLM responses |
| **Dependencies** | Fixed versions | Model versions, API availability |
| **Evaluation** | Pass/fail tests | Continuous quality metrics |
| **Rollback** | Code revert | Code + model + prompt revert |
| **Cost** | Compute time | API costs, rate limits |
| **Security** | Code secrets | Prompt injection, model safety |

### Key Principles

1. **Evaluation-Driven Deployment**: Gate deployments on quality metrics, not just test pass/fail
2. **Staged Rollouts**: Gradual traffic shifting with quality monitoring
3. **Version Everything**: Track code, prompts, models, and configurations together
4. **Cost-Aware Testing**: Balance thorough evaluation with API costs
5. **Fast Feedback**: Cache LLM responses for faster iteration

---

## Pipeline Architecture

### High-Level Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Commit     │────▶│  Fast Tests  │────▶│  Evaluation  │────▶│   Deploy     │
│              │     │  (Mocked)    │     │  (Real LLM)  │     │   (Staged)   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
  Linting &           Unit Tests           Golden Dataset        Canary → Prod
  Type Check          Integration          A/B Comparison        Monitoring
```

### Pipeline Stages

```yaml
# .github/workflows/agent-pipeline.yml
name: Agent CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}

jobs:
  # Stage 1: Fast Checks (< 2 min)
  fast-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint
        run: ruff check .

      - name: Type check
        run: mypy src/

      - name: Security scan
        run: bandit -r src/

  # Stage 2: Unit Tests (< 5 min)
  unit-tests:
    needs: fast-checks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run unit tests (mocked LLM)
        run: pytest tests/unit -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  # Stage 3: Integration Tests (< 10 min)
  integration-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run integration tests
        run: pytest tests/integration -v --timeout=300

  # Stage 4: Evaluation (< 15 min)
  evaluation:
    needs: integration-tests
    runs-on: ubuntu-latest
    if: github.event_name == 'push' || github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run golden dataset evaluation
        id: eval
        run: |
          python scripts/evaluate.py \
            --dataset golden_dataset.json \
            --output eval_results.json

      - name: Check evaluation thresholds
        run: |
          python scripts/check_thresholds.py \
            --results eval_results.json \
            --min-accuracy 0.85 \
            --max-latency-p95 5000

      - name: Upload evaluation results
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-results
          path: eval_results.json

      - name: Post evaluation summary
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('eval_results.json'));
            const body = `## Evaluation Results

            | Metric | Value | Threshold |
            |--------|-------|-----------|
            | Accuracy | ${results.accuracy.toFixed(2)} | ≥ 0.85 |
            | P95 Latency | ${results.latency_p95}ms | ≤ 5000ms |
            | Tool Accuracy | ${results.tool_accuracy.toFixed(2)} | ≥ 0.90 |
            `;
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: body
            });

  # Stage 5: Deploy
  deploy-staging:
    needs: evaluation
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4

      - name: Deploy to staging
        run: ./scripts/deploy.sh staging

  deploy-production:
    needs: evaluation
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Deploy canary (10%)
        run: ./scripts/deploy.sh production --canary 10

      - name: Monitor canary (5 min)
        run: ./scripts/monitor_canary.sh --duration 300

      - name: Promote to full deployment
        run: ./scripts/deploy.sh production --promote
```

---

## GitHub Actions Workflows

### Evaluation Workflow

```yaml
# .github/workflows/evaluation.yml
name: Agent Evaluation

on:
  workflow_dispatch:
    inputs:
      dataset:
        description: 'Evaluation dataset'
        required: true
        default: 'golden'
        type: choice
        options:
          - golden
          - extended
          - regression
      model:
        description: 'Model to evaluate'
        required: true
        default: 'claude-sonnet-4-20250514'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[eval]"

      - name: Load dataset
        run: |
          python -c "
          from datasets import load_dataset
          dataset = '${{ inputs.dataset }}'
          # Load from LangSmith or local
          "

      - name: Run evaluation
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
        run: |
          python scripts/run_evaluation.py \
            --dataset ${{ inputs.dataset }} \
            --model ${{ inputs.model }} \
            --output results/ \
            --concurrency 10

      - name: Generate report
        run: python scripts/generate_report.py results/

      - name: Upload to LangSmith
        run: |
          python -c "
          from langsmith import Client
          client = Client()
          # Upload evaluation results
          "

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-${{ github.run_number }}
          path: results/
```

### Prompt Testing Workflow

```yaml
# .github/workflows/prompt-tests.yml
name: Prompt Regression Tests

on:
  push:
    paths:
      - 'prompts/**'
      - 'src/prompts/**'
  pull_request:
    paths:
      - 'prompts/**'
      - 'src/prompts/**'

jobs:
  prompt-regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Need history for comparison

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Detect prompt changes
        id: changes
        run: |
          CHANGED=$(git diff --name-only origin/main -- prompts/ src/prompts/ | tr '\n' ' ')
          echo "changed_files=$CHANGED" >> $GITHUB_OUTPUT

      - name: Run prompt tests
        if: steps.changes.outputs.changed_files != ''
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          python scripts/test_prompts.py \
            --changed "${{ steps.changes.outputs.changed_files }}" \
            --compare-baseline \
            --output prompt_test_results.json

      - name: Check for regressions
        run: |
          python scripts/check_prompt_regression.py \
            --results prompt_test_results.json \
            --threshold 0.9

      - name: Comment on PR
        if: github.event_name == 'pull_request' && failure()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('prompt_test_results.json'));
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: `⚠️ Prompt regression detected!\n\n${JSON.stringify(results.regressions, null, 2)}`
            });
```

### Nightly Evaluation

```yaml
# .github/workflows/nightly-eval.yml
name: Nightly Comprehensive Evaluation

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:

jobs:
  comprehensive-eval:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[eval]"

      - name: Run comprehensive evaluation
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
        run: |
          python scripts/comprehensive_eval.py \
            --datasets golden extended edge-cases \
            --models claude-sonnet-4-20250514 claude-haiku-3 \
            --output nightly_results/

      - name: Compare to baseline
        run: |
          python scripts/compare_to_baseline.py \
            --current nightly_results/ \
            --baseline baselines/production/

      - name: Send alerts if regression
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "🚨 Nightly evaluation detected regression",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "See details: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
                  }
                }
              ]
            }

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: nightly-eval-${{ github.run_number }}
          path: nightly_results/
          retention-days: 30
```

---

## Testing in CI

### Mocked LLM Tests

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock
import json

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for unit tests."""
    client = MagicMock()

    def create_mock_response(content: str, tool_use: dict = None):
        response = MagicMock()
        response.content = []

        if content:
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = content
            response.content.append(text_block)

        if tool_use:
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.name = tool_use["name"]
            tool_block.input = tool_use["input"]
            tool_block.id = "tool_123"
            response.content.append(tool_block)

        response.stop_reason = "tool_use" if tool_use else "end_turn"
        return response

    client.messages.create = MagicMock(side_effect=lambda **kwargs: create_mock_response(
        content="Mocked response",
        tool_use=None
    ))

    return client


@pytest.fixture
def cached_responses():
    """Load cached LLM responses for deterministic testing."""
    cache_path = Path("tests/fixtures/response_cache.json")
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


@pytest.fixture
def deterministic_agent(mock_anthropic_client, cached_responses):
    """Create agent with mocked, deterministic responses."""
    agent = Agent(client=mock_anthropic_client)

    # Override to use cached responses
    original_run = agent.run

    def cached_run(query: str):
        cache_key = hash(query)
        if str(cache_key) in cached_responses:
            return cached_responses[str(cache_key)]
        return original_run(query)

    agent.run = cached_run
    return agent
```

### Response Caching for Tests

```python
# scripts/cache_responses.py
"""Cache LLM responses for faster CI testing."""
import json
import hashlib
from pathlib import Path
from anthropic import Anthropic

def cache_test_responses(test_queries: list[str], output_path: str):
    """Generate cached responses for test queries."""
    client = Anthropic()
    cache = {}

    for query in test_queries:
        cache_key = hashlib.sha256(query.encode()).hexdigest()[:16]

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=0,  # Deterministic
            messages=[{"role": "user", "content": query}]
        )

        cache[cache_key] = {
            "query": query,
            "response": response.content[0].text,
            "model": "claude-sonnet-4-20250514"
        }

    Path(output_path).write_text(json.dumps(cache, indent=2))
    print(f"Cached {len(cache)} responses to {output_path}")


if __name__ == "__main__":
    test_queries = [
        "What is the weather in San Francisco?",
        "Book a flight from SFO to JFK",
        "What is my order status for #12345?",
        # Add more test queries
    ]

    cache_test_responses(test_queries, "tests/fixtures/response_cache.json")
```

### Evaluation Script

```python
# scripts/evaluate.py
"""Run agent evaluation against golden dataset."""
import argparse
import asyncio
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List
import time

from langsmith import Client
from agent import ProductionAgent

@dataclass
class EvaluationResult:
    accuracy: float
    tool_accuracy: float
    latency_mean: float
    latency_p95: float
    latency_p99: float
    total_examples: int
    passed_examples: int
    failed_examples: List[dict]

async def evaluate_agent(dataset_path: str, output_path: str, concurrency: int = 5):
    """Run evaluation on golden dataset."""
    with open(dataset_path) as f:
        dataset = json.load(f)

    agent = ProductionAgent()
    results = []
    latencies = []
    failed = []

    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_example(example: dict):
        async with semaphore:
            start = time.time()
            try:
                response = await agent.run_async(example["input"])
                latency = (time.time() - start) * 1000

                # Check correctness
                is_correct = check_correctness(response, example["expected"])
                tool_correct = check_tool_usage(agent.last_tool_calls, example.get("expected_tools", []))

                return {
                    "id": example["id"],
                    "correct": is_correct,
                    "tool_correct": tool_correct,
                    "latency_ms": latency,
                    "response": response[:500]
                }
            except Exception as e:
                return {
                    "id": example["id"],
                    "correct": False,
                    "tool_correct": False,
                    "latency_ms": (time.time() - start) * 1000,
                    "error": str(e)
                }

    tasks = [evaluate_example(ex) for ex in dataset["examples"]]
    results = await asyncio.gather(*tasks)

    # Calculate metrics
    correct = sum(1 for r in results if r["correct"])
    tool_correct = sum(1 for r in results if r.get("tool_correct", True))
    latencies = [r["latency_ms"] for r in results]
    failed = [r for r in results if not r["correct"]]

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    evaluation = EvaluationResult(
        accuracy=correct / len(results),
        tool_accuracy=tool_correct / len(results),
        latency_mean=sum(latencies) / n,
        latency_p95=latencies_sorted[int(n * 0.95)],
        latency_p99=latencies_sorted[int(n * 0.99)],
        total_examples=len(results),
        passed_examples=correct,
        failed_examples=failed[:10]  # First 10 failures
    )

    Path(output_path).write_text(json.dumps(asdict(evaluation), indent=2))
    return evaluation


def check_correctness(response: str, expected: str) -> bool:
    """Check if response matches expected output."""
    # Exact match
    if response.strip().lower() == expected.strip().lower():
        return True

    # Semantic similarity
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([response, expected])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return similarity >= 0.85


def check_tool_usage(actual_tools: List[str], expected_tools: List[str]) -> bool:
    """Check if correct tools were used."""
    if not expected_tools:
        return True
    return set(expected_tools).issubset(set(actual_tools))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    result = asyncio.run(evaluate_agent(args.dataset, args.output, args.concurrency))
    print(f"Accuracy: {result.accuracy:.2%}")
    print(f"P95 Latency: {result.latency_p95:.0f}ms")
```

---

## Deployment Strategies

### Blue-Green Deployment

```python
# scripts/deploy.py
"""Deployment script for agent system."""
import argparse
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class DeploymentConfig:
    environment: str
    version: str
    canary_percentage: int = 0
    health_check_url: str = None
    rollback_threshold: float = 0.05  # 5% error rate

class Deployer:
    def __init__(self, config: DeploymentConfig):
        self.config = config

    def deploy_blue_green(self):
        """Blue-green deployment strategy."""
        # 1. Deploy to inactive environment
        inactive = self._get_inactive_environment()
        self._deploy_to_environment(inactive)

        # 2. Health check
        if not self._health_check(inactive):
            raise Exception(f"Health check failed for {inactive}")

        # 3. Run smoke tests
        if not self._run_smoke_tests(inactive):
            raise Exception(f"Smoke tests failed for {inactive}")

        # 4. Switch traffic
        self._switch_traffic(inactive)

        # 5. Monitor
        self._monitor_deployment(duration_seconds=300)

    def deploy_canary(self, percentage: int):
        """Canary deployment strategy."""
        # 1. Deploy canary
        self._deploy_canary(percentage)

        # 2. Monitor canary
        metrics = self._monitor_canary(duration_seconds=300)

        # 3. Check thresholds
        if metrics["error_rate"] > self.config.rollback_threshold:
            self._rollback_canary()
            raise Exception(f"Canary failed: error rate {metrics['error_rate']:.2%}")

        return metrics

    def promote_canary(self):
        """Promote canary to full deployment."""
        # Gradually increase traffic
        for percentage in [25, 50, 75, 100]:
            self._set_canary_percentage(percentage)
            time.sleep(60)  # 1 minute between steps

            metrics = self._get_current_metrics()
            if metrics["error_rate"] > self.config.rollback_threshold:
                self._rollback_canary()
                raise Exception(f"Promotion failed at {percentage}%")

        print("Canary promoted successfully")

    def _deploy_to_environment(self, env: str):
        """Deploy to specific environment."""
        subprocess.run([
            "kubectl", "apply",
            "-f", f"k8s/{self.config.environment}/{env}/",
            "--namespace", self.config.environment
        ], check=True)

    def _health_check(self, env: str) -> bool:
        """Check environment health."""
        import httpx
        for _ in range(30):  # 30 attempts
            try:
                response = httpx.get(
                    f"{self.config.health_check_url}/{env}/health",
                    timeout=5
                )
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(2)
        return False

    def _run_smoke_tests(self, env: str) -> bool:
        """Run smoke tests against environment."""
        result = subprocess.run([
            "pytest", "tests/smoke/",
            "--env", env,
            "-v", "--timeout=60"
        ])
        return result.returncode == 0

    def _switch_traffic(self, target_env: str):
        """Switch all traffic to target environment."""
        subprocess.run([
            "kubectl", "patch", "service", "agent-service",
            "--namespace", self.config.environment,
            "-p", f'{{"spec":{{"selector":{{"env":"{target_env}"}}}}}}'
        ], check=True)

    def _monitor_deployment(self, duration_seconds: int):
        """Monitor deployment for issues."""
        start = time.time()
        while time.time() - start < duration_seconds:
            metrics = self._get_current_metrics()
            if metrics["error_rate"] > self.config.rollback_threshold:
                self._rollback()
                raise Exception(f"Deployment failed: error rate {metrics['error_rate']:.2%}")
            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("environment", choices=["staging", "production"])
    parser.add_argument("--canary", type=int, default=0)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()

    config = DeploymentConfig(
        environment=args.environment,
        version=get_current_version(),
        canary_percentage=args.canary
    )

    deployer = Deployer(config)

    if args.rollback:
        deployer.rollback()
    elif args.promote:
        deployer.promote_canary()
    elif args.canary > 0:
        deployer.deploy_canary(args.canary)
    else:
        deployer.deploy_blue_green()
```

### Kubernetes Deployment

```yaml
# k8s/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
  labels:
    app: agent-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-service
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: agent-service
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
    spec:
      containers:
        - name: agent
          image: agent-service:${VERSION}
          ports:
            - containerPort: 8080
          env:
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agent-secrets
                  key: anthropic-api-key
            - name: MODEL_VERSION
              value: "claude-sonnet-4-20250514"
            - name: PROMPT_VERSION
              valueFrom:
                configMapKeyRef:
                  name: agent-config
                  key: prompt-version
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 20
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
data:
  prompt-version: "v2.3.1"
  model-config: |
    {
      "primary_model": "claude-sonnet-4-20250514",
      "fallback_model": "claude-haiku-3",
      "temperature": 0.7,
      "max_tokens": 4096
    }
```

---

## Model and Prompt Management

### Version Control for Prompts

```python
# src/prompts/manager.py
"""Prompt version management."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from datetime import datetime
from typing import Optional

@dataclass
class PromptVersion:
    name: str
    version: str
    content: str
    hash: str
    created_at: str
    metadata: dict

class PromptManager:
    """Manage versioned prompts."""

    def __init__(self, prompts_dir: str):
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def save_prompt(self, name: str, content: str, metadata: dict = None) -> PromptVersion:
        """Save a new prompt version."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"

        prompt_version = PromptVersion(
            name=name,
            version=version,
            content=content,
            hash=content_hash,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )

        # Save to file
        prompt_dir = self.prompts_dir / name
        prompt_dir.mkdir(exist_ok=True)

        version_file = prompt_dir / f"{version}.json"
        with open(version_file, "w") as f:
            json.dump({
                "version": version,
                "content": content,
                "hash": content_hash,
                "created_at": prompt_version.created_at,
                "metadata": metadata
            }, f, indent=2)

        # Update latest pointer
        latest_file = prompt_dir / "latest.json"
        with open(latest_file, "w") as f:
            json.dump({"version": version}, f)

        return prompt_version

    def get_prompt(self, name: str, version: str = "latest") -> Optional[PromptVersion]:
        """Get a specific prompt version."""
        prompt_dir = self.prompts_dir / name

        if version == "latest":
            latest_file = prompt_dir / "latest.json"
            if not latest_file.exists():
                return None
            with open(latest_file) as f:
                version = json.load(f)["version"]

        version_file = prompt_dir / f"{version}.json"
        if not version_file.exists():
            return None

        with open(version_file) as f:
            data = json.load(f)

        return PromptVersion(
            name=name,
            version=data["version"],
            content=data["content"],
            hash=data["hash"],
            created_at=data["created_at"],
            metadata=data.get("metadata", {})
        )

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a prompt."""
        prompt_dir = self.prompts_dir / name
        if not prompt_dir.exists():
            return []

        versions = []
        for f in prompt_dir.glob("v*.json"):
            versions.append(f.stem)

        return sorted(versions, reverse=True)

    def diff_versions(self, name: str, v1: str, v2: str) -> dict:
        """Compare two prompt versions."""
        prompt1 = self.get_prompt(name, v1)
        prompt2 = self.get_prompt(name, v2)

        if not prompt1 or not prompt2:
            return {"error": "Version not found"}

        import difflib
        diff = list(difflib.unified_diff(
            prompt1.content.splitlines(),
            prompt2.content.splitlines(),
            fromfile=f"{name}@{v1}",
            tofile=f"{name}@{v2}",
            lineterm=""
        ))

        return {
            "v1": v1,
            "v2": v2,
            "diff": "\n".join(diff),
            "hash_changed": prompt1.hash != prompt2.hash
        }
```

### Configuration Management

```yaml
# config/production.yaml
agent:
  model:
    primary: claude-sonnet-4-20250514
    fallback: claude-haiku-3
    temperature: 0.7
    max_tokens: 4096

  prompts:
    version: v20251215120000
    fallback_version: v20251201000000

  tools:
    enabled:
      - web_search
      - database_query
      - email_send
    disabled:
      - file_write  # Disabled in production

  rate_limits:
    requests_per_minute: 100
    tokens_per_minute: 100000

  monitoring:
    log_level: INFO
    metrics_enabled: true
    trace_sampling_rate: 0.1

  safety:
    content_filter: strict
    pii_detection: enabled
    max_tool_iterations: 10
```

```python
# src/config.py
"""Configuration management with validation."""
from pydantic import BaseModel, Field
from typing import List, Optional
import yaml
from pathlib import Path

class ModelConfig(BaseModel):
    primary: str = "claude-sonnet-4-20250514"
    fallback: str = "claude-haiku-3"
    temperature: float = Field(ge=0, le=1, default=0.7)
    max_tokens: int = Field(ge=1, le=8192, default=4096)

class PromptConfig(BaseModel):
    version: str
    fallback_version: Optional[str] = None

class ToolsConfig(BaseModel):
    enabled: List[str] = []
    disabled: List[str] = []

class RateLimitsConfig(BaseModel):
    requests_per_minute: int = 100
    tokens_per_minute: int = 100000

class MonitoringConfig(BaseModel):
    log_level: str = "INFO"
    metrics_enabled: bool = True
    trace_sampling_rate: float = Field(ge=0, le=1, default=0.1)

class SafetyConfig(BaseModel):
    content_filter: str = "strict"
    pii_detection: str = "enabled"
    max_tool_iterations: int = 10

class AgentConfig(BaseModel):
    model: ModelConfig
    prompts: PromptConfig
    tools: ToolsConfig = ToolsConfig()
    rate_limits: RateLimitsConfig = RateLimitsConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    safety: SafetyConfig = SafetyConfig()

def load_config(env: str) -> AgentConfig:
    """Load configuration for environment."""
    config_path = Path(f"config/{env}.yaml")
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return AgentConfig(**data["agent"])
```

---

## Monitoring and Rollback

### Deployment Monitoring

```python
# scripts/monitor_deployment.py
"""Monitor deployment health and trigger rollback if needed."""
import time
from dataclasses import dataclass
from prometheus_api_client import PrometheusConnect

@dataclass
class DeploymentMetrics:
    error_rate: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    request_rate: float
    success_rate: float

class DeploymentMonitor:
    def __init__(self, prometheus_url: str, rollback_thresholds: dict):
        self.prom = PrometheusConnect(url=prometheus_url)
        self.thresholds = rollback_thresholds

    def get_current_metrics(self, window: str = "5m") -> DeploymentMetrics:
        """Get current deployment metrics."""
        queries = {
            "error_rate": f'sum(rate(agent_requests_total{{status="error"}}[{window}])) / sum(rate(agent_requests_total[{window}]))',
            "latency_p50": f'histogram_quantile(0.50, sum(rate(agent_request_duration_seconds_bucket[{window}])) by (le))',
            "latency_p95": f'histogram_quantile(0.95, sum(rate(agent_request_duration_seconds_bucket[{window}])) by (le))',
            "latency_p99": f'histogram_quantile(0.99, sum(rate(agent_request_duration_seconds_bucket[{window}])) by (le))',
            "request_rate": f'sum(rate(agent_requests_total[{window}]))',
        }

        results = {}
        for name, query in queries.items():
            result = self.prom.custom_query(query)
            results[name] = float(result[0]["value"][1]) if result else 0

        results["success_rate"] = 1 - results["error_rate"]

        return DeploymentMetrics(**results)

    def check_thresholds(self, metrics: DeploymentMetrics) -> tuple[bool, list[str]]:
        """Check if metrics exceed thresholds."""
        violations = []

        if metrics.error_rate > self.thresholds.get("max_error_rate", 0.05):
            violations.append(f"Error rate {metrics.error_rate:.2%} > {self.thresholds['max_error_rate']:.2%}")

        if metrics.latency_p95 > self.thresholds.get("max_latency_p95", 5.0):
            violations.append(f"P95 latency {metrics.latency_p95:.2f}s > {self.thresholds['max_latency_p95']:.2f}s")

        if metrics.latency_p99 > self.thresholds.get("max_latency_p99", 10.0):
            violations.append(f"P99 latency {metrics.latency_p99:.2f}s > {self.thresholds['max_latency_p99']:.2f}s")

        return len(violations) == 0, violations

    def monitor_and_alert(self, duration_seconds: int, check_interval: int = 30):
        """Monitor deployment and alert on issues."""
        start = time.time()

        while time.time() - start < duration_seconds:
            metrics = self.get_current_metrics()
            healthy, violations = self.check_thresholds(metrics)

            if not healthy:
                print(f"⚠️ Threshold violations: {violations}")
                self._send_alert(violations)

                # Check if should rollback
                if self._should_rollback(metrics):
                    print("🔴 Initiating automatic rollback")
                    return False, violations
            else:
                print(f"✅ Metrics healthy: error_rate={metrics.error_rate:.2%}, p95={metrics.latency_p95:.2f}s")

            time.sleep(check_interval)

        return True, []

    def _should_rollback(self, metrics: DeploymentMetrics) -> bool:
        """Determine if automatic rollback should trigger."""
        # Critical thresholds for automatic rollback
        return (
            metrics.error_rate > 0.10 or  # 10% error rate
            metrics.latency_p95 > 30.0    # 30s P95 latency
        )

    def _send_alert(self, violations: list[str]):
        """Send alert to monitoring channel."""
        # Implement Slack/PagerDuty/etc. alerting
        pass
```

### Rollback Automation

```python
# scripts/rollback.py
"""Automated rollback for failed deployments."""
import subprocess
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class RollbackTarget:
    version: str
    config_version: str
    prompt_version: str
    timestamp: str

class RollbackManager:
    def __init__(self, environment: str):
        self.environment = environment
        self.history_file = f"deployments/{environment}/history.json"

    def get_previous_deployment(self, steps_back: int = 1) -> Optional[RollbackTarget]:
        """Get previous deployment info."""
        with open(self.history_file) as f:
            history = json.load(f)

        if len(history) <= steps_back:
            return None

        prev = history[-(steps_back + 1)]
        return RollbackTarget(**prev)

    def rollback(self, target: RollbackTarget):
        """Execute rollback to target version."""
        print(f"Rolling back to {target.version}")

        # 1. Rollback application
        subprocess.run([
            "kubectl", "rollout", "undo",
            f"deployment/agent-service",
            f"--to-revision={target.version}",
            "--namespace", self.environment
        ], check=True)

        # 2. Rollback configuration
        subprocess.run([
            "kubectl", "apply",
            "-f", f"config/versions/{target.config_version}/",
            "--namespace", self.environment
        ], check=True)

        # 3. Rollback prompts
        self._rollback_prompts(target.prompt_version)

        # 4. Wait for rollback to complete
        subprocess.run([
            "kubectl", "rollout", "status",
            "deployment/agent-service",
            "--namespace", self.environment,
            "--timeout=300s"
        ], check=True)

        print(f"✅ Rollback to {target.version} complete")

    def _rollback_prompts(self, version: str):
        """Rollback prompts to specific version."""
        from prompts.manager import PromptManager
        manager = PromptManager("prompts/")

        # Update latest pointers to rollback version
        # Implementation depends on prompt storage
        pass

    def record_deployment(self, target: RollbackTarget):
        """Record deployment in history."""
        try:
            with open(self.history_file) as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []

        history.append({
            "version": target.version,
            "config_version": target.config_version,
            "prompt_version": target.prompt_version,
            "timestamp": target.timestamp
        })

        # Keep last 50 deployments
        history = history[-50:]

        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("environment", choices=["staging", "production"])
    parser.add_argument("--steps-back", type=int, default=1)
    args = parser.parse_args()

    manager = RollbackManager(args.environment)
    target = manager.get_previous_deployment(args.steps_back)

    if target:
        manager.rollback(target)
    else:
        print("No previous deployment found")
```

---

## Best Practices Summary

### Pipeline Design

1. **Fast feedback first**: Run lint/type checks before slow LLM tests
2. **Cache responses**: Use cached LLM responses for deterministic unit tests
3. **Gate on metrics**: Require evaluation metrics to pass, not just tests
4. **Cost awareness**: Limit API calls in CI, use sampling for large datasets

### Testing Strategy

1. **Layer tests**: Unit (mocked) → Integration (real LLM) → Evaluation (golden dataset)
2. **Prompt regression**: Test prompts separately with their own pipeline
3. **Smoke tests**: Quick sanity checks before full evaluation
4. **Nightly comprehensive**: Full evaluation suite runs overnight

### Deployment

1. **Staged rollouts**: Canary → Gradual promotion → Full deployment
2. **Automatic rollback**: Trigger on metric thresholds
3. **Version everything**: Code + prompts + config as atomic unit
4. **Monitor actively**: Watch metrics closely after each deployment

---

## Related Documents

- [Testing Guide](testing-guide.md) - Testing strategies
- [API Optimization Guide](api-optimization-guide.md) - Performance optimization
- [Evaluation and Debugging](evaluation-and-debugging.md) - Debugging strategies
- [Security Essentials](../phase-5-security-compliance/security-essentials.md) - Security in CI/CD
