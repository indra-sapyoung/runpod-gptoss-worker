import os
import sys
import logging
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

logging.basicConfig(level=logging.INFO)

try:
    vllm_engine = vLLMEngine()
    openai_engine = OpenAIvLLMEngine(vllm_engine)
except Exception as e:
    logging.error(f"❌ Failed to initialize engine: {e}")
    sys.exit(1)  # Exit so RunPod replaces this worker

async def handler(job):
    try:
        job_input = JobInput(job["input"])
        engine = openai_engine if job_input.openai_route else vllm_engine
        results_generator = engine.generate(job_input)
        async for batch in results_generator:
            yield batch
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        # If engine is dead, exit so RunPod replaces this worker
        if any(keyword in error_msg for keyword in ["EngineCore encountered", "Engine core", "engine is dead"]) \
           or any(keyword in error_type for keyword in ["EngineDeadError", "EngineCoreError"]):
            logging.error(f"❌ Engine died ({error_type}): {e}. Exiting worker.")
            os._exit(1)  # Force exit (sys.exit may be caught by RunPod framework)
        raise

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)