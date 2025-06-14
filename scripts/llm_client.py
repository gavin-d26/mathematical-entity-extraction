"""
LLM client module for mathematical entity extraction using vLLM offline inference.
"""

from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Optional
import time


class VLLMClient:
    """Client for vLLM offline batch inference."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        gpu_id: int = 5,
    ):
        """
        Initialize the vLLM offline client.

        Args:
            model_name: Name/path of the model to use
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            max_model_len: Maximum model context length
            gpu_memory_utilization: GPU memory utilization ratio
            gpu_id: GPU ID to use
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.gpu_id = gpu_id

        # Initialize the LLM engine
        print(f"Initializing vLLM with model: {model_name} on GPU {gpu_id}")
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,  # Required for some models
            enforce_eager=True,  # Faster execution for debugging
        )
        print("vLLM initialization complete")

    def generate_single(
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.1
    ) -> str:
        """
        Generate a single completion.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=None,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def generate_batch(
        self, prompts: List[str], max_tokens: int = 4096, temperature: float = 0.1
    ) -> List[str]:
        """
        Generate completions for a batch of prompts using vLLM's efficient batch processing.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature

        Returns:
            List of generated texts
        """
        print(f"Processing batch of {len(prompts)} prompts with vLLM")

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=None,
        )

        # vLLM handles batching efficiently internally
        outputs = self.llm.generate(prompts, sampling_params)

        # Extract generated text from outputs
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)

        print(f"Batch processing complete: {len(results)} results generated")
        return results

    def test_connection(self) -> bool:
        """
        Test if the vLLM engine is working properly.

        Returns:
            True if engine is working, False otherwise
        """
        try:
            test_prompt = "Hello, world!"
            result = self.generate_single(test_prompt, max_tokens=10)
            return len(result) > 0
        except Exception as e:
            print(f"vLLM engine test failed: {e}")
            return False

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the loaded model.

        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "gpu_id": self.gpu_id,
        }
