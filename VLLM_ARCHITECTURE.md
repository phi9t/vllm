# vLLM Architecture Design Document

> A comprehensive guide to the vLLM LLM serving system architecture
> Version: 1.0 | Last Updated: 2024

---

## 1. Executive Summary

vLLM is a high-performance, open-source library for LLM inference and serving. Its core innovation is **PagedAttention**, which enables efficient KV cache management and continuous batching, dramatically improving throughput compared to naive approaches.

### Key Architectural Principles

1. **Memory Efficiency**: PagedAttention eliminates memory fragmentation and enables KV cache sharing
2. **Throughput Optimization**: Continuous batching with iteration-level scheduling maximizes GPU utilization
3. **Modularity**: Clean separation between engine, scheduler, model executor, and attention backends
4. **Extensibility**: Plugin architecture for different model types and hardware backends

### Performance Characteristics

- **Throughput**: 2-4x higher than naive batching (varies by workload)
- **Latency**: Predictable with iteration-level scheduling
- **Memory**: Near-zero fragmentation with block-based KV cache
- **Scalability**: Distributed serving with tensor/pipeline parallelism

---

## 2. Hierarchical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT INTERFACE LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ OpenAI API  │  │  LLM Class  │  │ AsyncEngine │  │   SamplingParams    │ │
│  │  (entrypoints│  │ (vllm/entrypoints│  │ (vllm/engine/async_llm.py)│  │ (vllm/sampling_params.py)│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM ENGINE LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        LLMEngine / AsyncLLMEngine                        ││
│  │                    (vllm/engine/llm_engine.py)                           ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ ││
│  │  │  Scheduler  │  │  Policy     │  │  Processor  │  │  OutputHandler  │ ││
│  │  │ (scheduler.py│  │(policy.py)  │  │(processor.py│  │(output_handler.py)│ ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL EXECUTOR LAYER                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      ModelRunner / GPUExecutor                           ││
│  │              (vllm/worker/model_runner.py)                               ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ ││
│  │  │  Attention  │  │   Sampler   │  │  LogitsProc │  │  Model Wrapper  │ ││
│  │  │  (attention/│  │(samplers.py)│  │(logits_processor.py)│ (model_executor/│ ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WORKER LAYER                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │   Worker (worker.py)│  │  CacheEngine        │  │  TokenizerGroup     │   │
│  │  ┌───────────────┐  │  │  (cache_engine.py)  │  │  (tokenizer.py)     │   │
│  │  │  GPU Worker   │  │  │  ┌─────────────┐    │  └─────────────────────┘   │
│  │  │  (gpu_worker.py)│  │  │  │ KV Cache    │    │                          │
│  │  └───────────────┘  │  │  │  Manager    │    │                          │
│  │  ┌───────────────┐  │  │  └─────────────┘    │                          │
│  │  │  Ray Worker   │  │  └─────────────────────┘                          │
│  │  │ (ray_gpu_worker.py)│                                                    │
│  │  └───────────────┘  │                                                      │
│  └─────────────────────┘                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CORE DATA STRUCTURES                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Request   │  │  Sequence   │  │SequenceGroup│  │  KVCacheBlock       │ │
│  │ (sequences.py)│  │ (sequences.py)│  │(sequences.py)│  │(block_manager.py)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ BlockTable  │  │  Inputs     │  │  Sampling   │  │  AttentionMetadata  │ │
│  │(block_manager.py)│  │(sampling_params.py)│  │(sampling_params.py)│  │(attention/)        │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ATTENTION BACKENDS                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   FlashAttn │  │    xFormers │  │ FlashInfer  │  │     PagedAttention  │ │
│  │ (flash_attn.py)│  │(xformers.py)│  │(flashinfer.py)│  │  (ops/paged_attn/)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐                                                            │
│  │    ROCm     │  │    IPEX     │                                                            │
│  │ (rocm_flash_attn.py)│  │(ipex_attn.py)│                                                            │
│  └─────────────┘  └─────────────┘                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL DEFINITIONS                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    LLaMA    │  │    GPT-NeoX │  │    Mixtral  │  │      Other Models   │ │
│  │ (llama.py)  │  │ (gpt_neox.py)│  │ (mixtral.py)│  │  (models/*.py)      │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Top 10 Most Important Building Blocks

| Rank | Component | Purpose | Key File(s) | Lines of Impact |
|------|-----------|---------|-------------|-----------------|
| 1 | **PagedAttention** | Core innovation - manages KV cache as fixed-size blocks, eliminating fragmentation and enabling sharing | `vllm/attention/ops/paged_attn.py`, `vllm/core/block_manager.py` | 10,000+ |
| 2 | **LLMEngine** | Central orchestrator - coordinates scheduler, model executor, and output processing | `vllm/engine/llm_engine.py` | 5,000+ |
| 3 | **Scheduler** | Iteration-level batching - decides which requests run each step | `vllm/core/scheduler.py`, `vllm/core/policy.py` | 3,000+ |
| 4 | **BlockManager** | Allocates and tracks KV cache blocks, handles prefix caching | `vllm/core/block_manager.py`, `vllm/core/block_allocator.py` | 2,500+ |
| 5 | **ModelRunner** | Executes forward passes on GPU, manages attention and sampling | `vllm/worker/model_runner.py` | 3,000+ |
| 6 | **Sampler** | Generates next tokens with temperature, top-p, top-k, etc. | `vllm/model_executor/layers/sampler.py` | 1,500+ |
| 7 | **Sequence** | Represents a single sequence of tokens and its metadata | `vllm/sequences.py` | 800+ |
| 8 | **CacheEngine** | Manages GPU/CPU KV cache memory, handles swapping | `vllm/worker/cache_engine.py` | 600+ |
| 9 | **Request/Inputs** | Encapsulates user prompt, parameters, and output handling | `vllm/inputs.py`, `vllm/entrypoints/llm.py` | 1,200+ |
| 10 | **SamplingParams** | Configuration for generation parameters (temperature, etc.) | `vllm/sampling_params.py` | 400+ |

### Detailed Component Analysis

#### 1. PagedAttention
```python
# Core concept: Fixed-size block management
class PagedAttention:
    """KV cache stored as fixed-size blocks (e.g., 16 tokens/block)
    
    Benefits:
    - No external fragmentation
    - Memory sharing between sequences
    - Efficient prefix caching
    """
    
    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.block_table = {}  # seq_id -> list of block indices
```

#### 2. LLMEngine
```python
class LLMEngine:
    """Main entry point for inference
    
    Responsibilities:
    - Manage request lifecycle
    - Coordinate scheduler and executor
    - Handle output processing
    """
    
    def step(self) -> List[RequestOutput]:
        # 1. Schedule requests
        # 2. Run model forward
        # 3. Process outputs
        # 4. Update sequence states
```

#### 3. Scheduler
```python
class Scheduler:
    """Iteration-level batching decisions
    
    Key methods:
    - schedule(): Pick requests for next iteration
    - fork(): Copy sequence for beam search
    - free(): Release completed sequences
    """
    
    def schedule(self) -> SchedulerOutput:
        # Priority: running > swapped > waiting
        # Respects: max_num_seqs, max_num_batched_tokens
```

---

## 4. Critical Data Flow: Request Lifecycle

### Complete Request Journey

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REQUEST LIFECYCLE FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: REQUEST SUBMISSION
┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────────────┐
│   User      │────▶│  LLMEngine  │────▶│        Request Processing           │
│  Request    │     │  (add_request)     │  - Validate inputs                  │
└─────────────┘     └─────────────┘     │  - Create Sequence objects          │
                                        │  - Initialize sampling params       │
                                        └─────────────────────────────────────┘
                                                      │
                                                      ▼
                                        ┌─────────────────────────────────────┐
                                        │      Priority Assignment            │
                                        │  - Add to waiting queue             │
                                        │  - Set arrival time                 │
                                        └─────────────────────────────────────┘

PHASE 2: SCHEDULING DECISION
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCHEDULER ITERATION                                 │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  1. Allocate Blocks (BlockManagerV2.allocate)                       │   │
│  │     - Check available GPU memory                                     │   │
│  │     - Allocate BlockTable for sequence                               │   │
│  │     - Check prefix cache (cached blocks first)                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  2. Select Batching Strategy (Policy)                               │   │
│  │     - Priority: running > swapped > waiting                          │   │
│  │     - Respect: max_num_seqs, max_num_batched_tokens                  │   │
│  │     - Chunk prefill for large prompts                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  3. Return SchedulerOutput                                          │   │
│  │     - scheduled_seq_groups: What to run this iteration               │   │
│  │     - num_batched_tokens: Total tokens in batch                      │   │
│  │     - blocks_to_swap_in/out: For CPU offloading                      │   │
│  │     - blocks_to_copy: For beam search/prefix sharing                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 3: MODEL EXECUTION
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL FORWARD PASS                                  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  ModelRunner.execute_model(scheduler_output)                         │   │
│  │                                                                              │   │
│  │  1. Prepare Inputs                                                   │   │
│  │     - Token IDs: [batch_size, seq_len]                               │   │
│  │     - Positions: Token positions in sequence                         │   │
│  │     - BlockTables: Physical block locations                          │   │
│  │                                                                              │   │
│  │  2. Build AttentionMetadata                                          │   │
│  │     - query_lens: Lengths of query tokens (usually 1 for decode)     │   │
│  │     - seq_lens: Total sequence lengths                               │   │
│  │     - slot_mapping: KV cache slot indices                            │   │
│  │                                                                              │   │
│  │  3. Forward Pass through Model                                       │   │
│  │     - Embedding lookup                                               │   │
│  │     - Transformer layers with PagedAttention                         │   │
│  │     - Output logits                                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 4: SAMPLING & OUTPUT
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOKEN GENERATION                                    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  1. LogitsProcessor (optional)                                       │   │
│  │     - Temperature scaling                                            │   │
│  │     - Top-p, Top-k filtering                                         │   │
│  │     - Presence/frequency penalties                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  2. Sampler.sample()                                                 │   │
│  │     - Softmax over logits                                            │   │
│  │     - Sample from distribution                                       │   │
│  │     - Return: token_ids, logprobs                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  3. Output Handler                                                   │   │
│  │     - Add token to sequence                                          │   │
│  │     - Check for stop conditions                                      │   │
│  │     - Build RequestOutput                                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 5: ITERATION COMPLETION
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STATE MANAGEMENT                                    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  1. Update Sequence States                                           │   │
│  │     - RUNNING → Waiting if preempted                                 │   │
│  │     - RUNNING → Finished if complete                                 │   │
│  │                                                                              │   │
│  │  2. Manage KV Cache                                                  │   │
│  │     - Allocate new blocks if needed                                  │   │
│  │     - Free blocks for finished sequences                             │   │
│  │     - Write to prefix cache if enabled                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  3. Return RequestOutput                                             │   │
│  │     - prompt: Original input                                         │   │
│  │     - prompt_token_ids: Tokenized input                              │   │
│  │     - outputs: Generated text + tokens + logprobs                    │   │
│  │     - finished: Whether generation is complete                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Sequence State Transitions

```
                    ┌─────────────┐
                    │   WAITING   │
                    │  (in queue) │
                    └──────┬──────┘
                           │ schedule()
                           ▼
                    ┌─────────────┐
    ┌──────────────▶│   RUNNING   │◀────────────────┐
    │               │  (executing)│                 │
    │               └──────┬──────┘                 │
    │                      │                        │
    │         ┌────────────┼────────────┐          │
    │         │            │            │          │
    │         ▼            ▼            ▼          │
    │  ┌─────────────┐┌──────────┐┌────────────┐   │
    │  │   SWAPPED   ││ FINISHED ││ CANCELLED  │   │
    │  │(CPU offload)││(complete)││ (aborted)  │   │
    │  └──────┬──────┘└──────────┘└────────────┘   │
    │         │                                    │
    └─────────┘ swap_in()                          │
                      └────────────────────────────┘
                                      continue
```

---

## 5. Core Primitives

### 5.1 Request

```python
@dataclass
class Request:
    """User's inference request
    
    Attributes:
        request_id: Unique identifier
        prompt: Input text or token IDs
        params: SamplingParams configuration
        arrival_time: When request was received
    """
    request_id: str
    prompt: Union[str, List[int]]
    params: SamplingParams
    arrival_time: float
```

### 5.2 Sequence

```python
class Sequence:
    """Represents a single sequence of tokens
    
    A sequence maintains:
    - Token IDs generated so far
    - KV cache block table mapping
    - Generation state (running/finished)
    - Output data (logprobs, etc.)
    """
    
    def __init__(self, seq_id, prompt_token_ids, block_size):
        self.seq_id = seq_id
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids = []
        self.block_table = BlockTable(block_size)
        self.status = SequenceStatus.WAITING
        
    def get_token_ids(self) -> List[int]:
        """Return all token IDs (prompt + generated)"""
        return self.prompt_token_ids + self.output_token_ids
        
    def append_token(self, token_id: int, logprob: float):
        """Add a newly generated token"""
        self.output_token_ids.append(token_id)
        self.output_logprobs.append(logprob)
```

### 5.3 SequenceGroup

```python
class SequenceGroup:
    """Group of sequences from same request (for beam search)
    
    A request may spawn multiple sequences:
    - n=1: Single generation
    - n>1: Multiple samples
    - beam_search: Multiple beams
    """
    
    def __init__(self, request_id, seqs, sampling_params, arrival_time):
        self.request_id = request_id
        self.seqs = seqs  # List[Sequence]
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        self.best_of = sampling_params.best_of
        
    def get_seqs(self, status: Optional[SequenceStatus] = None) -> List[Sequence]:
        """Get sequences, optionally filtered by status"""
        if status is None:
            return self.seqs
        return [seq for seq in self.seqs if seq.status == status]
```

### 5.4 KVCacheBlock

```python
@dataclass
class KVCacheBlock:
    """Fixed-size block for KV cache storage
    
    Each block stores:
    - block_number: Physical index in KV cache
    - ref_count: Number of sequences sharing this block
    - last_accessed: Timestamp for LRU eviction
    - is_merged: Whether block is shared (prefix caching)
    """
    block_number: int
    ref_count: int = 0
    last_accessed: float = 0.0
    computed: bool = False
    
    def incr_ref(self):
        self.ref_count += 1
        
    def decr_ref(self):
        self.ref_count -= 1
        return self.ref_count == 0
```

### 5.5 BlockTable

```python
class BlockTable:
    """Maps logical sequence positions to physical KV cache blocks
    
    Example with block_size=4:
    Sequence: [t0, t1, t2, t3, t4, t5, t6, t7, t8]
    BlockTable: [block_0, block_1, block_2]
                t0-t3    t4-t7    t8
    """
    
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.blocks: List[KVCacheBlock] = []
        
    def get_block_numbers(self) -> List[int]:
        """Return list of physical block indices"""
        return [block.block_number for block in self.blocks]
        
    def get_num_blocks(self) -> int:
        """Number of blocks allocated"""
        return len(self.blocks)
        
    def allocate(self, block: KVCacheBlock):
        """Add a new block to the table"""
        self.blocks.append(block)
        block.incr_ref()
```

### 5.6 SamplingMetadata

```python
@dataclass
class SamplingMetadata:
    """Sampling configuration for a batch
    
    Contains per-sequence sampling parameters:
    - Temperature for scaling logits
    - Top-p and top-k for filtering
    - Penalties for repetition
    """
    temperature: torch.Tensor  # [num_seqs]
    top_p: torch.Tensor        # [num_seqs]
    top_k: torch.Tensor        # [num_seqs]
    min_p: torch.Tensor        # [num_seqs]
    
    # Penalties
    frequency_penalties: torch.Tensor   # [num_seqs]
    presence_penalties: torch.Tensor    # [num_seqs]
    repetition_penalties: torch.Tensor  # [num_seqs]
    
    # Token indices for penalty calculation
    output_token_ids: List[List[int]]
```

---

## 6. Key Design Patterns

### 6.1 Factory Pattern

```python
# Model Registry
_MODEL_REGISTRY = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "MistralForCausalLM": MistralForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
    # ... more models
}

def get_model_cls(arch: str) -> Type[nn.Module]:
    """Factory: Get model class by architecture name"""
    if arch not in _MODEL_REGISTRY:
        raise ValueError(f"Unsupported architecture: {arch}")
    return _MODEL_REGISTRY[arch]

# Attention Backend Factory
def get_attention_backend(backend_name: str):
    """Factory: Get attention implementation"""
    backends = {
        "flash_attn": FlashAttentionBackend,
        "xformers": XFormersBackend,
        "flashinfer": FlashInferBackend,
    }
    return backends.get(backend_name)
```

### 6.2 Strategy Pattern

```python
class SchedulingPolicy(ABC):
    """Strategy: Different scheduling policies"""
    
    @abstractmethod
    def sort_by_priority(
        self,
        waiting_queue: List[SequenceGroup]
    ) -> List[SequenceGroup]:
        pass

class FCFS(SchedulingPolicy):
    """First-Come-First-Served"""
    def sort_by_priority(self, waiting_queue):
        return sorted(waiting_queue, key=lambda x: x.arrival_time)

class Priority(SchedulingPolicy):
    """Priority-based scheduling"""
    def sort_by_priority(self, waiting_queue):
        return sorted(waiting_queue, 
                     key=lambda x: (x.priority, x.arrival_time))
```

### 6.3 Observer Pattern

```python
class OutputHandler:
    """Observer: Receives and processes generation outputs"""
    
    def __init__(self):
        self.callbacks: List[Callable] = []
        
    def add_callback(self, callback: Callable):
        """Register output callback"""
        self.callbacks.append(callback)
        
    def on_output(self, request_output: RequestOutput):
        """Notify all observers of new output"""
        for callback in self.callbacks:
            callback(request_output)

# Usage in AsyncLLMEngine
async def generate(self, prompt, sampling_params):
    request_id = str(uuid.uuid4())
    
    # Create observable output queue
    output_queue: asyncio.Queue = asyncio.Queue()
    self.output_handler.add_callback(output_queue.put)
    
    # Add request
    self.add_request(request_id, prompt, sampling_params)
    
    # Observe outputs until finished
    while True:
        output = await output_queue.get()
        if output.finished:
            break
        yield output
```

### 6.4 Command Pattern

```python
class WorkerCommand(ABC):
    """Command: Encapsulate operations on worker"""
    
    @abstractmethod
    def execute(self, worker: Worker) -> Any:
        pass

class ExecuteModelCommand(WorkerCommand):
    """Command: Execute model forward pass"""
    
    def __init__(self, execute_model_req: ExecuteModelRequest):
        self.execute_model_req = execute_model_req
        
    def execute(self, worker: Worker) -> ModelOutput:
        return worker.execute_model(self.execute_model_req)

class SwapBlocksCommand(WorkerCommand):
    """Command: Swap KV cache blocks between GPU and CPU"""
    
    def __init__(self, blocks_to_swap: Dict[int, int]):
        self.blocks_to_swap = blocks_to_swap
        
    def execute(self, worker: Worker) -> None:
        worker.swap_blocks(self.blocks_to_swap)
```

### 6.5 State Pattern

```python
class SequenceStatus(Enum):
    """State: Sequence lifecycle states"""
    WAITING = 0   # Waiting for allocation
    RUNNING = 1   # Currently executing
    SWAPPED = 2   # Swapped to CPU
    FINISHED_STOPPED = 3   # Completed normally
    FINISHED_LENGTH_CAPPED = 4  # Reached max tokens
    FINISHED_ABORTED = 5   # User cancelled
    FINISHED_IGNORED = 6   # Prompt too long

class Sequence:
    """Context: Sequence with state"""
    
    def __init__(self):
        self._status = SequenceStatus.WAITING
        
    @property
    def status(self) -> SequenceStatus:
        return self._status
        
    def set_status(self, new_status: SequenceStatus):
        """State transition with validation"""
        valid_transitions = {
            SequenceStatus.WAITING: [SequenceStatus.RUNNING],
            SequenceStatus.RUNNING: [
                SequenceStatus.SWAPPED, 
                SequenceStatus.FINISHED_STOPPED,
                SequenceStatus.FINISHED_LENGTH_CAPPED,
                SequenceStatus.FINISHED_ABORTED
            ],
            SequenceStatus.SWAPPED: [SequenceStatus.WAITING],
        }
        
        if new_status not in valid_transitions.get(self._status, []):
            raise ValueError(f"Invalid transition: {self._status} -> {new_status}")
            
        self._status = new_status
```

---

## 7. Deep Dive Sections

### 7.1 PagedAttention: The Core Innovation

#### Problem Statement

Traditional LLM serving has a critical memory inefficiency:

```
┌─────────────────────────────────────────────────────────────────┐
│                    NAIVE KV CACHE APPROACH                       │
│                                                                   │
│  Request A (prompt=100, max_tokens=200)                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  [KV for 300 tokens]  ← Allocated upfront                   │ │
│  │  Actual usage: varies from 100 → 300                        │ │
│  │  Internal fragmentation: High                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Request B (prompt=50, max_tokens=100)                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  [KV for 150 tokens]  ← Allocated upfront                   │ │
│  │  Actual usage: varies from 50 → 150                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  PROBLEMS:                                                        │
│  1. Over-allocation: Reserves max_tokens for every request        │
│  2. External fragmentation: Variable sizes waste memory           │
│  3. No sharing: Common prefixes stored redundantly                │
└─────────────────────────────────────────────────────────────────┘
```

#### PagedAttention Solution

```
┌─────────────────────────────────────────────────────────────────┐
│                    PAGED KV CACHE APPROACH                       │
│                                                                   │
│  Physical KV Cache (fixed-size blocks = 16 tokens)               │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐   │
│  │ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5 │   │
│  │  [Used] │  [Used] │  [Free] │  [Used] │  [Used] │  [Free] │   │
│  └────┬────┴────┬────┴─────────┴────┬────┴────┬────┴─────────┘   │
│       │         │                   │         │                   │
│       └────┐    └─────────────┐     │         │                   │
│            │                  │     │         │                   │
│  Request A │                  │     │ Request B                    │
│  (3 blocks)│                  │     │ (2 blocks)                   │
│  ┌─────────┴─────────┬─────────┐    │  ┌─────────┴─────────┐        │
│  │  Block 0 (t0-t15) │ Block 1 │    │  │     Block 3       │        │
│  │  Block 2 (t16-31) │(t32-47) │    │  │     Block 4       │        │
│  └───────────────────┴─────────┘    │  └───────────────────┘        │
│                                     │                               │
│  Logical → Physical Mapping via BlockTable                       │
│  Token i → Block[i // block_size], Slot[i % block_size]          │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Implementation Details

**Block Allocation**

```python
class BlockManagerV2:
    """Manages allocation of KV cache blocks"""
    
    def allocate(self, seq_group: SequenceGroup) -> None:
        """Allocate blocks for a sequence group"""
        for seq in seq_group.get_seqs():
            num_required_blocks = len(seq.get_token_ids()) // self.block_size + 1
            
            # Try to allocate from free list
            for _ in range(num_required_blocks - len(seq.block_table)):
                if self.free_blocks:
                    block = self.free_blocks.pop()
                    seq.block_table.append(block)
                else:
                    # Out of memory - sequence must wait or be preempted
                    raise NoFreeBlocksError()
```

**Attention Computation**

```python
def paged_attention_forward(
    query: torch.Tensor,           # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,       # [num_blocks, block_size, num_heads, head_size]
    value_cache: torch.Tensor,     # [num_blocks, block_size, num_heads, head_size]
    block_tables: torch.Tensor,    # [num_seqs, max_num_blocks_per_seq]
    seq_lens: torch.Tensor,        # [num_seqs]
    block_size: int,
    max_seq_len: int,
) -> torch.Tensor:
    """
    Compute attention using paged KV cache.
    
    For each query token:
    1. Get its sequence's block table
    2. Gather KV cache from physical blocks
    3. Compute attention over all prefix tokens
    """
    output = torch.empty_like(query)
    
    # Reshape query for attention computation
    query = query.view(-1, num_heads, head_size)
    
    # For each sequence in the batch
    for seq_idx in range(len(seq_lens)):
        seq_len = seq_lens[seq_idx]
        blocks = block_tables[seq_idx, : (seq_len + block_size - 1) // block_size]
        
        # Gather KV cache from physical blocks
        keys = key_cache[blocks].view(-1, num_heads, head_size)[:seq_len]
        values = value_cache[blocks].view(-1, num_heads, head_size)[:seq_len]
        
        # Compute attention: softmax(Q @ K.T / sqrt(d)) @ V
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / sqrt(head_size)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output[seq_idx] = torch.matmul(attn_weights, values)
    
    return output
```

**Block Copying for Beam Search**

```python
def copy_blocks(
    kv_cache: List[torch.Tensor],
    block_mapping: Dict[int, List[int]]
) -> None:
    """
    Copy KV cache blocks for beam search.
    
    When a sequence forks in beam search, we need to copy its KV cache
    to the new beam. PagedAttention enables efficient copying by just
    updating block references (reference counting).
    
    Args:
        kv_cache: [key_cache, value_cache] tensors
        block_mapping: {src_block: [dst_blocks]} mapping
    """
    for src_block, dst_blocks in block_mapping.items():
        for dst_block in dst_blocks:
            # Copy entire block at once (efficient GPU operation)
            for cache in kv_cache:
                cache[dst_block].copy_(cache[src_block])
```

**Memory Sharing Visualization**

```
┌─────────────────────────────────────────────────────────────────┐
│                    BLOCK SHARING EXAMPLE                         │
│                                                                   │
│  System Prompt: "You are a helpful AI assistant."               │
│  Tokens: [100, 234, 456, 789, ...]                               │
│                                                                   │
│  User A Request: "What is Python?"                              │
│  User B Request: "What is Java?"                                │
│  User C Request: "Explain recursion."                           │
│                                                                   │
│  PHYSICAL BLOCKS:                                                │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┐             │
│  │ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │             │
│  │ (ref=3) │ (ref=1) │ (ref=1) │ (ref=1) │ (ref=1) │             │
│  │ System  │ Sys+    │ A:What  │ B:What  │ C:Exp.. │             │
│  │ Prompt  │ Prompt  │ is Py.. │ is Java │ lain..  │             │
│  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘             │
│       │         │         │         │         │                  │
│       └────┬────┘         │         │         │                  │
│            │              │         │         │                  │
│       ┌────┴────┐         │         │         │                  │
│       │         │         │         │         │                  │
│       ▼         ▼         ▼         ▼         ▼                  │
│    Seq A     Seq B      Diverge   Diverge   Diverge              │
│                                                                   │
│  BENEFIT: System prompt blocks (0-1) shared by 3 sequences       │
│           Only 2 unique blocks per sequence instead of 4          │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Scheduler & Continuous Batching

#### Scheduling Decisions

```python
class Scheduler:
    """
    Makes iteration-level batching decisions.
    
    Key insight: Instead of waiting for all requests in a batch to finish,
    vLLM can add/remove requests at every iteration.
    """
    
    def schedule(self) -> SchedulerOutput:
        # 1. First, try to run any currently RUNNING sequences
        running_scheduled = self._schedule_running()
        
        # 2. If GPU memory available, swap in SWAPPED sequences
        swapped_in = self._schedule_swapped(running_scheduled)
        
        # 3. If still have capacity, admit WAITING sequences
        new_admitted = self._schedule_waiting(running_scheduled + swapped_in)
        
        # 4. Return what to run this iteration
        return SchedulerOutput(
            scheduled_seq_groups=running_scheduled + swapped_in + new_admitted,
            num_batched_tokens=self._compute_num_tokens(...),
            blocks_to_swap_in=self._get_swap_in_blocks(),
            blocks_to_swap_out=self._get_swap_out_blocks(),
            blocks_to_copy=self._get_blocks_to_copy(),  # For beam search
            ignored_seq_groups=self._get_ignored(),
        )
```

#### Batching Constraints

```python
class SchedulerConfig:
    """
    Constraints that scheduler must respect:
    
    - max_num_seqs: Maximum sequences in a batch (GPU memory limit)
    - max_num_batched_tokens: Maximum tokens processed in one iteration
    - max_num_prefill_seqs: Limit concurrent prefill operations
    - preemption_mode: RECOMUTE (recompute) or SWAP (CPU offload)
    """
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 4096
    max_num_prefill_seqs: int = 1  # Usually 1 for chunked prefill
    preemption_mode: str = "RECOMPUTE"
```

#### Chunked Prefill

```python
def _schedule_chunked_prefill(self, waiting_queue):
    """
    Chunked prefill: Process large prompts in iterations
    
    Instead of processing entire prompt in one iteration (which blocks
    all other requests), break it into smaller chunks.
    
    Example: 4096 token prompt with chunk_size=512
    - Iteration 1: Process tokens 0-511, generate token 512
    - Iteration 2: Process tokens 512-1023, generate token 1024
    - ...until all prompt tokens processed
    """
    for seq_group in waiting_queue:
        num_new_tokens = seq_group.get_num_uncomputed_tokens()
        
        if num_new_tokens > self.scheduler_config.max_num_batched_tokens:
            # Chunk the prefill
            num_new_tokens = self.scheduler_config.max_num_batched_tokens
            
        # Only schedule if we have capacity
        if self._has_capacity(num_new_tokens):
            self._allocate_and_run(seq_group, num_new_tokens)
```

#### Preemption Strategies

```python
class Scheduler:
    def _preempt(self, seq_group: SequenceGroup) -> None:
        """
        Preempt a running sequence when memory is needed.
        
        Two strategies:
        1. RECOMPUTE: Drop KV cache, recompute from beginning later
           - Fast recovery but wastes compute
           - Good for short sequences
           
        2. SWAP: Move KV cache to CPU, swap back later
           - No recomputation but slower
           - Good for long sequences
        """
        if self.scheduler_config.preemption_mode == "RECOMPUTE":
            # Free blocks, mark as WAITING
            for seq in seq_group.get_seqs():
                self.block_manager.free(seq)
            seq_group.set_status(SequenceStatus.WAITING)
            
        else:  # SWAP
            # Move blocks to CPU
            for seq in seq_group.get_seqs():
                self._swap_out(seq)
            seq_group.set_status(SequenceStatus.SWAPPED)
```

### 7.3 Sampler Pipeline

#### Sampling Flow

```python
class Sampler(nn.Module):
    """
    Token generation pipeline:
    
    1. Logits → Temperature scaling
    2. Logits → Top-p/Top-k filtering
    3. Logits → Penalty application
    4. Softmax → Probability distribution
    5. Sample → Next token IDs
    """
    
    def forward(
        self,
        logits: torch.Tensor,           # [num_seqs, vocab_size]
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        
        # Step 1: Temperature scaling
        # logits = logits / temperature
        logits = self._apply_temperature(logits, sampling_metadata.temperature)
        
        # Step 2: Top-p/Top-k filtering
        # Set logits of filtered tokens to -inf
        logits = self._apply_top_p_top_k(logits, sampling_metadata)
        
        # Step 3: Penalties
        # Reduce logits for repeated tokens
        logits = self._apply_penalties(logits, sampling_metadata)
        
        # Step 4: Softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Step 5: Sample tokens
        token_ids = self._sample(probs, sampling_metadata)
        
        return SamplerOutput(
            samples=token_ids,
            probs=probs,
            logprobs=torch.log(probs),
        )
```

#### Top-p (Nucleus) Sampling

```python
def _apply_top_p(
    logits: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    """
    Nucleus sampling: Only consider tokens whose cumulative
    probability exceeds top_p (e.g., 0.9).
    
    Algorithm:
    1. Sort probs in descending order
    2. Compute cumulative sum
    3. Find cutoff where cumsum > top_p
    4. Set logits beyond cutoff to -inf
    """
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumsum > top_p
    sorted_indices_to_remove = cumsum > top_p.unsqueeze(-1)
    
    # Scatter back to original order
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    
    logits[indices_to_remove] = -float('inf')
    return logits
```

#### Speculative Decoding Integration

```python
class SpeculativeDecoder:
    """
    Speculative decoding uses a smaller draft model to predict
    multiple tokens, then verifies them with the main model.
    
    Speedup: Up to 2-3x for token generation phase
    """
    
    def generate(self, seq_group: SequenceGroup, num_speculative_tokens: int = 5):
        """
        1. Draft model generates 5 candidate tokens
        2. Main model verifies all 5 in parallel
        3. Accept tokens until first mismatch
        4. Resample from adjusted distribution if needed
        """
        # Get draft tokens
        draft_tokens = self.draft_model.generate(
            seq_group, 
            num_tokens=num_speculative_tokens
        )
        
        # Verify with target model (parallel evaluation)
        logits = self.target_model.forward_speculative(
            seq_group, 
            draft_tokens
        )
        
        # Accept/reject logic
        accepted_tokens = []
        for i, (draft_token, logit) in enumerate(zip(draft_tokens, logits)):
            prob_ratio = self._compute_acceptance_prob(logit, draft_token)
            
            if torch.rand() < prob_ratio:
                accepted_tokens.append(draft_token)
            else:
                # Rejection: resample and stop
                accepted_tokens.append(self._resample(logit))
                break
                
        return accepted_tokens
```

### 7.4 Prefix Caching

#### Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREFIX CACHING                                │
│                                                                   │
│  System Prompt: "You are a helpful assistant."                  │
│  User 1: "What is Python?"                                      │
│  User 2: "What is Java?"                                        │
│  User 3: "Explain functions."                                   │
│                                                                   │
│  Without Prefix Caching:                                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  User 1: Compute attention for "You are...What is Python?" │ │
│  │  User 2: Compute attention for "You are...What is Java?"   │ │
│  │  User 3: Compute attention for "You are...Explain..."      │ │
│  │  Total: 3 * 1000 tokens = 3000 token computations          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  With Prefix Caching:                                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Compute "You are a helpful assistant." once → CACHE        │ │
│  │  User 1: Use cache + compute "What is Python?"              │ │
│  │  User 2: Use cache + compute "What is Java?"                │ │
│  │  User 3: Use cache + compute "Explain functions."           │ │
│  │  Total: 100 + 3 * 100 = 400 token computations (7.5x faster)│ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

#### Radix Cache Implementation

```python
class RadixCache:
    """
    Radix tree-based prefix cache for KV cache blocks.
    
    Key features:
    - Exact prefix matching using token sequences
    - LRU eviction for memory pressure
    - Reference counting for block management
    """
    
    def __init__(self, block_manager, block_size: int):
        self.block_manager = block_manager
        self.block_size = block_size
        self.root = TreeNode()  # Radix tree root
        self.num_cached_blocks = 0
        
    def match_prefix(self, seq: Sequence) -> Tuple[int, List[KVCacheBlock]]:
        """
        Find longest matching prefix in cache.
        
        Returns:
            matched_len: Number of tokens matched
            blocks: List of cached blocks to reuse
        """
        token_ids = seq.get_token_ids()
        node = self.root
        matched_blocks = []
        matched_len = 0
        
        for token_id in token_ids:
            if token_id not in node.children:
                break
            node = node.children[token_id]
            matched_blocks.extend(node.blocks)
            matched_len += len(node.blocks) * self.block_size
            
        return matched_len, matched_blocks
        
    def insert(self, seq: Sequence) -> None:
        """
        Insert a completed sequence's KV cache into prefix cache.
        """
        token_ids = seq.get_token_ids()
        node = self.root
        
        for token_id in token_ids:
            if token_id not in node.children:
                # Create new node
                node.children[token_id] = TreeNode()
                
            node = node.children[token_id]
            
        # Store block references at leaf
        node.blocks = seq.block_table.blocks.copy()
        
        # Increment reference counts
        for block in node.blocks:
            block.incr_ref()
            
    def evict_lru(self, num_blocks: int) -> None:
        """
        Evict least recently used cached blocks.
        Called when memory pressure requires freeing blocks.
        """
        # Traverse tree and find LRU nodes
        lru_nodes = self._find_lru_nodes()
        
        for node in lru_nodes:
            if num_blocks <= 0:
                break
            for block in node.blocks:
                if block.decr_ref():
                    self.block_manager.free_block(block)
                    num_blocks -= 1
```

#### Cache Lookup in Scheduling

```python
class BlockManagerV2:
    """
    Block manager with prefix caching integration.
    """
    
    def allocate(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        for seq in seq_group.get_seqs():
            # 1. Try prefix cache lookup
            matched_len, cached_blocks = self.prefix_cache.match_prefix(seq)
            
            if matched_len > 0:
                # Reuse cached blocks
                for block in cached_blocks:
                    block.incr_ref()
                seq.block_table.blocks.extend(cached_blocks)
                
            # 2. Allocate remaining blocks
            remaining_tokens = len(seq.get_token_ids()) - matched_len
            num_new_blocks = remaining_tokens // self.block_size + 1
            
            for _ in range(num_new_blocks):
                block = self._allocate_block()
                seq.block_table.append(block)
                
            # 3. Mark matched portion as computed (skip attention)
            if matched_len > 0:
                seq.mark_tokens_computed(matched_len)
```

---

## 8. File-to-Component Mapping

### Layer 1: API Layer

| File | Component | Purpose |
|------|-----------|---------|
| `vllm/entrypoints/llm.py` | LLM | High-level synchronous API |
| `vllm/entrypoints/openai/api_server.py` | OpenAI-compatible API | HTTP server |
| `vllm/sampling_params.py` | SamplingParams | Generation configuration |
| `vllm/inputs.py` | Inputs | Input preprocessing |

### Layer 2: Engine Layer

| File | Component | Purpose |
|------|-----------|---------|
| `vllm/engine/llm_engine.py` | LLMEngine | Main orchestration |
| `vllm/engine/async_llm_engine.py` | AsyncLLMEngine | Async variant |
| `vllm/engine/arg_utils.py` | EngineArgs | Argument parsing |
| `vllm/engine/output_processor/multi_step.py` | Multi-Step Output Processor | Multi-step scheduling |

### Layer 3: Core Layer

| File | Component | Purpose |
|------|-----------|---------|
| `vllm/core/scheduler.py` | Scheduler | Iteration-level batching |
| `vllm/core/policy.py` | SchedulingPolicy | Batching strategies |
| `vllm/core/block_manager_v2.py` | BlockManagerV2 | KV cache allocation |
| `vllm/core/block_allocator.py` | BlockAllocator | Block memory management |
| `vllm/core/block_prefix_caching.py` | PrefixCacher | Prefix caching logic |
| `vllm/sequences.py` | Sequence/SequenceGroup | Sequence data structures |

### Layer 4: Model Execution Layer

| File | Component | Purpose |
|------|-----------|---------|
| `vllm/worker/model_runner.py` | ModelRunner | GPU execution |
| `vllm/worker/worker.py` | Worker | Worker process |
| `vllm/worker/cache_engine.py` | CacheEngine | KV cache management |
| `vllm/model_executor/models/` | Model Definitions | Various model implementations |
| `vllm/model_executor/layers/sampler.py` | Sampler | Token generation |
| `vllm/model_executor/layers/logits_processor.py` | LogitsProcessor | Logit processing |

### Layer 5: Attention Layer

| File | Component | Purpose |
|------|-----------|---------|
| `vllm/attention/layer.py` | Attention | Attention wrapper |
| `vllm/attention/ops/paged_attn.py` | PagedAttention | Core paged attention op |
| `vllm/attention/backends/flash_attn.py` | FlashAttentionBackend | Flash attention |
| `vllm/attention/backends/xformers.py` | XFormersBackend | xFormers attention |
| `vllm/attention/backends/flashinfer.py` | FlashInferBackend | FlashInfer attention |

### Layer 6: Utilities

| File | Component | Purpose |
|------|-----------|---------|
| `vllm/config.py` | Config Classes | Configuration dataclasses |
| `vllm/logger.py` | init_logger | Logging utilities |
| `vllm/utils.py` | Utilities | Helper functions |
| `vllm/envs.py` | Environment Variables | Env var management |

---

## 9. Common Tasks for AI Agents

### Task 1: Add a New Model

```python
# 1. Create model file: vllm/model_executor/models/my_model.py

from vllm.model_executor.models import ModelRegistry
from vllm.attention import Attention

class MyModelForCausalLM(nn.Module):
    def __init__(self, config, cache_config, quant_config):
        super().__init__()
        self.config = config
        # Initialize layers...
        
    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        # Implement forward pass
        # Use PagedAttention for efficient KV cache
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, kv_caches[i], attn_metadata)
        logits = self.lm_head(hidden_states)
        return logits

# 2. Register in vllm/model_executor/models/__init__.py
_MODEL_REGISTRY = {
    "MyModelForCausalLM": MyModelForCausalLM,
    # ...
}
```

### Task 2: Add Custom Sampling Parameter

```python
# 1. Modify vllm/sampling_params.py

@dataclass
class SamplingParams:
    # Existing params...
    temperature: float = 1.0
    top_p: float = 1.0
    
    # New parameter
    my_custom_param: float = 0.0
    
    def __post_init__(self):
        self._verify_args()
        
    def _verify_args(self) -> None:
        # Validate new parameter
        if not 0.0 <= self.my_custom_param <= 1.0:
            raise ValueError("my_custom_param must be in [0, 1]")

# 2. Use in sampler
class Sampler(nn.Module):
    def forward(self, logits, sampling_metadata):
        # Access custom param
        custom_params = sampling_metadata.my_custom_params
        # Apply to logits...
```

### Task 3: Add a New Scheduler Policy

```python
# 1. Create policy in vllm/core/policy.py

class ShortestJobFirst(SchedulingPolicy):
    """Prioritize shorter sequences"""
    
    def get_priority(self, now: float, seq_group: SequenceGroup) -> float:
        # Lower estimated time = higher priority
        num_tokens = seq_group.get_num_uncomputed_tokens()
        return -num_tokens  # Negative for min-heap

# 2. Register in policy.py
Policy = Enum("Policy", ["FCFS", "PRIORITY", "SJF"])

# 3. Update _policy_factory
_policy_factory = {
    Policy.FCFS: FCFS(),
    Policy.PRIORITY: Priority(),
    Policy.SJF: ShortestJobFirst(),
}
```

### Task 4: Debug Memory Issues

```python
# Add to your debugging code
from vllm.core.block_manager_v2 import BlockManagerV2

def debug_memory_usage(engine: LLMEngine):
    block_manager = engine.scheduler.block_manager
    
    # Get statistics
    num_free_blocks = len(block_manager.block_allocator.free_blocks)
    num_total_blocks = block_manager.block_allocator.num_blocks
    
    print(f"GPU KV Cache Usage:")
    print(f"  Total blocks: {num_total_blocks}")
    print(f"  Free blocks: {num_free_blocks}")
    print(f"  Used blocks: {num_total_blocks - num_free_blocks}")
    print(f"  Utilization: {(num_total_blocks - num_free_blocks) / num_total_blocks:.1%}")
    
    # Per-sequence breakdown
    for seq_group in engine.scheduler.running:
        for seq in seq_group.get_seqs():
            num_blocks = len(seq.block_table)
            num_tokens = len(seq.get_token_ids())
            print(f"  Seq {seq.seq_id}: {num_blocks} blocks, {num_tokens} tokens")
```

### Task 5: Add Metrics/Monitoring

```python
# 1. Create metrics collector
class EngineMetrics:
    def __init__(self):
        self.requests_total = 0
        self.tokens_generated = 0
        self.scheduling_time_ms = []
        
    def record_request(self, num_tokens: int, generation_time: float):
        self.requests_total += 1
        self.tokens_generated += num_tokens
        
    def record_scheduling(self, duration_ms: float):
        self.scheduling_time_ms.append(duration_ms)

# 2. Integrate into LLMEngine
class LLMEngine:
    def __init__(self, ...):
        self.metrics = EngineMetrics()
        
    def step(self):
        start_time = time.time()
        scheduler_output = self.scheduler.schedule()
        schedule_time = (time.time() - start_time) * 1000
        self.metrics.record_scheduling(schedule_time)
        # ... rest of step
```

### Task 6: Extend for Multi-Modal

```python
# 1. Create multi-modal input processor
from vllm.inputs import InputMetadata

class MultiModalInputProcessor:
    def process(
        self, 
        prompt: str, 
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None
    ) -> InputMetadata:
        """Process text + visual inputs"""
        text_tokens = self.tokenizer.encode(prompt)
        
        if images:
            image_features = self.vision_encoder.encode(images)
            # Interleave text and image tokens
            tokens = self.interleave_tokens(text_tokens, image_features)
        else:
            tokens = text_tokens
            
        return InputMetadata(
            prompt_token_ids=tokens,
            multi_modal_data={"images": images},
        )

# 2. Update model to handle multi-modal
class MultiModalLlama(nn.Module):
    def forward(self, input_ids, positions, kv_caches, attn_metadata, 
                multi_modal_data=None):
        if multi_modal_data:
            # Process visual features
            visual_embeds = self.vision_encoder(multi_modal_data["images"])
            # Merge with text embeddings
            inputs_embeds = self.merge_embeddings(input_ids, visual_embeds)
        else:
            inputs_embeds = self.embed_tokens(input_ids)
            
        # Continue with transformer...
```

---

## 10. Glossary

| Term | Definition |
|------|------------|
| **Attention** | Mechanism allowing model to focus on relevant tokens when generating output |
| **Beam Search** | Decoding strategy maintaining multiple candidate sequences and selecting best |
| **Block** | Fixed-size unit (typically 16 tokens) for KV cache storage in PagedAttention |
| **BlockTable** | Mapping from logical token positions to physical KV cache blocks |
| **Continuous Batching** | Dynamic adding/removing of requests at each iteration vs. static batching |
| **Decode Phase** | Generating one token at a time after prompt processing |
| **KV Cache** | Key-Value tensors stored to avoid recomputing attention for prior tokens |
| **Iteration-level Scheduling** | Making batching decisions at each model forward pass |
| **PagedAttention** | vLLM's core innovation - managing KV cache in fixed-size blocks |
| **Prefix Caching** | Storing and reusing KV cache for common prompt prefixes |
| **Preemption** | Pausing a running sequence to free resources for others |
| **Prefill Phase** | Initial processing of prompt tokens to build KV cache |
| **Prompt** | Input text provided to the model |
| **Request** | Complete user query including prompt and generation parameters |
| **Sampler** | Component that selects next token from probability distribution |
| **Scheduler** | Component deciding which sequences run in each iteration |
| **Sequence** | Single stream of tokens (one generation path) |
| **SequenceGroup** | Group of sequences from same request (for beam search/n>1) |
| **Speculative Decoding** | Using draft model to predict tokens for parallel verification |
| **SWAP** | Moving KV cache blocks between GPU and CPU memory |
| **Temperature** | Sampling parameter controlling randomness (0=deterministic, >1=random) |
| **Token** | Unit of text (word piece or subword) processed by model |
| **Top-k** | Sampling parameter limiting consideration to k most likely tokens |
| **Top-p (Nucleus)** | Sampling parameter limiting to tokens comprising p probability mass |
| **vLLM** | Vector LLM - the serving system described in this document |

---

## 11. References

### Academic Papers

1. **PagedAttention** (vLLM Core)
   - Paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
   - Authors: Kwon et al., SOSP 2023
   - Link: https://arxiv.org/abs/2309.06180

2. **FlashAttention**
   - Paper: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
   - Authors: Dao et al., NeurIPS 2022
   - Link: https://arxiv.org/abs/2205.14135

3. **FlashAttention-2**
   - Paper: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
   - Authors: Dao, ICML 2023
   - Link: https://arxiv.org/abs/2307.08691

4. **Continuous Batching**
   - Paper: "Orca: A Distributed Serving System for Transformer-Based Generative Models"
   - Authors: Yu et al., OSDI 2022
   - Link: https://www.usenix.org/conference/osdi22/presentation/yu

5. **Speculative Decoding**
   - Paper: "Fast Inference from Transformers via Speculative Decoding"
   - Authors: Leviathan et al., ICML 2023
   - Link: https://arxiv.org/abs/2211.17192

6. **vLLM Prefix Caching**
   - Paper: "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion"
   - Authors: Liu et al., OSDI 2024
   - Link: https://arxiv.org/abs/2405.16444

### Documentation & Resources

7. **vLLM Official Documentation**
   - URL: https://docs.vllm.ai/
   - Contains: API docs, performance tuning, deployment guides

8. **vLLM GitHub Repository**
   - URL: https://github.com/vllm-project/vllm
   - Contains: Source code, examples, issues, PRs

9. **vLLM Design Documents**
   - Location: `docs/source/` in repository
   - Topics: Architecture, quantization, distributed serving

10. **LLM Serving Survey**
    - Paper: "Efficient Large Language Models: A Survey"
    - Authors: Miao et al., 2023
    - Link: https://arxiv.org/abs/2312.03863

### Related Projects

11. **Text Generation Inference (TGI)** - HuggingFace
    - URL: https://github.com/huggingface/text-generation-inference

12. **TensorRT-LLM** - NVIDIA
    - URL: https://github.com/NVIDIA/TensorRT-LLM

13. **DeepSpeed-Inference** - Microsoft
    - URL: https://github.com/microsoft/DeepSpeed

14. **FasterTransformer** - NVIDIA
    - URL: https://github.com/NVIDIA/FasterTransformer

---

## Appendix: Quick Reference

### Common Code Patterns

```python
# Initialize engine
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

# Generate with custom params
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
)

outputs = llm.generate(["Hello, world!"], sampling_params)

# Async engine
from vllm import AsyncLLMEngine

engine = AsyncLLMEngine.from_engine_args(engine_args)
async for output in engine.generate(prompt, sampling_params, request_id):
    print(output.outputs[0].text)
```

### Environment Variables

```bash
# Key environment variables for vLLM
export VLLM_ATTENTION_BACKEND=flash_attn  # or xformers, flashinfer
export VLLM_WORKER_MULTIPROC_METHOD=spawn  # spawn/fork
export CUDA_VISIBLE_DEVICES=0,1,2,3        # GPU selection
export VLLM_LOGGING_LEVEL=INFO             # DEBUG, INFO, WARNING
```

### Debugging Commands

```bash
# Profile memory usage
python -m vllm.entrypoints.openai.api_server --model model_name --profile

# Check GPU utilization
nvidia-smi dmon -s mu

# Enable detailed logging
VLLM_LOGGING_LEVEL=DEBUG python your_script.py

# Memory profiling
python -m memory_profiler your_script.py
```

---

*Document generated for vLLM architecture understanding and AI agent assistance.*
*Last updated: 2024*
