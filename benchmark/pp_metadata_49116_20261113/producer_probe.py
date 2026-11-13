"""Exercise actual logprob producer layouts and preserved routed-expert output."""

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import (
    LogprobStage,
    get_token_ids_logprobs_raw,
    get_top_logprobs_raw,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.state_capturer.routed_experts import PP_ROUTED_EXPERTS_KEY

ROOT = Path(__file__).resolve().parent


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and b.device.type == 'cpu'
        assert a.dtype == b.dtype and a.shape == b.shape
        torch.testing.assert_close(a, b)
        assert b.untyped_storage().nbytes() == b.numel() * b.element_size()
    elif isinstance(a, list):
        assert type(b) is list and len(a) == len(b)
        assert a is not b
        for left, right in zip(a, b):
            same(left, right)
    else:
        assert a == b


cases = []
for stage, kind, no_copy in itertools.product((LogprobStage.PREFILL, LogprobStage.DECODE), ('top', 'token_ids'), (False, True)):
    lens = [2, 3] if stage == LogprobStage.PREFILL else None
    rows = 5 if lens else 2
    logits = torch.arange(rows * 8, dtype=torch.float32).view(rows, 8)
    kwargs = dict(stage=stage, extend_logprob_pruned_lens_cpu=lens, no_copy_to_cpu=no_copy)
    if kind == 'top':
        values, indices = get_top_logprobs_raw(logits, [2, 3], **kwargs)
    else:
        values, indices = get_token_ids_logprobs_raw(logits, [[1, 3], None], **kwargs)
    prefix = 'input' if lens else 'next_token'
    field = prefix + '_' + ('top' if kind == 'top' else 'token_ids') + '_logprobs_'
    output = LogitsProcessorOutput(next_token_logits=None, **{field + 'val': values, field + 'idx': indices})
    result = GenerationBatchResult(next_token_ids=torch.tensor([1, 2]), logits_output=output)
    wire = SchedulerPPMixin()._pp_prepare_tensor_dict(result, SimpleNamespace(return_logprob=True))
    same(values, wire[field + 'val'])
    same(indices, wire[field + 'idx'])
    cases.append({'stage': stage.name, 'kind': kind, 'no_copy_to_cpu': no_copy, 'passed': True})

ids = torch.tensor([7])
routes = torch.arange(8).view(1, 2, 4)
result = GenerationBatchResult(next_token_ids=ids, logits_output=LogitsProcessorOutput(next_token_logits=None), routed_experts_output=SimpleNamespace(topk=routes))
wire = SchedulerPPMixin()._pp_prepare_tensor_dict(result, SimpleNamespace(return_logprob=False))
assert wire[PP_ROUTED_EXPERTS_KEY] is routes and wire['next_token_ids'] is ids
record = {'producer_layouts': cases, 'routed_experts_preserved_without_copy': True, 'cases_passed': len(cases) + 1}
(ROOT / 'results/producer-layouts.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
