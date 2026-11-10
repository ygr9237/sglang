"""PP metadata must be ready for CPU serialization before switching streams."""

import pickle
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.distributed.parallel_state import _split_tensor_dict
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestPPOutputMetadata(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_mixed_numeric_and_strided_cuda_metadata(self):
        from sglang.srt.managers.scheduler_pp_mixin import _pp_freeze_output_metadata

        forward = torch.cuda.Stream()
        comm = torch.cuda.Stream()
        with torch.cuda.stream(forward):
            base = torch.arange(100000, dtype=torch.float32, device="cuda")
            view = base.view(10000, 10)[5:7, :2]
            frozen = _pp_freeze_output_metadata([7, view])
            base.zero_()
        with torch.cuda.stream(comm):
            self.assertEqual(frozen[0], 7)
            self.assertEqual(frozen[1].device.type, "cpu")
            self.assertEqual(frozen[1].shape, (2, 2))
            self.assertEqual(frozen[1].untyped_storage().nbytes(), 16)
            restored = pickle.loads(pickle.dumps(frozen))
            self.assertEqual(restored[1].tolist(), [[50, 51], [60, 61]])
        forward.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_nested_cuda_views_are_ready_before_communication(self):
        forward = torch.cuda.Stream()
        comm = torch.cuda.Stream()
        with torch.cuda.stream(forward):
            ids = torch.tensor([7], device="cuda")
            values = torch.arange(1000, device="cuda", dtype=torch.float32)
            result = GenerationBatchResult(
                next_token_ids=ids,
                logits_output=LogitsProcessorOutput(
                    next_token_logits=None,
                    next_token_token_ids_logprobs_val=[values[500:502]],
                ),
            )
            wire = SchedulerPPMixin()._pp_prepare_tensor_dict(
                result, SimpleNamespace(return_logprob=True)
            )
        with torch.cuda.stream(comm):
            metadata, tensors = _split_tensor_dict(wire)
            self.assertIs(tensors[0], ids)
            frozen = wire["next_token_token_ids_logprobs_val"][0]
            self.assertEqual(frozen.device.type, "cpu")
            self.assertEqual(frozen.untyped_storage().nbytes(), 8)
            payload = pickle.dumps(metadata)
            restored = dict(pickle.loads(payload))  # Locally generated test metadata.
            self.assertEqual(
                restored["next_token_token_ids_logprobs_val"][0].tolist(), [500, 501]
            )


if __name__ == "__main__":
    unittest.main()
