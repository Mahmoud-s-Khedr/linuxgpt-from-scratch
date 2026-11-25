import unittest
import torch
from models import build_model
from sanity_check import check_kv_cache_parity, check_overfit_and_sft_mask


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_training_mode_survives_generation(self):
        for arch in ['gpt2', 'llama']:
            with self.subTest(arch=arch):
                model = build_model(arch, preset='tiny', vocab_size=512, dropout=0.05)
                x = torch.tensor([[10, 20, 30]])
                model.train()
                result = model.generate(x, max_new_tokens=2, temperature=0)
                self.assertTrue(model.training)
                self.assertTrue(all(m.training for m in model.modules()))
                self.assertEqual(result.shape, (1, 5))
                model.eval()
                model.generate(x, max_new_tokens=2, temperature=0)
                self.assertFalse(model.training)

    def test_cached_logits_match_and_masked_loss_can_learn(self):
        for arch in ['gpt2', 'llama']:
            with self.subTest(arch=arch):
                check_kv_cache_parity(arch)
                check_overfit_and_sft_mask(arch)


if __name__ == '__main__':
    unittest.main()
