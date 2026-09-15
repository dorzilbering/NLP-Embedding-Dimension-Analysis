import json
import numpy as np
import pytest
from pilot.core import cached_embeddings, checked, fit_pca, normalize, project, score_pairs, select_data
from run_pilot import save_outputs


def test_shapes_and_finite_values():
    with pytest.raises(ValueError):
        checked(np.zeros((2, 3)), dim=3072)
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError):
            checked([[bad, 1]])
    with pytest.raises(ValueError):
        normalize([[0, 0]])


def test_pca_reuses_basis_without_refitting():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(64, 16))
    heldout = rng.normal(size=(7, 16))
    pca = fit_pca(train, max_dim=8)
    mean, components = pca.mean_.copy(), pca.components_.copy()
    for dim in (8, 4):
        result = project(heldout, pca, dim)
        assert result.shape == (7, dim)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(pca.mean_, mean)
    np.testing.assert_array_equal(pca.components_, components)
    with pytest.raises(ValueError):
        project(heldout, pca, 12)


def test_leakage_excludes_even_unselected_heldout_texts():
    train = [{"sentence1": f"Training {i}", "sentence2": f"Other {i}"} for i in range(450)]
    train.append({"sentence1": "  TEST text  ", "sentence2": "VALIDATION TEXT"})
    dataset = {"train": train,
               "validation": [{"sentence1": "validation text", "sentence2": "a", "score": 1},
                              {"sentence1": "b", "sentence2": "c", "score": 2}],
               "test": [{"sentence1": "test text", "sentence2": "d", "score": 3}]}
    selected, _, ids = select_data(dataset, 800, 2, 42)
    assert len(selected) == 800 and ids == [0, 1]
    assert all("TEST" not in t and "VALIDATION" not in t for t in selected)
    assert select_data(dataset, 800, 2, 42)[0] == selected


def test_known_spearman_and_undefined_score():
    left = np.array([[1, 0]] * 4)
    right = np.array([[-1, 0], [0, 1], [0.5, np.sqrt(0.75)], [1, 0]])
    score, _ = score_pairs(left, right, [0, 1, 2, 3])
    assert score == pytest.approx(1)
    with pytest.raises(ValueError):
        score_pairs(left, left, [0, 1, 2, 3])


def test_cache_reuse_and_identity(tmp_path):
    calls = []
    def encode(texts):
        calls.append(1)
        return np.ones((len(texts), 3072))
    first, hit = cached_embeddings(tmp_path, {"revision": "a"}, ["text"], encode)
    assert not hit
    second, hit = cached_embeddings(tmp_path, {"revision": "a"}, ["text"], encode)
    assert hit and len(calls) == 1
    np.testing.assert_array_equal(first, second)
    cached_embeddings(tmp_path, {"revision": "b"}, ["text"], encode)
    assert len(calls) == 2


def test_output_serialization_and_plot(tmp_path):
    # Synthetic unit-test fixtures only. Never written as project experiment results.
    rows = [{"dimension": d, "cosine_spearman": 0.1} for d in (3072, 768, 384)]
    save_outputs(tmp_path, rows, {"synthetic_unit_test": True})
    assert json.loads((tmp_path / "results.json").read_text())["scores"] == rows
    assert (tmp_path / "scores.csv").exists()
    assert (tmp_path / "dimension_vs_spearman.png").read_bytes().startswith(b"\x89PNG")


def test_pooling_both_padding_directions():
    torch = pytest.importorskip("torch")
    from pilot.model import last_valid_pool
    hidden = torch.arange(24).reshape(2, 4, 3)
    right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    left = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    assert torch.equal(last_valid_pool(hidden, right), torch.stack([hidden[0, 1], hidden[1, 3]]))
    assert torch.equal(last_valid_pool(hidden, left), hidden[:, 3])
    with pytest.raises(ValueError):
        last_valid_pool(hidden, torch.zeros_like(right))


def test_tiny_phi_final_state_and_padding():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from pilot.model import last_valid_pool
    # Random tiny architecture for API/masking verification only; no downloads.
    torch.manual_seed(42)
    config = transformers.Phi3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                    num_hidden_layers=2, num_attention_heads=4,
                                    num_key_value_heads=2, max_position_embeddings=128,
                                    original_max_position_embeddings=128, pad_token_id=0)
    model = transformers.AutoModel.from_config(config).eval()
    singles = []
    with torch.inference_mode():
        for ids in ([1, 2], [3, 4, 5, 6]):
            tokens = torch.tensor([ids])
            output = model(tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
            singles.append(output.last_hidden_state[0, -1])
        expected = torch.stack(singles)
        for ids in ([[1, 2, 0, 0], [3, 4, 5, 6]], [[0, 0, 1, 2], [3, 4, 5, 6]]):
            tokens = torch.tensor(ids)
            mask = (tokens != 0).long()
            output = model(tokens, attention_mask=mask, use_cache=False)
            pooled = last_valid_pool(output.last_hidden_state, mask)
            torch.testing.assert_close(pooled, expected, atol=1e-5, rtol=1e-4)


def test_gpu_guard_prevents_cpu_execution(monkeypatch):
    torch = pytest.importorskip("torch")
    from pilot.model import check_gpu
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU model execution is intentionally disabled"):
        check_gpu()
