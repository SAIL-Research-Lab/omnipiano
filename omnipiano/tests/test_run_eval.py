from omnipiano.integrations import policy_registry


def test_registered_policies_are_returned(monkeypatch):
    policy = object()
    monkeypatch.setattr(policy_registry, "_POLICIES", {})
    policy_registry.register_policy("test", policy)
    assert policy_registry.registered_policies() == (("test", policy),)
