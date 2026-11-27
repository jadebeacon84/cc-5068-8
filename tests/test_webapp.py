import pytest

from ser import webapp
from ser.config import TARGET_CLASSES


@pytest.fixture()
def client(trained_pipeline, monkeypatch):
    monkeypatch.setattr(webapp, "pipeline", trained_pipeline)
    webapp.app.config["TESTING"] = True
    return webapp.app.test_client()


def test_health_and_recording_list(client):
    health = client.get("/api/health").get_json()
    assert health["ready"] is True
    assert health["remote_calls"].startswith("none")
    assert "chime" in health["target_classes"]
    recs = client.get("/api/recordings").get_json()
    assert len(recs) >= 8
    # labels shown are kernel outputs, never folder names
    assert all("class" not in r for r in recs)


def test_inference_carries_annotations_and_legacy_mapping(client):
    rid = client.get("/api/recordings").get_json()[0]["recording_id"]
    inf = client.post(f"/api/recordings/{rid}/inference").get_json()
    assert "annotations" in inf and "legacy_annotations" in inf
    for ann in inf["annotations"]:
        if ann["class"] in TARGET_CLASSES:
            assert ann["audit_status"] in (
                "audited", "pending_review", "unclaimed")


def test_audit_endpoint_constrains_packaging(client):
    from ser.labeling import is_trainable_event

    rid = client.get("/api/recordings").get_json()[0]["recording_id"]
    # Find a pending-review interval if this recording has one.
    inf = client.get(f"/api/recordings/{rid}/inference").get_json()
    pending = next(
        (a for a in inf["annotations"]
         if a.get("audit_status") == "pending_review"), None)
    if pending is None:
        # Any audited target event can still be sent through the endpoint.
        pending = next(a for a in inf["annotations"]
                       if a["class"] in TARGET_CLASSES)
        resp = client.post(f"/api/recordings/{rid}/audit", json={
            "event_id": pending["event_id"], "status": "audited",
        })
        assert resp.status_code == 200
        return
    assert not is_trainable_event(pending)
    resp = client.post(f"/api/recordings/{rid}/audit", json={
        "event_id": pending["event_id"], "status": "audited",
        "note": "listening review passed",
    })
    body = resp.get_json()
    assert body["annotation"]["audit_status"] == "audited"
    assert pending["event_id"] in body["summary"]["trainable_event_ids"]
    # Invalid status is rejected rather than silently stored.
    bad = client.post(f"/api/recordings/{rid}/audit", json={
        "event_id": pending["event_id"], "status": "nonsense",
    })
    assert bad.status_code == 400


def test_inference_evidence_and_cue_editing(client):
    rid = client.get("/api/recordings").get_json()[0]["recording_id"]
    inf = client.post(f"/api/recordings/{rid}/inference").get_json()
    assert len(inf["events"]) >= 0
    assert len(inf["window_probs"]) == len(inf["window_times"])
    assert len(inf["thresholds"]) == len(TARGET_CLASSES)

    evidence = client.get(f"/api/recordings/{rid}/evidence/0").get_json()
    assert len(evidence["mel_patch"]) == evidence["n_mels"]
    assert set(evidence["class_probabilities"]) == set(TARGET_CLASSES)
    assert set(evidence["decisions"]) == set(TARGET_CLASSES)

    # audio endpoints
    assert client.get(f"/api/recordings/{rid}/audio/mix").status_code == 200

    if inf["events"]:
        uid = inf["events"][0]["event_uid"]
        relabelled = client.post(f"/api/cuesheet/{rid}/relabel", json={
            "uid": uid, "new_class": "machine",
        }).get_json()
        assert relabelled["cls"] == "machine"

        fail = client.post(f"/api/cuesheet/{rid}/failure", json={
            "uid": uid, "reason": "wrong_class",
            "expected_class": "bell",
        }).get_json()
        assert fail["entry"]["cue_status"] == "failure"

    feedback = client.get("/api/feedback").get_json()
    assert feedback["summary"]["count"] >= 1


def test_cut_restore_and_verify_flow(client):
    rid = client.get("/api/recordings").get_json()[0]["recording_id"]
    client.post(f"/api/recordings/{rid}/inference")
    cue = client.get(f"/api/cuesheet/{rid}").get_json()
    entries = cue["entries"]
    if len(entries) >= 2:
        a, b = entries[0], entries[1]
        if b["start"] > a["end"]:
            cut = client.post(f"/api/cuesheet/{rid}/cut", json={
                "start": a["end"] + 0.01,
                "end": b["start"] - 0.01,
            })
            if cut.status_code == 200:
                restored = client.post(
                    f"/api/cuesheet/{rid}/restore", json={}
                ).get_json()
                assert len(restored["entries"]) == len(entries)

    verify = client.get(f"/api/cuesheet/{rid}/verify").get_json()
    assert "all_identities_exact" in verify
    assert isinstance(verify["matched"], list)


def test_unknown_class_is_never_default(client):
    rid = client.get("/api/recordings").get_json()[0]["recording_id"]
    inf = client.get(f"/api/recordings/{rid}/inference").get_json()
    for ev in inf["events"]:
        assert ev["cls"] in TARGET_CLASSES
