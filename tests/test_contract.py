import pytest
from starlette.testclient import TestClient

from fraud_api.app import app

EXAMPLE_PAYLOAD = {
    'id': 'tx-1329056812',
    'transaction': {
        'amount': 41.12,
        'installments': 2,
        'requested_at': '2026-03-11T18:45:53Z',
    },
    'customer': {
        'avg_amount': 82.24,
        'tx_count_24h': 3,
        'known_merchants': ['MERC-003', 'MERC-016'],
    },
    'merchant': {
        'id': 'MERC-016',
        'mcc': '5411',
        'avg_amount': 60.25,
    },
    'terminal': {
        'is_online': False,
        'card_present': True,
        'km_from_home': 29.2331036248,
    },
    'last_transaction': None,
}

EXAMPLE_PAYLOAD_WITH_LAST_TX = {
    **EXAMPLE_PAYLOAD,
    'last_transaction': {
        'timestamp': '2026-03-11T14:58:35Z',
        'km_from_current': 18.8626479774,
    },
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_ready(client):
    response = client.get('/ready')
    assert response.status_code == 200


def test_fraud_score_shape(client):
    response = client.post('/fraud-score', json=EXAMPLE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {'approved', 'fraud_score'}
    assert isinstance(body['approved'], bool)
    assert 0.0 <= body['fraud_score'] <= 1.0


def test_fraud_score_with_last_transaction(client):
    response = client.post('/fraud-score', json=EXAMPLE_PAYLOAD_WITH_LAST_TX)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {'approved', 'fraud_score'}
