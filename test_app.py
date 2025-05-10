import pytest
from application import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200

def test_predict_route(client):
    data = {
        "feature1": "0.5",
        "feature2": "0.3",
        "feature3": "1.1",
        "feature4": "2.2",
        "feature5": "0.0",
        "feature6": "3.3",
        "feature7": "1.2",
        "feature8": "0.8",
        "feature9": "2.0",
        "feature10": "1.0",
        "model_choice": "linear"
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    assert b"Prediction:" in response.data or b"Error:" in response.data

