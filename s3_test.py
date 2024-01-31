import boto3
from moto import mock_s3
import pytest

# using this article https://www.sanjaysiddhanti.com/2020/04/08/s3testing/

# from recipe import Recipe, S3_BUCKET

s3_client = boto3.client('s3',
                         endpoint_url='http://192.168.0.109:9000',
                         aws_access_key_id='minioadmin',
                         aws_secret_access_key='minioadmin')

@pytest.fixture
def s3():
    """Pytest fixture that creates the recipes bucket in 
    the fake moto AWS account
    
    Yields a fake boto3 s3 client
    """
    with mock_s3():
        try:
            s3_client.create_bucket(Bucket="test")
            yield s3_client
        except: 
            yield s3_client


def test_create_and_get(s3):
    # Recipe(name="nachos", instructions="Melt cheese on chips").save()
    s3_client.put_object(Body='testing_minio', Bucket='test', Key="test.txt")
    # recipe = Recipe.get_by_name("nachos")
    # assert recipe.name == "nachos"
    # assert recipe.instructions == "Melt cheese on chips"
    response = s3_client.get_object(Bucket='test', Key="test.txt")
    test_txt = response["Body"].read().decode("utf-8")
    assert test_txt == 'testing_minio'