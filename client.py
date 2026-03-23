import requests

URL = "http://localhost:8000"
endpoint = "/predict"

url_with_endpoint = f"{URL}{endpoint}"

def response_from_server(url, image_file, verbose=True):
    """Makes a POST request to the server and returns the response.

    Args:
        url (str): URL that the request is sent to.
        image_file (_io.BufferedReader): File to upload, should be an image.
        verbose (bool): True if the status of the response should be printed. False otherwise.

    Returns:
        requests.model.Response: Response from the server.
    """
    
    files = {'file': image_file}
    response = requests.post(url, files=files)
    status_code = response.status_code
    if verbose and status_code != 200:
        print("There was an error when handling the request. Status code: {status_code}")
    return response
