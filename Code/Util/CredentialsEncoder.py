import json
import base64

# Seu JSON original
def convertJsonToBase64(json_data):
    # Converter o dicionário JSON em uma string JSON
    json_string = json.dumps(json_data)

    # Codificar a string JSON em Base64
    base64_encoded = base64.b64encode(json_string.encode('utf-8')).decode('utf-8')

    return base64_encoded

def convertBase64ToJson(base64_string):
    # Decodificar a string Base64 de volta para JSON
    json_string = base64.b64decode(base64_string).decode('utf-8')
    json_data = json.loads(json_string)

    return json_data

if __name__ == "__main__":
    # Exemplo de uso. Insira seu JSON aqui
    json_data = {
    "type": "service_account",
    "project_id": "",
    "private_key_id": "",
    "private_key": "",
    "client_email": "",
    "client_id": "",
    "auth_uri": "",
    "token_uri": "",
    "auth_provider_x509_cert_url": "",
    "client_x509_cert_url": "",
    "universe_domain": ""
  }
    base64_string = convertJsonToBase64(json_data)
    #Esse valor deve ser colocado no arquivo .env no GOOGLE_CLOUD_CREDENTIALS_IARA
    print("String Base64:", base64_string)

