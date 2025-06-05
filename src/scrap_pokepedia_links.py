import requests

api_url = "https://www.pokepedia.fr/api.php"
params = {
    "action": "query",
    "format": "json",
    "list": "allpages",
    "aplimit": "500"
}

def retrieve_links(api_url, params) -> list[str]:
    "Retrieves all the weblinks hosted by Poképedia"
    all_links = []
    continue_token = None

    while True:
        if continue_token:
            params["apcontinue"] = continue_token

        response = requests.get(api_url, params=params)
        data = response.json()

        pages = data.get("query", {}).get("allpages", [])
        for page in pages:
            title = page["title"].replace(" ", "_")
            link = f"https://www.pokepedia.fr/{title}"
            all_links.append(link)

        if "continue" in data:
            continue_token = data["continue"]["apcontinue"]
        else:
            break
    return all_links
