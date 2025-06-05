# pip install SPARQLWrapper requests pandas

from SPARQLWrapper import SPARQLWrapper, JSON
import requests
import time
import pandas as pd

# Étape 1 : Requête SPARQL pour articles liés à des sports ou événements sportifs
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setQuery("""
SELECT ?article WHERE {
  ?item wdt:P31 ?type.
  FILTER(?type IN (
    wd:Q349,          # sport
    wd:Q27020041,     # compétition sportive
    wd:Q847017,       # club sportif
    wd:Q988108,       # fédération sportive
    wd:Q2066131       # événement sportif
  ))
  ?article schema:about ?item;
           schema:isPartOf <https://fr.wikipedia.org/>;
           schema:dateModified ?modif.
  FILTER(?modif > "2020-06-01T00:00:00Z"^^xsd:dateTime)
}
LIMIT 30
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

# Extraire les titres
titles = [r['article']['value'].split('/')[-1] for r in results["results"]["bindings"]]

# Étape 2 : Télécharger le texte brut
def get_article_text(title):
    url = f"https://fr.wikipedia.org/api/rest_v1/page/plain/{title}"
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.text
    else:
        return f"Erreur {resp.status_code}"

# Étape 3 : Collecte
articles = {}
for title in titles:
    print(f"Téléchargement de : {title}")
    articles[title] = get_article_text(title)
    time.sleep(1)  # respecter les quotas

# Sauvegarde
df = pd.DataFrame(articles.items(), columns=["Titre", "Contenu"])
df.to_csv("articles_sport_only.csv", index=False, encoding='utf-8')
print("✅ Fichier 'articles_sport_only.csv' généré.")
