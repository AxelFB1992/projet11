import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAGENDA_API_KEY")
#On a décidé de remplacer cette recherche spécifique sur un agenda seulement...
#AGENDA_UID = 16676449
#BASE_URL = f"https://api.openagenda.com/v2/agendas/{AGENDA_UID}/events"

#... Par une recherche sur plusieurs sources
SOURCES = [
    {"uid": 16676449, "nom": "Agenda Région Pays de la Loire", "admin_level2": None},
    #{"uid": 28658827, "nom": "Le Kiosque Mayenne", "admin_level2": None},
    #{"uid": 56526489, "nom": "Les nuits de la Mayenne", "admin_level2": None},
    {"uid": 14115607, "nom": "Unidivers Oui sortir", "admin_level2": None},
]


def fetch_events_from_agenda(agenda_uid, admin_level2=None):
    """Récupère les événements à venir d'un agenda."""
    base_url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    headers = {"key": API_KEY}
    params = {
        "relative[]": "upcoming",
        "detailed": 1,
        "monolingual": "fr",
        "size": 100,
        "sort": "timingsWithFeatured.asc",
    }
    #Si le département est renseigné dans l'appel de fonction, on le passe dans les paramètre de la requête http
    if admin_level2:
        params["adminLevel2[]"] = admin_level2

    #On stocke tous les évenements là-dedans
    all_events = []

    #Dans la mesure où l'on limite le résultat de la recherche à 100 résultat, il faut prévoir le cas où il y en a plus que 100
    #Dans ce cas, il faut prévoir un curseur qui indique où l'on se situe dans la recherche : c'est le but de la variable after
    after = None
    while True:
        #Si il y a des résultats après les 100 premiers stockés, alors on initialise la variable after que l'on renseigne
        #dans les paramèètre de la requête pour dire que l'on veut récuperer directement les résultats après ce curseur
        if after:
            params["after[]"] = after
        else:
            params.pop("after[]", None)

        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        #Cela permet de parser le résultat de la requête en format json, comme on l'avait déjà fait dans explorer_agenda.py
        data = response.json()
        all_events.extend(data.get("events", []))

        #Cette instruction permet de récuperer l'endroit où l'on s'est arrêté et de reprendre la recherche à partir de là si besoin
        after = data.get("after")
        if not after:
            break

    return all_events

if __name__ == "__main__":

    #La liste qui va nous permettent de stockker tous les evenements de tous les agendas
    all_events = []
    #On crée une collection (un set) pour stocker les uids que l'on a déjà vu 
    #Cela permet de ne pas récuperer deux fois le même uids (par exemple le même évènement dans deux agendas)
    seen_uids = set()

    #On parcourt toutes les sources, une par une. 
    for source in SOURCES:
        #C'est ici que l'on utilise la fonctio définis précedemment qui nous permet de récuperer les évenements d'un agenda
        #Qui se déroule dans un départmeent si le paramètre admin_level2 est défini
        events = fetch_events_from_agenda(source["uid"], source["admin_level2"])
        #On va compter les évenements qui sont des nouveaux évenements (hors ceux déjà vu dans seen uids)
        nouveaux = 0
        for e in events:
            #On parcourt l'evenement et on le stocke seulement si il n'a pas déjà été vu
            if e["uid"] not in seen_uids:
                e["_source_agenda"] = source["nom"]  # traçabilité pour le rapport
                all_events.append(e)
                seen_uids.add(e["uid"])
                nouveaux += 1
        #Pour chaque source, on fait un petit récapitulatif pour dire combien d'évenement on été récupérés et stockés
        print(f"{source['nom']} : {len(events)} récupérés, {nouveaux} nouveaux (hors doublons)")

    #Une fois que toutes les sources ont été parcourus et tous les évenements stockés dans all_events, on les stocke dans un fichiers
    #Ce fichier stockera chaque évenement sous forme de ligne, et chaque ligne sera une balise json
    output_path = "data/raw_events.json"
    #Si le dossier existe déjà...
    os.makedirs("data", exist_ok=True)
    #On ouvre ou on crée le fichier
    with open(output_path, "w", encoding="utf-8") as f:
        #Et on copie tous les évenements dedans sous le format json : 1 évenement --> 1 document json
        json.dump(all_events, f, ensure_ascii=False, indent=2)

    print(f"\nTotal final : {len(all_events)} événements uniques sauvegardés dans {output_path}")
