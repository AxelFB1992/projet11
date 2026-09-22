# Puls-Events — Système RAG pour la recommandation d'événements culturels

## Introduction

**Puls-Events** est une entreprise fictive du secteur événementiel/culturel, positionnée sur la valorisation et la recommandation d'événements culturels en France. Dans un environnement où l'offre culturelle est dispersée entre de multiples plateformes, agendas institutionnels et réseaux locaux, Puls-Events souhaite proposer à ses utilisateurs un moyen simple et naturel de découvrir les événements qui les intéressent, sans avoir à naviguer entre des dizaines de sources différentes.

Pour cela, l'entreprise souhaite à terme s'équiper d'un **chatbot intelligent**, capable de répondre en langage naturel à des questions comme *"Quels concerts de musique classique se jouent à Nantes ?"* ou *"Quels spectacles pour enfants sont prévus en Vendée ?"*, en s'appuyant sur des données d'événements réelles et à jour plutôt que sur les seules connaissances générales d'un modèle de langage.

La mission confiée dans le cadre de ce projet n'est pas de livrer ce chatbot en production, mais de réaliser un **Proof of Concept (POC)** : un système de type **RAG (Retrieval-Augmented Generation)** démontrant la faisabilité technique de l'approche, sur un périmètre volontairement restreint (la région Pays de la Loire), avant d'envisager un passage à l'échelle nationale.

---

## Partie 1 — Présentation de l'application et objectifs

Ce projet met en œuvre une chaîne complète permettant de :

1. **Collecter** des événements culturels réels via l'API [OpenAgenda](https://developers.openagenda.com/), à partir de plusieurs agendas couvrant la région Pays de la Loire.
2. **Nettoyer et structurer** ces données pour ne conserver que des événements culturels, géolocalisés et exploitables.
3. **Vectoriser** ces événements dans une base vectorielle **FAISS**, à l'aide des embeddings **Mistral** (`mistral-embed`).
4. **Interroger** cette base via un système RAG construit avec **LangChain**, combinant recherche par similarité (avec filtrage géographique automatique) et génération de réponse par un LLM (`mistral-small-latest`).
5. **Vérifier** la qualité des données à chaque étape via une suite de tests automatisés (`pytest`).

L'objectif du POC est de démontrer qu'il est possible, à partir de sources ouvertes et d'un stack technique léger, de répondre à des questions en langage naturel sur des événements culturels réels — avec une réponse ancrée dans les données récupérées, et non hallucinée par le modèle.

---

## Partie 2 — Architecture de l'application

### A. Description des dossiers et fichiers présents

```
projet11/
├── .env                          # Clés API (OPENAGENDA_API_KEY, MISTRAL_API_KEY) — non versionné
├── .gitignore
├── requirements.txt               # Dépendances Python (généré via pip freeze)
├── README.md
├── data/
│   ├── raw_events.json            # Événements bruts, toutes sources confondues
│   ├── events_clean.csv           # Événements nettoyés et filtrés (format tabulaire)
│   └── events_clean.json          # Événements nettoyés et filtrés (format JSON, avec embedding_text)
├── scripts/
│   ├── explorer_agendas.py        # Exploration manuelle des agendas OpenAgenda
│   ├── explore_filtered_search.py # Diagnostic du filtrage par métadonnées FAISS (fetch_k)
│   ├── fetch_events.py            # Récupération des événements bruts depuis OpenAgenda
│   ├── preprocess_events.py       # Nettoyage, structuration et filtrage des événements
│   ├── vectorize_events.py        # Construction de l'index vectoriel FAISS
│   ├── rag_chain.py               # Chaîne RAG complète (retrieval + génération)
│   └── chat_interactive.py        # Interface conversationnelle en ligne de commande
├── notebooks/
│   └── explore_agendas_regions.ipynb  # Exploration exploratoire des volumes d'agendas par région
├── tests/
│   └── test_data_quality.py       # Tests de qualité des données nettoyées
├── vectorstore/
│   └── faiss_index/                # Index FAISS sauvegardé (index.faiss + index.pkl) — régénérable
└── projet11-env/                  # Environnement virtuel Python — non versionné
```

**Environnement virtuel et dépendances**

Le projet s'exécute dans un environnement virtuel Python (`projet11-env`, créé via `python -m venv`), qui isole les dépendances du reste du système. Les bibliothèques principales sont :

| Bibliothèque | Rôle |
|---|---|
| `langchain` | Orchestration de la chaîne RAG (LCEL) |
| `langchain-community` | Intégration FAISS (`langchain_community.vectorstores.FAISS`) |
| `langchain-mistralai` | Intégration des modèles Mistral (`ChatMistralAI`, `MistralAIEmbeddings`) |
| `faiss-cpu` | Moteur de recherche vectorielle (index `IndexFlatL2`) |
| `mistralai` | Client officiel de l'API Mistral |
| `python-dotenv` | Chargement des clés API depuis `.env` |
| `pandas` | Manipulation et nettoyage des données tabulaires |
| `pytest` | Tests automatisés de qualité des données |
| `requests` | Appels à l'API OpenAgenda |

L'ensemble des dépendances, avec leurs versions exactes, est figé dans `requirements.txt` pour garantir la reproductibilité de l'environnement.

---

### B. Exploration des agendas dans OpenAgenda

#### 1. `explorer_agendas.py` — exploration initiale

Ce script sert à explorer manuellement le catalogue d'agendas disponibles sur OpenAgenda : recherche d'agendas par mot-clé, consultation du détail d'un agenda (nombre d'événements, description), comptage d'événements avec filtres (région, département, période), et inspection de la structure des champs (lieux, dates). Il a permis d'identifier les 4 premières sources retenues pour la région Pays de la Loire (l'agenda régional officiel, deux agendas mayennais, et un agrégateur "Grand Ouest").

#### 2. `explore_agendas_regions.ipynb` — exploration secondaire

Ce notebook a été produit dans un second temps, après le constat que le corpus initial (93 événements) semblait faible pour une étude et une présentation. Il explore deux pistes :

- **L'élargissement de la couverture Pays de la Loire** : recherche d'agendas culturels locaux non encore exploités (par ville, par institution).
- **La comparaison avec d'autres régions françaises**, pour situer le volume obtenu par rapport à ce qui existe ailleurs sur la plateforme.

**Solution retenue** : plutôt que de changer de région ou de renoncer à la granularité locale, la stratégie choisie a été d'**ajouter des agendas culturels locaux de la région Pays de la Loire** — spécifiquement des agendas institutionnels dédiés à la culture (un théâtre, un réseau de médiathèques), plutôt qu'un agenda administratif généraliste (écarté après vérification manuelle car trop bruité). Cette approche a permis de faire passer le corpus de 93 à 212 événements, tout en résolvant un déséquilibre géographique initial (deux départements sur cinq n'avaient aucun événement).

---

### C. Récupération des événements des sources retenues — `fetch_events.py`

Ce script interroge l'API OpenAgenda pour chacune des sources retenues, définies dans une liste `SOURCES` (uid de l'agenda + nom) :

| Source | uid | Rôle |
|---|---|---|
| Agenda Région Pays de la Loire | 16676449 | Agenda institutionnel régional généraliste |
| Unidivers Oui sortir | 14115607 | Agrégateur culturel Grand Ouest |
| Théâtre de Laval - CDN | 31651509 | Comble l'absence d'événements en Mayenne |
| Réseau des médiathèques & Archives du Mans | 48454528 | Comble l'absence d'événements en Sarthe |
| Département de la Vendée | 7894666 | Renforce la couverture en Vendée (filtré, voir D.3.b) |

Pour chaque source, seuls les événements **à venir** (`relative[]=upcoming`) sont récupérés, avec pagination automatique (curseur `after[]`) tant que l'API renvoie des résultats. Chaque événement est marqué d'un champ `_source_agenda` pour tracer sa provenance, et un déduplication par `uid` est appliquée pour éviter qu'un même événement ne soit compté deux fois s'il apparaît dans plusieurs sources. Le résultat agrégé est sauvegardé dans `data/raw_events.json`.

---

### D. Nettoyage et mise en forme des données — `preprocess_events.py`

#### 1. Nettoyage des données

La fonction `clean_text()` supprime les caractères de mise en forme markdown résiduels (`*`, `_`, `#`, `` ` ``) présents dans les descriptions, et normalise les espaces multiples en un seul espace.

#### 2. Extraction dans un format exploitable

La fonction `extract_event()` parcourt chaque événement brut et en extrait les champs utiles sous forme de dictionnaire structuré : titre, description, mots-clés, lieu (ville, département, région, code postal, coordonnées), dates de début/fin, et source d'origine. Ces dictionnaires sont ensuite assemblés en `DataFrame` pandas pour faciliter les traitements suivants.

Après extraction, plusieurs filtres successifs sont appliqués : suppression des événements sans titre ou sans date, déduplication par `uid`, exclusion des événements sans lieu physique (webinaires), et exclusion des événements situés hors de la région Pays de la Loire.

#### 3. Suppression des événements non culturels

**a. Liste noire, appliquée à l'ensemble du DataFrame**

Un ensemble de motifs regex (`NON_CULTURAL_PATTERNS`) cible les événements clairement hors périmètre : salons étudiants, forums d'orientation, offres d'emploi, Parcoursup, alternance, etc. Cette liste s'applique à toutes les sources, y compris "Département de la Vendée" — mais avec peu d'effet sur cette dernière, car ce type de contenu (orientation/emploi) y est marginal comparé à son propre bruit spécifique (voir b.).

**b. Liste blanche, réservée à "Département de la Vendée"**

Contrairement aux autres sources — des agendas institutionnels **dédiés à la culture** (théâtre, médiathèques), donc déjà à forte densité culturelle — l'agenda "Département de la Vendée" est un agenda **généraliste** (conseils municipaux, vide-greniers, forums santé, événements sportifs...). La liste noire seule y est insuffisante : environ 60% de son contenu brut est hors-sujet. Un filtre `CULTURAL_PATTERNS` (liste blanche de mots-clés culturels — théâtre, musique, exposition, spectacle, médiathèque, patrimoine, etc.) y est donc appliqué en complément, en exigeant qu'au moins un de ces termes soit présent dans le titre, la description ou les mots-clés de l'événement. Ce filtre n'est volontairement **pas** étendu aux autres sources : appliqué à des agendas déjà propres mais dont les titres n'utilisent pas toujours un vocabulaire générique (ex. titres d'œuvres comme *"Sous l'arbre"* ou *"L'Ecossaise"*), il ferait perdre des événements culturels légitimes (faux négatifs) sans gain de précision.

#### 4. Construction du champ `embedding_text`

La fonction `build_embedding_text()` construit, pour chaque événement, le texte qui sera effectivement vectorisé : titre, description, lieu (nom du lieu, ville, département) et mots-clés, concaténés avec un séparateur `|`. La date est volontairement **exclue** de ce champ : une date vectorisée n'apporte pas de signal sémantique utile à la recherche par similarité — elle est conservée à part, comme métadonnée structurée exploitable pour du filtrage déterministe au moment de la recherche.

#### 5. Écriture des fichiers de sortie

Le DataFrame final est sauvegardé sous deux formats redondants dans `data/` : `events_clean.csv` (pour inspection tabulaire rapide) et `events_clean.json` (utilisé par `vectorize_events.py`, car il conserve fidèlement le champ `embedding_text`).

---

### E. Tests de la qualité des données nettoyées — `test_data_quality.py`

Une suite de 9 tests `pytest` valide automatiquement la cohérence du corpus après chaque exécution de `preprocess_events.py` :

| Test | Objectif |
|---|---|
| `test_data_not_empty` | Le corpus nettoyé n'est pas vide |
| `test_no_missing_title` | Aucun événement sans titre |
| `test_no_missing_date` | Aucun événement sans date de début |
| `test_no_duplicate_events` | Aucun `uid` en double |
| `test_region_is_pays_de_la_loire` | Tous les événements sont bien situés en Pays de la Loire |
| `test_department_within_region` | Le département de chaque événement fait partie des 5 départements de la région |
| `test_events_are_upcoming_or_recent` | Aucun événement trop ancien (> 1 an) ni trop lointain dans le futur — marge de sécurité volontaire, le pipeline ne récupérant en pratique que des événements `upcoming` |
| `test_embedding_text_not_empty` | Le champ destiné à la vectorisation n'est jamais vide |
| `test_coordinates_are_valid` | Les coordonnées géographiques (latitude/longitude) sont dans des bornes plausibles |

Ces tests servent de garde-fou avant la vectorisation : un corpus qui échoue à l'un d'eux ne devrait pas être utilisé pour reconstruire l'index.

---

### F. Vectorisation des données — `vectorize_events.py`

#### 1. `load_documents()` — séparation métadonnées / texte vectorisé

Chaque événement de `events_clean.json` est transformé en objet `Document` LangChain, composé de deux parties distinctes : `page_content` (le champ `embedding_text`, seul élément réellement vectorisé) et `metadata` (tous les autres champs structurés — ville, département, dates, coordonnées... — conservés tels quels pour un filtrage déterministe ultérieur).

#### 2. `chunk_documents()` — découpage en chunks

Le texte de chaque document est découpé via un `RecursiveCharacterTextSplitter` (taille de chunk 800 caractères, chevauchement 100). Le découpage s'opère **document par document**, uniquement sur `page_content` — jamais sur les métadonnées, qui sont recopiées à l'identique sur chaque chunk généré à partir d'un même document. Les textes de ce corpus étant courts, le ratio chunks/documents reste proche de 1:1.

#### 3. `build_vectorstore()` — vectorisation et construction de l'index FAISS

Chaque chunk est transformé en vecteur numérique via le modèle d'embeddings Mistral (`mistral-embed`, 1024 dimensions), puis indexé dans une structure FAISS de type `IndexFlatL2` (recherche exacte par distance euclidienne — adaptée à un corpus de cette taille, à faire évoluer vers un index approximatif de type IVF/HNSW à plus grande échelle).

#### 4. `test_search()` — validation locale

Une recherche par similarité simple (sans filtrage géographique) est exécutée sur quelques requêtes de test, pour vérifier que l'index retourne des résultats pertinents avant de passer à l'intégration dans la chaîne RAG.

#### 5. Sauvegarde de l'index

L'index vectoriel construit est sauvegardé dans `vectorstore/faiss_index/` (fichiers `index.faiss` et `index.pkl`). Ce dossier n'est pas versionné dans Git — il est intégralement régénérable en relançant `vectorize_events.py` à partir de `events_clean.json`.

---

### G. Mise en place du système RAG — `rag_chain.py`

#### 1. Fonctionnement des 4 éléments de la chaîne

La chaîne RAG est construite avec la syntaxe LCEL (`|`) de LangChain, dans la fonction `build_rag_chain()`, qui assemble quatre éléments :

- **Le contexte** : une fonction `retrieve(question)`, écrite pour ce projet, qui interroge l'index FAISS via `vectorstore.similarity_search()`. Elle tente d'abord d'extraire un lieu mentionné dans la question (fonction `extract_location`, comparée aux départements/villes connus) ; si un lieu est détecté, la recherche est filtrée sur ce lieu (avec `fetch_k` égal à la taille totale du corpus, pour éviter qu'un filtrage par métadonnées ne rate des résultats valides à cause de la limite par défaut de FAISS) ; si aucun résultat ne correspond au filtre, une recherche non filtrée est faite en repli, avec instruction au LLM (via le prompt système) de ne pas présenter ces résultats comme correspondant au lieu demandé.
- **Le prompt** : un `ChatPromptTemplate` combinant un prompt système (instructions de comportement — notamment la mise en forme systématique des réponses en liste numérotée) et la question de l'utilisateur.
- **Le LLM** : `ChatMistralAI`, utilisant le modèle `mistral-small-latest`, chargé de générer la réponse en langage naturel à partir du contexte récupéré et de la question.
- **Le parseur de sortie** : `StrOutputParser()`, qui extrait la réponse texte brute de l'objet retourné par le LLM.

L'assemblage `{"context": retrieve, "question": ...} | prompt | llm | StrOutputParser()` produit un objet de type `RunnableSequence` : chaque maillon (y compris `retrieve`, une fonction Python classique, automatiquement convertie en `RunnableLambda`) est compatible avec l'interface `Runnable` de LangChain, ce qui permet de les enchaîner avec l'opérateur `|` et de les exécuter uniformément via `.invoke()`.

#### 2. Tests locaux

Le script exécute, en fin d'exécution, une série de 4 questions de test représentatives (recherche par thématique, par lieu, avec et sans résultat disponible), pour valider le comportement global de la chaîne avant son intégration dans une interface utilisateur.

---

### H. Chat interactif — `chat_interactive.py`

#### 1. Réutilisation de `build_rag_chain()`

Ce script importe et appelle `build_rag_chain()` depuis `rag_chain.py`, réutilisant ainsi telle quelle la chaîne RAG déjà validée, sans dupliquer sa logique.

#### 2. Boucle interactive

Une boucle infinie lit les questions saisies par l'utilisateur au clavier, les transmet à `chain.invoke()`, et affiche la réponse — jusqu'à ce que l'utilisateur saisisse une commande de sortie (ex. `quit`/`exit`), qui rompt la boucle.

---

## Partie 3 — Utilisation de l'application

### A. Reproduction de l'environnement

#### 1. Environnement virtuel et dépendances

```bash
# Depuis la racine du projet
python -m venv projet11-env
source projet11-env/bin/activate        # Linux/WSL
# projet11-env\Scripts\activate          # Windows (hors WSL)

pip install -r requirements.txt
```

Créer un fichier `.env` à la racine du projet, contenant :

```
OPENAGENDA_API_KEY=votre_clé_openagenda
MISTRAL_API_KEY=votre_clé_mistral
```

#### 2. Construction des données, tests et index

Toutes les commandes s'exécutent **depuis la racine du projet** :

```bash
python scripts/fetch_events.py        # Récupération des événements bruts → data/raw_events.json
python scripts/preprocess_events.py   # Nettoyage et filtrage → data/events_clean.csv / .json
pytest tests/test_data_quality.py -v  # Vérification de la qualité des données
python scripts/vectorize_events.py    # Construction de l'index → vectorstore/faiss_index/
```

#### 3. Test du système RAG

```bash
python scripts/rag_chain.py           # Exécute 4 questions de test prédéfinies
python scripts/chat_interactive.py    # Ouvre une session de questions/réponses en continu
```

---

### B. Fonctionnement global de l'application

Quand un utilisateur pose une question, quatre grandes étapes se succèdent :

1. **Compréhension du lieu** : le système tente de détecter si la question mentionne un lieu précis (ville, département), pour orienter la recherche.
2. **Recherche par similarité** : la question est elle-même transformée en vecteur numérique (embedding), puis comparée mathématiquement à tous les vecteurs d'événements déjà indexés, pour identifier les plus proches sémantiquement — c'est le principe de la **recherche vectorielle**, indépendante des mots exacts employés (une question sur "un opéra" peut ainsi remonter un événement titré "Carmen").
3. **Construction du contexte** : les événements les plus pertinents trouvés sont formatés en texte et injectés dans un prompt, aux côtés de la question — c'est le cœur de l'approche **RAG** : le modèle de langage ne répond pas "de mémoire", mais à partir d'informations réelles et à jour qu'on lui fournit à la volée.
4. **Génération de la réponse** : un grand modèle de langage (Mistral) rédige une réponse en langage naturel, structurée, à partir de ce contexte — sans avoir accès à Internet ni à aucune autre source que les événements qui lui ont été transmis.

Cette architecture garantit que le système ne peut recommander que des événements réellement présents dans la base — il ne peut pas "inventer" un événement, et il indique explicitement quand il n'a rien de pertinent à proposer.

---

## Conclusion

### Bilan

Le POC démontre la faisabilité d'un système de recommandation d'événements culturels reposant entièrement sur des données réelles (OpenAgenda), avec une chaîne de traitement reproductible de bout en bout : collecte multi-sources, nettoyage et filtrage (y compris un filtrage culturel adapté à l'hétérogénéité des sources), vectorisation, et génération de réponses contextualisées via LangChain et Mistral.

Le corpus final compte **212 événements culturels**, répartis sur les 5 départements de la région Pays de la Loire (de 22 à 55 événements par département), après un travail itératif d'équilibrage géographique des sources. Le système répond correctement à des questions thématiques, géographiques, et signale honnêtement l'absence de résultat quand aucun événement ne correspond à la demande.

**Limites assumées du POC** :
- Le corpus reste de taille modeste (quelques centaines d'événements), ce qui limite la richesse des réponses possibles sur des requêtes très spécifiques.
- Les déduplications inter-agendas (un même événement physique publié sous des `uid` différents sur plusieurs sources) ne sont pas détectées.
- L'index FAISS utilisé (`IndexFlatL2`, recherche exacte) n'est pas conçu pour passer à l'échelle sur un corpus de plusieurs dizaines de milliers d'événements.
- La dépendance `langchain-community` utilisée pour l'intégration FAISS est en cours de dépréciation par son éditeur.

### Perspectives d'évolution

Pour passer de ce POC régional à un système national correspondant à l'ambition de Puls-Events, plusieurs pistes se dégagent :

- **Extension géographique** : généraliser la stratégie multi-sources (agendas institutionnels culturels dédiés, plutôt que des agendas généralistes nécessitant un filtrage renforcé) à l'ensemble des régions françaises, potentiellement en activant l'endpoint OpenAgenda expérimental de "lecture transverse" (`/v2/events`), qui permettrait d'interroger tous les agendas de la plateforme en une seule requête.
- **Passage à l'échelle de l'index** : migrer vers un index FAISS approximatif (IVF, HNSW) pour maintenir des temps de réponse raisonnables sur un corpus significativement plus volumineux.
- **Déduplication inter-sources** : mettre en place une détection de doublons basée sur la similarité (titre, date, lieu) plutôt que sur le seul `uid`, pour fusionner les événements identiques publiés sur plusieurs agendas.
- **Migration technique** : remplacer `langchain-community` par les paquets d'intégration autonomes recommandés par LangChain, pour anticiper la dépréciation annoncée.
- **Évaluation continue** : construire un jeu de questions/réponses annotées de référence pour mesurer objectivement la qualité des réponses du système dans le temps, notamment à mesure que le corpus grandit.
- **Interface utilisateur** : faire évoluer le chat interactif en ligne de commande vers une interface web ou une intégration directe dans les canaux de communication de Puls-Events (site, application).
