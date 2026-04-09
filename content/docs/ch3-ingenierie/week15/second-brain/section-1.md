---
title: "1. L'Infrastructure de la Donnée"
weight: 1
---

# Section 1 : L'Infrastructure de la Donnée – ETL et RAG Avancé

Bonjour à toutes et à tous ! J'espère que vous avez bien en tête la carte globale de l'assistant que nous avons tracée dans l'introduction. Aujourd'hui, nous allons nous salir les mains dans la "soute" du navire. 

> [!IMPORTANT]
**Je dois insister :** l'intelligence d'un assistant ne dépend pas de la beauté de son interface, mais de la pureté de sa donnée. Dans cette première section technique, nous allons voir comment transformer des notes Notion désordonnées en un "carburant" de haute précision pour notre IA. Nous allons parler d'**ETL**, de **scoring de qualité** et de **RAG Parent-Child**. Respirez, car nous construisons aujourd'hui les fondations de la mémoire de votre assistant !

---

## 1.1 L'Architecture FTI : La Rigueur Industrielle

Dans notre cours, nous avons vu des modèles isolés. Mais en production, le projet **Second Brain** adopte une architecture **FTI** (Feature, Training, Inference). 

**L'intuition de l'expert** : Pourquoi séparer les pipelines ?

<a id="fig-15-4"></a>

{{< bookfig src="15_4.png" week="15" >}}


Comme le montre la Figure 15-4, la **Feature Pipeline** (notre section d'aujourd'hui) est le processus qui tourne en continu ou périodiquement. Elle prépare les données sans attendre que l'utilisateur pose une question. 
*   **Avantage technique** : Cela réduit la latence au moment de l'inférence. L'IA n'a pas à chercher dans vos notes en temps réel ; elle interroge un index déjà optimisé. 
*   **Lien avec la Semaine 13** : C'est le fondement du déploiement responsable. On ne bricole pas la donnée à la volée, on la certifie avant l'usage.

---

## 1.2 Le Pipeline ETL : Du Chaos Notion à la Structure JSON

Le point de départ est votre base de connaissances Notion. C'est l'application directe de notre **Semaine 2** sur la tokenisation et la gestion des documents. 

### A. L'extraction chirurgicale
Le projet utilise un `NotionDocumentClient` (`src/second_brain_offline/infrastructure/notion/page.py`). Ce client ne se contente pas de copier le texte ; il décompose les "blocs" Notion (titres, paragraphes, listes) pour reconstruire une structure Markdown propre.

```python
# [SOURCE: src/second_brain_offline/infrastructure/notion/page.py#L96]
def __parse_blocks(self, blocks: list[dict], depth: int = 0) -> tuple[str, list[str]]:
    content = ""
    urls = []
    for block in blocks:
        block_type = block.get("type")
        if block_type in {"heading_1", "heading_2", "heading_3"}:
            # On préserve la hiérarchie pour le futur chunking (Section 6.3)
            content += f"# {self.__parse_rich_text(block[block_type].get('rich_text', []))}\n\n"
        elif block_type == "paragraph":
            content += f"{self.__parse_rich_text(block[block_type].get('rich_text', []))}\n"
    # ...
    return content, urls
```

> [!IMPORTANT]
**Je dois insister :** Notez que le client extrait également les `child_urls`. C'est le début du "volant d'inertie" de la donnée : l'assistant pourra aller crawler les liens que vous avez enregistrés dans vos notes !

---

## 1.3 L'Audit de Qualité : Le LLM comme Juge de Paix

> [!WARNING]
Mes chers étudiants, un LLM ne doit pas tout lire.

Si vous indexez des pages d'erreur 404 ou des listes de cookies, vous allez "polluer" la mémoire de votre assistant. 

Le projet implémente un concept avancé de la **Section 9.3** : le **LLM-as-a-Judge**. Avant d'entrer dans la base de données, chaque document passe devant un expert (GPT-4o-mini) qui lui attribue un score de qualité.

```python
# [SOURCE: src/second_brain_offline/application/agents/quality.py#L42]
SYSTEM_PROMPT_TEMPLATE = """You are an expert judge tasked with evaluating the quality of a given DOCUMENT.
Guidelines:
1. Evaluate the DOCUMENT based on reliable information.
2. Evaluate that the DOCUMENT contains relevant information and not only links or error messages.
...
Return only the score in JSON format: {"score": <0.0 to 1.0>}"""
```

**Analyse de l'ingénieur** : 
*   **Score < 0.3** : Le document est rejeté (ex: une page contenant uniquement "Veuillez accepter les cookies").
*   **Score > 0.8** : Le document est considéré comme une pépite de savoir.
**C'est une étape d'ingénierie cruciale :** on utilise l'IA pour nettoyer les données qui serviront à... l'IA. C'est l'auto-curation sémantique.

---

## 1.4 La Stratégie de Chunking "Parent-Child"

C'est ici que nous appliquons la science de la **Section 6.3**. Dans un RAG classique, si on découpe trop petit, on perd le contexte. Si on découpe trop grand, on noie le signal dans le bruit.

Le projet utilise une architecture de **Parent Document Retrieval** (`src/second_brain_online/application/rag/retrievers.py`).

**Explication du mécanisme** :
1.  **Child Chunks (200 tokens)** : On découpe le document en petits morceaux très précis. Ce sont eux que l'on transforme en vecteurs (Embeddings) et que l'on indexe dans FAISS.
2.  **Parent Chunks (800 tokens)** : Pour chaque petit morceau, on garde en mémoire le paragraphe global auquel il appartient.
3.  **Le miracle du Retrieval** : Quand l'utilisateur pose une question, on cherche le petit morceau le plus proche (Précision), mais on donne au LLM le gros morceau correspondant (Contexte).

> [!IMPORTANT]
**Je dois insister :** C'est la solution technique au problème du "Lost in the Middle" étudié en **Section 14.4**. On indexe pour la recherche, mais on récupère pour la compréhension.

---

## 1.5 Ingestion dans MongoDB : La Mémoire Industrielle

Contrairement aux labs où nous utilisions de simples fichiers `.json`, le projet utilise **MongoDB Atlas** avec ses capacités de recherche vectorielle.

```python
# [SOURCE: src/second_brain_online/application/rag/retrievers.py#L49]
vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=settings.MONGODB_URI,
    namespace=f"{settings.MONGODB_DATABASE_NAME}.rag",
    text_key="chunk",
    embedding_key="embedding",
    relevance_score_fn="dotProduct", # Section 6.2 !
)
```

> [!WARNING]
**Attention : erreur fréquente ici !** Beaucoup d'étudiants oublient de créer l'index de recherche sur MongoDB. Le projet automatise cela via la classe `MongoDBIndex` (`indexes.py`). Sans cet index, votre recherche sémantique ne sera qu'une simple recherche textuelle lente.

---

## 1.6 Éthique et Confidentialité de la Donnée

> [!CAUTION]
Mes chers étudiants, nous touchons ici à la vie privée.

Dans le fichier `document.py`, vous trouverez une méthode capitale : `.obfuscate()`. 

```python
# [SOURCE: src/second_brain_offline/domain/document.py#L86]
def obfuscate(self) -> "Document":
    """Create an obfuscated version of this document by modifying in place."""
    self.metadata = self.metadata.obfuscate()
    self.id = self.metadata.id
    return self
```

> [!NOTE]
**Pourquoi est-ce vital ?** Lorsque vous envoyez vos données Notion pour le fine-tuning ou pour l'évaluation, vous devez protéger les identifiants réels. Le projet montre comment masquer les IDs et les URLs sensibles avant tout traitement externe. C'est l'application directe de notre **Section 13.3** sur le RGPD et la protection des données.

---

## Synthèse

Vous venez de voir comment nous passons d'une note griffonnée sur Notion à un vecteur sécurisé, filtré par un juge LLM et indexé dans une base de données professionnelle.

> [!TIP]
**Le message à retenir** : Le RAG ne commence pas par la question de l'utilisateur. Il commence par la **Feature Pipeline**. Plus votre pipeline ETL est robuste, moins votre assistant aura besoin de "réfléchir" pour trouver la vérité.

Dans la **Section 2**, nous allons voir comment utiliser ces données propres pour créer un expert. Nous allons étudier le pipeline de **Distillation** et le **Fine-tuning** de notre modèle Llama-3.1. Prêts à passer de la donnée à l'expertise ?