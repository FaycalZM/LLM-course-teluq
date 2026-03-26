---
title: "8.4 Vérification et contrôle"
weight: 5
---

## Le rempart contre l'imprévisibilité : L'IA au service de la production
Bonjour à toutes et à tous ! Nous arrivons à la dernière étape de notre semaine sur l'ingénierie des prompts. Prenez un instant pour réaliser le chemin parcouru : nous avons appris à structurer nos demandes (section 8.1), à guider l'IA par l'exemple (section 8.2) et à débloquer son raisonnement logique (section 8.3). Mais une question cruciale demeure : comment faire pour que l'IA ne devienne pas le "maillon faible" de votre système informatique ? 
> [!IMPORTANT]
🔑 **Je dois insister :** une réponse fluide et intelligente ne sert à rien si elle fait planter votre application parce qu'il manque une virgule dans un fichier JSON ou si elle contient une information confidentielle interdite. 

Aujourd'hui, nous allons apprendre à mettre des "garde-fous" à la machine. Bienvenue dans l'ère de l'**IA contrôlée et vérifiée**.


## Pourquoi valider la sortie ? Les trois risques majeurs
Dans un environnement professionnel, on ne peut pas se contenter d'un "ça a l'air de marcher". Voici les trois piliers de vérification indispensables pour tout ingénieur en LLM :
1.  **La structure (Format)** : Si votre logiciel attend un format JSON pour mettre à jour une base de données et que l'IA répond "Voici votre JSON : { ... }", la phrase d'introduction fera échouer votre code.
2.  **La validité (Logic)** : Même si le format est bon, les valeurs doivent être cohérentes. Une IA qui génère un personnage de jeu avec une "force" de 1 000 000 alors que le maximum est 100 brise l'équilibre de votre application.
3.  **L'éthique et la sécurité** : C'est le point non-négociable. Nous devons nous assurer que la sortie ne contient pas de données personnelles (PII), de propos haineux ou de failles de sécurité (injections de code).


## Technique 1 : La boucle de vérification par les pairs (Self-Correction)
La méthode la plus simple consiste à utiliser un LLM pour en vérifier un autre, ou pour se vérifier lui-même. C'est ce qu'illustre la **Figure 8-12**.

{{< bookfig src="152.png" week="08" >}}

Décortiquons le contenu de cette figure :
*   **Le cycle itératif** : La figure montre un premier passage où l'IA génère une réponse (A). Cette réponse est ensuite passée à un "vérificateur" (qui peut être le même modèle avec un prompt différent).
*   **Les règles (Guardrails)** : Le vérificateur a une liste de critères (ex: "Le texte est-il en JSON ? Est-il courtois ?"). 
*   **La rétroaction** : Si une règle est enfreinte, le vérificateur renvoie une erreur à l'IA initiale ("Tu as oublié de fermer l'accolade, recommence").

> [!TIP]
🔑 **Mon intuition :** C'est comme un auteur qui se relit ou qui demande à un éditeur de corriger son manuscrit. Cela améliore la qualité, mais attention : cela double votre consommation de tokens et votre temps de réponse !


## Technique 2 : L'échantillonnage contraint (Constrained Sampling)
C'est ici que l'ingénierie devient "magique" et extrêmement puissante. Au lieu de laisser l'IA générer du texte et de corriger l'erreur après coup, nous allons **intervenir directement dans son cerveau** (au niveau des probabilités) pour l'empêcher physiquement de faire une erreur de syntaxe.

Regardons les **Figures 8-13 et 8-14** :
*   **Figure 8-13 : Génération de pièces manquantes** : On donne au modèle un "moule" (un template JSON avec des trous). On force le modèle à ne remplir que les trous. Le modèle n'a pas le choix, il ne peut pas ajouter de texte inutile autour.
{{< bookfig src="153.png" week="08" >}}

*   **Figure 8-14 : Constrained Sampling par masque de probabilité** : C'est la technique la plus avancée. Imaginez que le modèle doive choisir le prochain token. Normalement, il a 50 000 choix possibles. Si nous savons que l'étape suivante dans un JSON doit être un chiffre, nous appliquons un "masque" : nous mettons la probabilité de tous les mots du dictionnaire à zéro, sauf pour les chiffres de 0 à 9.
{{< bookfig src="154.png" week="08" >}}
 
> [!NOTE]
🔑 **Je dois insister :** Avec cette technique, le modèle ne peut mathématiquement PAS produire une erreur de format. Il devient un moteur de données déterministe au sein d'une architecture probabiliste.


## L'usage des Grammaires (GBNF)
Pour réaliser ce tour de force avec des modèles locaux (comme ceux que nous utilisons sur Colab), nous utilisons souvent le format **GBNF** (*Guidance/GGML BNF*). C'est une façon de définir une "recette" syntaxique que l'IA doit suivre.
*   *Analogie* : C'est comme un rail de chemin de fer. Le train (l'IA) peut aller vite et être puissant, mais il ne peut pas sortir des rails que vous avez posés.


## Laboratoire de code : Forcer une sortie JSON parfaite (Colab T4)
Nous allons utiliser la bibliothèque `llama-cpp-python` qui permet de charger des modèles compressés (GGUF) et de leur imposer une structure JSON stricte. 
> [!WARNING]
⚠️ **Attention !** Assurez-vous d'avoir redémarré votre session Colab pour libérer la VRAM.

```python
# Installation des outils de contrôle
# !pip install llama-cpp-python 

from llama_cpp import Llama
import json

# 1. Chargement d'un modèle compact (TinyLlama ou Phi-3 GGUF)
llm = Llama(
    model_path="path_to_your_model.gguf", # Remplacez par votre lien de téléchargement
    n_gpu_layers=-1, # On utilise tout le GPU T4 !
    n_ctx=2048
)

# 2. Définition de la tâche
prompt = "Create a health report for a patient named Sarah. Include age, heart_rate, and status."

# 3. GÉNÉRATION CONTRAINTE (L'arme secrète)
# On demande explicitement un 'json_object' au moteur d'inférence
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt}],
    response_format={
        "type": "json_object",
        "schema": { # On définit le schéma attendu
            "type": "object",
            "properties": {
                "age": {"type": "number"},
                "heart_rate": {"type": "number"},
                "status": {"type": "string"}
            },
            "required": ["age", "heart_rate", "status"]
        }
    },
    temperature=0.7 # On peut garder de la créativité sur le contenu !
)

# 4. Extraction et validation
json_output = json.loads(response["choices"][0]["message"]["content"])
print(f"Sortie validée : {json_output}")
```

> [!NOTE]
🔑 **Note** : Remarquez que nous avons gardé une température de 0.7. Le modèle est libre d'inventer le "status" (ex: "Stable", "Excellente santé"), mais il est **obligé** de le mettre dans la bonne case du JSON. C'est l'équilibre parfait entre l'imagination de l'IA et la rigueur de l'informatique classique.


## Frameworks de contrôle : Guardrails, Guidance et Outlines
Dans le monde professionnel, vous n'allez pas tout coder à la main. Il existe des bibliothèques dédiées à la sécurité et au contrôle des LLM :
*   **Guardrails AI** : Permet de définir des "Rails" en XML pour vérifier les sorties (ex: "Vérifie qu'il n'y a pas d'insultes").
*   **Guidance (Microsoft)** : Un langage de programmation qui entrelace le texte humain et le contrôle de l'IA.
*   **Outlines** : Une bibliothèque spécialisée dans l'échantillonnage contraint pour garantir des sorties JSON ou Regex à 100%.


## Éthique et Responsabilité : L'illusion du contrôle total

> [!CAUTION]
⚠️ Ne vous laissez pas bercer par un sentiment de sécurité trompeur.

Ce n'est pas parce que votre IA sort un JSON parfait que les informations à l'intérieur sont vraies.
1.  **Hallucinations structurées** : L'IA peut vous donner un JSON très propre contenant des faits totalement inventés (ex: un faux diagnostic médical). 
>> [!TIP]
🔑 **Règle d'or :** La structure n'est pas la vérité.
2.  **Biais de formatage** : Parfois, forcer un format (ex: n'accepter que "Oui" ou "Non") empêche l'IA d'exprimer une nuance importante ou une incertitude nécessaire. C'est une forme de censure technique qui peut mener à des décisions erronées. 
3.  **Sur-confiance** : Un développeur qui voit une sortie JSON a tendance à lui faire plus confiance qu'à un texte libre. C'est un piège psychologique. Gardez toujours un esprit critique.


🔑 **Mon message** : Le contrôle est le pont qui permet aux LLM de sortir des laboratoires pour entrer dans les usines, les hôpitaux et les banques. 

> En maîtrisant ces techniques, vous ne construisez plus seulement des gadgets, vous construisez des systèmes fiables. Mais n'oubliez jamais : vous êtes le chef d'orchestre, et la machine n'est que l'instrument. La responsabilité finale de la décision appartient toujours à l'humain.

---
Nous avons terminé notre immense semaine sur l'ingénierie des prompts ! Vous savez maintenant comment parler à l'IA, comment la faire raisonner, et comment la tenir en laisse pour qu'elle respecte vos contraintes. C'est un arsenal de compétences qui fait de vous des professionnels recherchés. La semaine prochaine, nous allons attaquer un défi encore plus grand : comment donner une mémoire infinie à votre IA grâce au **RAG** (*Retrieval-Augmented Generation*). Préparez-vous, car c'est là que l'IA rencontre vos propres données ! Mais d'abord, place au laboratoire final de la semaine ➡️.